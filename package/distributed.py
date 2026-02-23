"""
dino_loader.distributed
=======================
Topology detection, NCCL configuration, and SLURM process-group bootstrap.

No imports from other dino_loader submodules — this module is intentionally
a leaf so it can be imported (and NCCL configured) before anything else.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Topology
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterTopology:
    """Immutable description of the hardware topology, detected at startup."""
    # NVLink
    nvlink_gen:              int  = 4    # 4 = H100, 5 = B200/GB200
    nvlink_lanes_per_gpu:    int  = 9    # 18 → NVL72
    gpus_per_nvlink_domain:  int  = 8

    # Memory interconnect
    is_grace_blackwell:      bool = False  # NVLink-C2C unified memory
    has_pcie:                bool = True   # False for NVLink-C2C racks

    # Network
    has_infiniband:          bool = True
    has_sharp:               bool = False
    has_nvlink_sharp:        bool = False  # NVLS: in-switch reduction on NVLink fabric

    # CUDA runtime
    cuda_major:              int  = 12
    cuda_minor:              int  = 0

    @property
    def is_nvl72(self) -> bool:
        return self.nvlink_lanes_per_gpu >= 18

    @property
    def cuda_ge_128(self) -> bool:
        return (self.cuda_major, self.cuda_minor) >= (12, 8)

    @property
    def label(self) -> str:
        if self.is_grace_blackwell:
            return "GB200-NVL72" if self.is_nvl72 else "GB200-PCIe"
        if self.nvlink_gen >= 5:
            return "B200-PCIe"
        return "H100"


def detect_topology(force: Optional[str] = None) -> ClusterTopology:
    """
    Probe hardware and return a ClusterTopology.

    Parameters
    ----------
    force : "nvl72" | "pcie" | None
        Override auto-detection for testing or misconfigured nodes.
    """
    topo = ClusterTopology()

    # CUDA version
    try:
        parts = torch.version.cuda.split(".")
        topo.cuda_major, topo.cuda_minor = int(parts[0]), int(parts[1])
    except Exception:
        log.warning("Could not determine CUDA version; assuming 12.0")

    # GPU model → NVLink generation
    try:
        names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip().splitlines()
        gpu_name = names[0] if names else ""
        if any(k in gpu_name for k in ("B200", "GB200", "Blackwell")):
            topo.nvlink_gen = 5
        if "GB200" in gpu_name:
            topo.is_grace_blackwell = True
    except Exception:
        log.warning("nvidia-smi name query failed; NVLink gen assumed 4 (H100)")

    # Active NVLink lanes → detect NVL72 domain
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "nvlink", "--status", "-i", "0"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        topo.nvlink_lanes_per_gpu = len(re.findall(r"\bActive\b", out))
        if topo.is_nvl72:
            topo.gpus_per_nvlink_domain = 72
            topo.has_pcie = False
            topo.has_infiniband = False  # intra-rack via NVLink; IB only inter-rack
    except Exception:
        log.warning("NVLink lane count detection failed; assuming 9 (non-NVL72)")

    # NVLink-C2C platform sysfs
    c2c = Path("/sys/bus/platform/devices")
    if c2c.exists():
        for p in c2c.iterdir():
            if "c2c" in p.name.lower():
                topo.is_grace_blackwell = True
                topo.has_pcie = False
                break

    # InfiniBand
    ib = Path("/sys/class/infiniband")
    topo.has_infiniband = ib.exists() and any(True for _ in ib.iterdir())

    # SHARP (IB-level)
    if topo.has_infiniband:
        try:
            mods = Path("/proc/modules").read_text()
            topo.has_sharp = "ib_sharp" in mods
        except Exception:
            pass

    # NVLS (NVLink-level switch reduction, NVLink gen ≥ 4)
    topo.has_nvlink_sharp = topo.nvlink_gen >= 4

    # Override
    if force == "nvl72":
        topo.nvlink_lanes_per_gpu   = 18
        topo.gpus_per_nvlink_domain = 72
        topo.has_infiniband         = False
        topo.has_pcie               = False
    elif force == "pcie":
        topo.nvlink_lanes_per_gpu   = 9
        topo.gpus_per_nvlink_domain = 8
        topo.has_pcie               = True

    log.info("Detected topology: %s (NVLink lanes=%d, IB=%s, SHARP=%s, NVLS=%s, CUDA=%d.%d)",
             topo.label, topo.nvlink_lanes_per_gpu,
             topo.has_infiniband, topo.has_sharp, topo.has_nvlink_sharp,
             topo.cuda_major, topo.cuda_minor)
    return topo


# ══════════════════════════════════════════════════════════════════════════════
# NCCL configuration
# ══════════════════════════════════════════════════════════════════════════════

def configure_nccl(topo: ClusterTopology) -> None:
    """
    Set NCCL environment variables for the detected topology.
    MUST be called before torch.distributed.init_process_group().

    Design decisions
    ────────────────
    NVL72 / Grace-Blackwell:
      - Disable IB intra-rack (NVLink is the fabric).
      - NCCL_ALGO=Auto: NCCL 2.21+ understands NVL72 switch topology and
        selects Ring/Tree/NVLS per message size better than we can hardcode.
      - NCCL_NVLS_ENABLE=1: activate in-switch NVLink reductions.
      - NCCL_PROTO=LL128: lower latency than Simple on a unified NVLink domain.
      - NCCL_SHARP_ENABLE=0: SHARP requires IB; not applicable.

    B200 PCIe / H100 (standard multi-node with IB):
      - NCCL_ALGO=Auto: NVLink 5.0 shifts the Ring/Tree message-size crossover
        vs NVLink 4.0; hardcoding Tree as in v1 is wrong for B200.
      - NCCL_SHARP_ENABLE=1 if SHARP detected.
      - Wider buffers and more channels for NVLink 5.0.
    """
    env: dict[str, str] = {
        # Common
        "NCCL_P2P_LEVEL":        "NVL",
        "NCCL_SHM_DISABLE":      "0",
        "NCCL_ALGO":             "Auto",
        "NCCL_BUFFSIZE":         str(8 * 1024 * 1024),   # 8 MB
        "NCCL_NCHANNELS_PER_NET": "8" if topo.nvlink_gen >= 5 else "4",
        "NCCL_NVLS_ENABLE":      "1" if topo.has_nvlink_sharp else "0",
    }

    if topo.is_nvl72 or topo.is_grace_blackwell:
        env.update({
            "NCCL_IB_DISABLE":   "1",
            "NCCL_NET_GDR_LEVEL": "0",
            "NCCL_SHARP_ENABLE": "0",
            "NCCL_PROTO":        "LL128",
        })
    else:
        env.update({
            "NCCL_NET_GDR_LEVEL": "5",
            "NCCL_PROTO":         "Simple,LL128",
            "NCCL_IB_TIMEOUT":    "23",
            "NCCL_IB_RETRY_CNT":  "7",
            "NCCL_IB_GID_INDEX":  "3",
            "NCCL_SHARP_ENABLE":  "1" if topo.has_sharp else "0",
        })

    for k, v in env.items():
        if k not in os.environ:   # never override explicit user settings
            os.environ[k] = v

    log.info("NCCL configured for %s topology", topo.label)


# ══════════════════════════════════════════════════════════════════════════════
# SLURM bootstrap
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DistribEnv:
    rank:             int
    world_size:       int
    local_rank:       int
    local_world_size: int
    topology:         ClusterTopology


def slurm_init(force_topology: Optional[str] = None) -> DistribEnv:
    """
    Initialise torch.distributed from SLURM env vars, with topology-aware
    NCCL configuration.  Returns a DistribEnv describing this process.

    Must be called before any CUDA or distributed operation.
    """
    try:
        rank             = int(os.environ["SLURM_PROCID"])
        world_size       = int(os.environ["SLURM_NTASKS"])
        local_rank       = int(os.environ["SLURM_LOCALID"])
        local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    except KeyError as e:
        raise RuntimeError(
            f"SLURM environment variable {e} not set. "
            "Are you running inside a SLURM allocation?"
        ) from e

    # Resolve master address from first node in allocation
    try:
        node_list = os.environ["SLURM_NODELIST"]
        master = subprocess.check_output(
            ["scontrol", "show", "hostnames", node_list],
            text=True, timeout=10,
        ).split()[0]
    except Exception as exc:
        raise RuntimeError("Could not resolve SLURM master node address") from exc

    os.environ.setdefault("MASTER_ADDR", master)
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"]       = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    topo = detect_topology(force=force_topology)
    configure_nccl(topo)   # must happen before init_process_group

    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    log.info("Process group initialised: rank %d / %d, local %d / %d",
             rank, world_size, local_rank, local_world_size)

    return DistribEnv(
        rank=rank, world_size=world_size,
        local_rank=local_rank, local_world_size=local_world_size,
        topology=topo,
    )
