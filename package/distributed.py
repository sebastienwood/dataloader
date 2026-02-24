"""
dino_loader.distributed
=======================
Topology detection, NCCL configuration, and SLURM process-group bootstrap.

No imports from other dino_loader submodules — this module is intentionally
a leaf so it can be imported (and NCCL configured) before anything else.

Changes from previous version (intern review)
----------------------------------------------
ACCEPTED
  [A-1] torch.cuda.get_device_name() replaces nvidia-smi subprocess for GPU
        model detection: zero fork overhead vs 200-500 ms for nvidia-smi.
  [A-2] torch.version.cuda replaces nvidia-smi subprocess for CUDA version.
  [A-3] NCCL_SOCKET_IFNAME network fencing: prevents NCCL from accidentally
        binding to a 1 GbE management interface and achieving catastrophically
        low all-reduce bandwidth.  Made configurable via LoaderConfig.
  [A-4] dist.is_initialized() guard: prevents crash if slurm_init is called
        twice (notebook restart, unit tests).
  [A-5] verify_interconnect() post-init health check: concept accepted with
        corrected bandwidth formula, sufficient warmup, and softened thresholds.

REJECTED / FIXED
  [R-1] MASTER_ADDR resolution broken for non-rank-0 processes.
        The intern's code only ran scontrol on local_rank==0 then relied on
        "srun usually propagates this" — it does not.  MASTER_ADDR is a
        PyTorch convention, not a SLURM variable.  Reverted to all-ranks
        resolution: every process runs scontrol independently; it is
        idempotent and cheap relative to the training run.
  [R-2] verify_interconnect bandwidth formula had wrong units.
        size_mb * 8 / 1024 gives Gbits/s but thresholds (300, 150) were
        intended as GB/s — off by 8x, making the check always pass.
        Fixed to GB/s throughout with correct ring-correction factor.
        Increased warmup from 1 to 20 iterations (cold NCCL is always slow).
        Demoted the threshold breach to WARNING (not a fatal error at step 0).
  [R-3] sysfs NVLink probe path /sys/class/nvidia-nvlink/gpu0/link* does not
        exist on standard Linux NVIDIA driver installations.  With a missing
        path the intern's code silently falls through, always setting
        nvlink_lanes_per_gpu=9 and never detecting NVL72.  Reverted to the
        nvidia-smi nvlink --status approach which is correct, with the
        subprocess cost paid once at startup.
  [R-4] NCCL_SHARP_ENABLE missing from NVL72/Grace-Blackwell branch.
        Without it, NCCL attempts SHARP initialisation even on NVL72 nodes
        (which have no InfiniBand), causing slow startup and log warnings.
        Explicit NCCL_SHARP_ENABLE=0 restored for that branch.
  [R-5] Topology log line removed by intern: essential for debugging cluster
        issues on 288-GPU jobs.  Restored.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

    Detection strategy
    ------------------
    - CUDA version and GPU model use torch.cuda directly [A-1, A-2]: zero
      subprocess overhead vs 200-500 ms for nvidia-smi.
    - NVLink lane count still uses nvidia-smi nvlink --status [R-3]: the
      sysfs path /sys/class/nvidia-nvlink/gpu0/link* proposed by the intern
      does not exist on standard driver installations; nvidia-smi is slower
      but correct.  This probe runs once at startup.
    - All other probes use sysfs (IB, SHARP, C2C).
    """
    topo = ClusterTopology()

    # ── CUDA version [A-2] ───────────────────────────────────────────────────
    try:
        parts = torch.version.cuda.split(".")
        topo.cuda_major, topo.cuda_minor = int(parts[0]), int(parts[1])
    except Exception:
        log.warning("Could not determine CUDA version; assuming 12.0")

    # ── GPU model → NVLink generation [A-1] ─────────────────────────────────
    # torch.cuda.get_device_name() avoids the 200-500 ms nvidia-smi fork.
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if any(k in gpu_name for k in ("B200", "GB200", "Blackwell")):
                topo.nvlink_gen = 5
            if "GB200" in gpu_name:
                topo.is_grace_blackwell = True
        else:
            log.warning("CUDA not available; GPU model detection skipped")
    except Exception:
        log.warning("GPU model detection failed; NVLink gen assumed 4 (H100)")

    # ── Active NVLink lanes → detect NVL72 domain [R-3] ─────────────────────
    # nvidia-smi is the only reliable cross-driver path for this information.
    # The sysfs alternative (/sys/class/nvidia-nvlink/gpu0/link*) does not
    # exist on standard NVIDIA driver installations.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "nvlink", "--status", "-i", "0"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        topo.nvlink_lanes_per_gpu = len(re.findall(r"\bActive\b", out))
        if topo.is_nvl72:
            topo.gpus_per_nvlink_domain = 72
            topo.has_pcie               = False
            topo.has_infiniband         = False  # intra-rack via NVLink; IB only inter-rack
    except Exception:
        log.warning("NVLink lane count detection failed; assuming 9 (non-NVL72)")

    # ── NVLink-C2C platform sysfs ────────────────────────────────────────────
    c2c = Path("/sys/bus/platform/devices")
    if c2c.exists():
        for p in c2c.iterdir():
            if "c2c" in p.name.lower():
                topo.is_grace_blackwell = True
                topo.has_pcie           = False
                break

    # ── InfiniBand ───────────────────────────────────────────────────────────
    ib = Path("/sys/class/infiniband")
    topo.has_infiniband = ib.exists() and any(True for _ in ib.iterdir())

    # ── SHARP (IB-level) ─────────────────────────────────────────────────────
    if topo.has_infiniband:
        try:
            topo.has_sharp = "ib_sharp" in Path("/proc/modules").read_text()
        except Exception:
            pass

    # ── NVLS (NVLink-level switch reduction, NVLink gen ≥ 4) ─────────────────
    topo.has_nvlink_sharp = topo.nvlink_gen >= 4

    # ── Force overrides (for testing / misconfigured nodes) ──────────────────
    if force == "nvl72":
        topo.nvlink_lanes_per_gpu   = 18
        topo.gpus_per_nvlink_domain = 72
        topo.has_infiniband         = False
        topo.has_pcie               = False
    elif force == "pcie":
        topo.nvlink_lanes_per_gpu   = 9
        topo.gpus_per_nvlink_domain = 8
        topo.has_pcie               = True

    # [R-5] Log topology — essential for debugging on large jobs
    log.info(
        "Detected topology: %s (NVLink lanes=%d, IB=%s, SHARP=%s, NVLS=%s, CUDA=%d.%d)",
        topo.label, topo.nvlink_lanes_per_gpu,
        topo.has_infiniband, topo.has_sharp, topo.has_nvlink_sharp,
        topo.cuda_major, topo.cuda_minor,
    )
    return topo


# ══════════════════════════════════════════════════════════════════════════════
# NCCL configuration
# ══════════════════════════════════════════════════════════════════════════════

def configure_nccl(
    topo:              ClusterTopology,
    socket_ifname_exclude: str = "lo,docker,veth,eth0",   # [A-3]
) -> None:
    """
    Set NCCL environment variables for the detected topology.
    MUST be called before torch.distributed.init_process_group().

    Parameters
    ----------
    socket_ifname_exclude : comma-separated interface name prefixes to exclude.
        The leading '^' (NCCL exclude syntax) is prepended automatically.
        Default excludes loopback, Docker bridges, veth pairs, and eth0.
        Override if your cluster uses eth0 as the primary IB interface
        (e.g., set "" to disable fencing entirely).

    Design decisions
    ----------------
    NVL72 / Grace-Blackwell:
      - Disable IB (NVLink is the intra-rack fabric; IB only for inter-rack).
      - NCCL_SHARP_ENABLE=0 explicitly: SHARP requires IB; without this
        NCCL probes for SHARP on NVL72 nodes and emits confusing warnings. [R-4]
      - NCCL_ALGO=Auto: NCCL 2.21+ understands NVL72 topology.
      - NCCL_NVLS_ENABLE=1: in-switch NVLink reduction.
      - NCCL_PROTO=LL128: lower latency on unified NVLink domain.

    B200 PCIe / H100 (standard multi-node with IB):
      - NCCL_ALGO=Auto: NVLink 5.0 shifts the Ring/Tree crossover vs 4.0;
        hardcoding Tree is wrong for B200.
      - NCCL_SHARP_ENABLE=1 if SHARP is detected.
      - Wider buffers and more channels for NVLink 5.0.
    """
    env: dict[str, str] = {
        # Common to all topologies
        "NCCL_P2P_LEVEL":         "NVL",
        "NCCL_SHM_DISABLE":       "0",
        "NCCL_ALGO":              "Auto",
        "NCCL_BUFFSIZE":          str(8 * 1024 * 1024),   # 8 MB
        "NCCL_NCHANNELS_PER_NET": "8" if topo.nvlink_gen >= 5 else "4",
        "NCCL_NVLS_ENABLE":       "1" if topo.has_nvlink_sharp else "0",
    }

    # Network fencing [A-3]: exclude management / docker / loopback interfaces
    if socket_ifname_exclude:
        env["NCCL_SOCKET_IFNAME"] = f"^{socket_ifname_exclude}"

    if topo.is_nvl72 or topo.is_grace_blackwell:
        env.update({
            "NCCL_IB_DISABLE":    "1",
            "NCCL_NET_GDR_LEVEL": "0",
            "NCCL_SHARP_ENABLE":  "0",     # [R-4] explicit: SHARP needs IB
            "NCCL_PROTO":         "LL128",
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
# Interconnect health check  [A-5]
# ══════════════════════════════════════════════════════════════════════════════

# Minimum expected bus bandwidth thresholds (GB/s).
# These are intentionally conservative to avoid false positives on cold starts.
# NVLink 5.0 (B200): ~400 GB/s per GPU at full load; 100 GB/s is degraded.
# NVLink 4.0 (H100): ~200 GB/s per GPU at full load;  50 GB/s is degraded.
_BW_THRESHOLD_GB_S = {5: 100.0, 4: 50.0}
_WARMUP_ITERS      = 20   # cold NCCL is always slow; 20 iters reach steady state
_MEASURE_ITERS     = 5    # average over 5 post-warmup iterations


def verify_interconnect(
    topo:       ClusterTopology,
    rank:       int,
    world_size: int,
) -> None:
    """
    Issue a canary all-reduce to measure bus bandwidth and detect degradation.

    Emits a WARNING (not an error) if bandwidth is below the expected minimum.
    A training job should not be aborted at step 0 due to a cold-start
    measurement artefact; the warning surfaces in logs for human review.

    Bandwidth formula
    -----------------
    Bus BW (GB/s) = payload_bytes * 2 * (N-1)/N / elapsed_s / 1e9

    The factor 2*(N-1)/N is the standard ring all-reduce correction:
    each byte traverses 2*(N-1)/N hops across the ring.  This is used
    for comparability with nccl-tests even though NCCL may use Tree or
    NVLS internally — it gives a lower-bound estimate in those cases.
    """
    if world_size < 2:
        return

    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # 256 MB payload in float32
    payload_bytes = 256 * 1024 * 1024
    n_elems = payload_bytes // 4
    tensor  = torch.randn(n_elems, device=device)

    # Warmup: enough iterations for NCCL to finish topology negotiation
    # and for CUDA to reach thermal/clock steady state [R-2]
    for _ in range(_WARMUP_ITERS):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    # Measure
    total_s = 0.0
    for _ in range(_MEASURE_ITERS):
        t0 = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        total_s += time.perf_counter() - t0
    avg_s = total_s / _MEASURE_ITERS

    # Bus bandwidth in GB/s [R-2]: fixed units (was Gbps in intern version)
    ring_factor = 2.0 * (world_size - 1) / world_size
    bus_bw_gbs  = (payload_bytes * ring_factor) / avg_s / 1e9

    if rank == 0:
        threshold = _BW_THRESHOLD_GB_S.get(topo.nvlink_gen, 50.0)
        status    = "OK" if bus_bw_gbs >= threshold else "DEGRADED"
        log_fn    = log.info if status == "OK" else log.warning
        log_fn(
            "Interconnect health [%s]: %.1f GB/s bus BW (threshold %.0f GB/s) — %s",
            topo.label, bus_bw_gbs, threshold, status,
        )


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


def slurm_init(
    force_topology:        Optional[str] = None,
    socket_ifname_exclude: str           = "lo,docker,veth,eth0",   # [A-3]
) -> DistribEnv:
    """
    Initialise torch.distributed from SLURM env vars, with topology-aware
    NCCL configuration.  Returns a DistribEnv describing this process.

    Must be called before any CUDA or distributed operation.

    MASTER_ADDR resolution [R-1]
    ----------------------------
    Every rank resolves the master address independently via scontrol.
    The intern's approach — only local_rank==0 runs scontrol and others
    "rely on srun to propagate it" — is incorrect.  MASTER_ADDR is a
    PyTorch env var, not a SLURM var; srun does not propagate it.
    scontrol is idempotent, fast (< 50 ms), and runs once per job.
    """
    try:
        rank             = int(os.environ["SLURM_PROCID"])
        world_size       = int(os.environ["SLURM_NTASKS"])
        local_rank       = int(os.environ["SLURM_LOCALID"])
        local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    except KeyError as e:
        raise RuntimeError(
            f"SLURM environment variable {e} not set. "
            "Are you running inside a SLURM allocation via srun/sbatch?"
        ) from e

    # Every rank resolves the master address independently [R-1]
    if "MASTER_ADDR" not in os.environ:
        try:
            node_list = os.environ["SLURM_NODELIST"]
            master    = subprocess.check_output(
                ["scontrol", "show", "hostnames", node_list],
                text=True, timeout=10,
            ).split()[0]
            os.environ["MASTER_ADDR"] = master
        except Exception as exc:
            raise RuntimeError(
                "Could not resolve SLURM master node address from "
                f"SLURM_NODELIST={os.environ.get('SLURM_NODELIST', '<unset>')}"
            ) from exc

    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"]       = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    topo = detect_topology(force=force_topology)
    configure_nccl(topo, socket_ifname_exclude=socket_ifname_exclude)

    if not dist.is_initialized():   # [A-4]
        dist.init_process_group(
            backend     = "nccl",
            init_method = "env://",
        )
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    verify_interconnect(topo, rank, world_size)   # [A-5]
    dist.barrier()

    log.info(
        "Process group initialised: rank %d / %d, local %d / %d",
        rank, world_size, local_rank, local_world_size,
    )
    return DistribEnv(
        rank=rank, world_size=world_size,
        local_rank=local_rank, local_world_size=local_world_size,
        topology=topo,
    )
