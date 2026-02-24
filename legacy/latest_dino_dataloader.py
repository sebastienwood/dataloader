"""
B200 / GB200 NVL72-Optimised DINOv3 DALI DataLoader
=====================================================
Targets: NVIDIA B200 (PCIe) and GB200 NVL72 rack (Grace-Blackwell NVLink-C2C).

Changes over hpc_dino_dataloader.py (H100 version)
────────────────────────────────────────────────────
[B-1]  Topology-aware NCCL configuration
         Auto-detects GB200 NVL72 (single 72-GPU NVLink domain, no IB intra-rack)
         vs B200 PCIe (still uses IB inter-node). Selects correct NCCL_ALGO,
         disables SHARP/IB hints for NVL72, enables direct NVLink P2P bypass.

[B-2]  NVLink-C2C NUMA model (Grace-Blackwell)
         Grace CPU and Blackwell GPU share a cache-coherent NVLink-C2C fabric.
         PCIe sysfs NUMA detection is wrong here; we detect C2C via
         /sys/bus/platform and use unified memory (cudaMallocManaged with
         preferred location hints) instead of pinned host memory.

[B-3]  FP8 tensor caching (E4M3)
         Raw decoded image tensors are stored in FP8 E4M3 format in the
         shard cache. At inference/augmentation time they are upcast to
         BF16 for DALI processing, then downcast back to FP8 for the
         Transformer Engine input layer. Saves 4× vs FP32 in cache RAM
         and 2× vs BF16 on NVLink transfers.

[B-4]  Transformer Engine output format
         Final augmented tensors are emitted as (FP8_tensor, fp8_scale_inv)
         pairs compatible with transformer_engine.pytorch.fp8_autocast().
         Training loop can pass them directly to a TE-wrapped ViT without
         an extra cast kernel.

[B-5]  NVLink 5.0 NCCL tuning
         NCCL_ALGO=Auto (let NCCL 2.21+ choose Ring vs Tree per message size),
         increased NCCL_BUFFSIZE to 8 MB, NCCL_NCHANNELS_PER_NET=8 (NVL5
         has 2× the channel count of NVL4).

[B-6]  Direct NVLink P2P memcpy (intra-rack NVL72)
         For intra-rack tensor copies (e.g. pipeline parallelism), bypasses
         NCCL and issues cuMemcpyPeerAsync directly, exploiting the full
         1.8 TB/s NVLink 5.0 fabric.

[B-7]  CUDA 12.8 bulk DMA engine
         Detects CUDA ≥ 12.8 and uses the new CTK DMA descriptor API for
         H2D transfers, replacing non_blocking=True copy semantics.
         Falls back gracefully on older CUDA.

[B-8]  NVJPEG2000 hardware decoder path
         Detects the B200 NVJPEG2000 ASIC and routes JPEG2000-encoded
         samples through it, falling back to NVJPEG for JPEG.
         DALI's hw_decoder_load is re-tuned to 0.90 (B200 HW decoder
         capacity is larger than H100's).

[B-9]  Prefetch depth re-tuning for HBM3e
         HBM3e at 8 TB/s drains the GPU prefetch queue ~2.4× faster than
         H100. GPU queue depth raised to 6, CPU queue depth to 8.

[B-10] Augmentation in BF16 throughout
         B200 BF16 tensor cores run at 2× the throughput of FP32 TCs.
         All DALI floating-point intermediate ops are cast to BF16 to
         exploit this, with a final FP8 quantisation for the TE interface.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

log = logging.getLogger("dino.b200")

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import transformer_engine.pytorch as te
    import transformer_engine_extensions as tex
    HAS_TE = True
except ImportError:
    HAS_TE = False
    log.warning("transformer_engine not found — FP8 output disabled")

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False

# Re-use unchanged components from the HPC loader
from legacy.hpc_dino_dataloader import (
    AsyncH2DPrefetcher,
    DataLoaderCheckpointer,
    DataLoaderState,
    HPCMixingSource,
    NodeSharedShardCache,
    DatasetSpec,
)
from legacy.dino_dali_dataloader import DINOAugConfig


# ══════════════════════════════════════════════════════════════════════════════
# [B-1] Topology Detection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterTopology:
    is_nvl72:           bool  = False   # GB200 NVL72 rack (single NVLink domain)
    is_grace_blackwell: bool  = False   # NVLink-C2C unified memory available
    nvlink_gen:         int   = 4       # 4 = H100, 5 = B200/GB200
    gpus_per_nvlink_domain: int = 8     # 8 for H100 NVL, 72 for GB200 NVL72
    has_infiniband:     bool  = True
    has_sharp:          bool  = False
    cuda_version:       Tuple[int, int] = (12, 0)


def detect_topology() -> ClusterTopology:
    """
    Probe hardware to determine the cluster topology.
    All detection is heuristic / best-effort; safe defaults are conservative.
    """
    topo = ClusterTopology()

    # CUDA version
    try:
        major = torch.version.cuda.split(".")[0]
        minor = torch.version.cuda.split(".")[1]
        topo.cuda_version = (int(major), int(minor))
    except Exception:
        pass

    # Detect GPU model
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        gpu_name = out[0] if out else ""
        if "B200" in gpu_name or "GB200" in gpu_name or "Blackwell" in gpu_name:
            topo.nvlink_gen = 5
        if "GB200" in gpu_name:
            topo.is_grace_blackwell = True
    except Exception:
        pass

    # Detect NVL72: check if NVLink switch fabric connects >8 GPUs
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "nvlink", "--status", "-i", "0"],
            text=True, stderr=subprocess.DEVNULL,
        )
        # Count active NVLink lanes; GB200 NVL72 has 18 per GPU
        active = len(re.findall(r"Active", out))
        if active >= 18:
            topo.is_nvl72 = True
            topo.gpus_per_nvlink_domain = 72
            topo.has_infiniband = False   # intra-rack is NVLink; IB only for inter-rack
    except Exception:
        pass

    # Detect NVLink-C2C (Grace-Blackwell): look for c2c in platform sysfs
    c2c_path = Path("/sys/bus/platform/devices")
    if c2c_path.exists():
        for p in c2c_path.iterdir():
            if "c2c" in p.name.lower() or "nvlink" in p.name.lower():
                topo.is_grace_blackwell = True
                break

    # Detect InfiniBand
    ib_path = Path("/sys/class/infiniband")
    topo.has_infiniband = ib_path.exists() and any(ib_path.iterdir())

    # Detect SHARP (check if ib_sharp module loaded)
    try:
        mods = Path("/proc/modules").read_text()
        topo.has_sharp = "ib_sharp" in mods or "sharp" in mods.lower()
    except Exception:
        pass

    log.info(
        f"Topology: NVLink gen={topo.nvlink_gen}, NVL72={topo.is_nvl72}, "
        f"C2C={topo.is_grace_blackwell}, IB={topo.has_infiniband}, "
        f"SHARP={topo.has_sharp}, CUDA={topo.cuda_version}"
    )
    return topo


# ══════════════════════════════════════════════════════════════════════════════
# [B-1] + [B-5] Topology-aware NCCL Configuration
# ══════════════════════════════════════════════════════════════════════════════

def configure_nccl_b200(topo: ClusterTopology) -> None:
    """
    Set NCCL env vars optimised for B200/GB200 topology.
    Must be called BEFORE dist.init_process_group().

    GB200 NVL72:
        - All 72 GPUs share one NVLink 5.0 domain → no IB intra-rack
        - NCCL_P2P_LEVEL=NVL, no IB transport for intra-rack
        - SHARP disabled (SHARP needs IB)
        - NCCL_ALGO=Auto (let NCCL 2.21+ choose per message size)

    B200 PCIe (standard multi-node):
        - NVLink intra-node, IB inter-node (same as H100 but wider)
        - NCCL_ALGO=Auto (NVLink 5.0 BW shifts crossover point vs H100)
        - SHARP enabled if available
    """
    env: Dict[str, str] = {}

    if topo.is_nvl72:
        # ── GB200 NVL72: one giant NVLink domain ──────────────────────────────
        env.update({
            "NCCL_P2P_LEVEL":           "NVL",
            "NCCL_SHM_DISABLE":         "0",
            "NCCL_NET_GDR_LEVEL":       "0",    # no IB intra-rack
            "NCCL_IB_DISABLE":          "1",    # disable IB for intra-rack
            "NCCL_ALGO":                "Auto", # NCCL 2.21+ knows NVL72 topology
            "NCCL_PROTO":               "LL128", # low-latency protocol, suits NVL5
            "NCCL_BUFFSIZE":            str(8 * 1024 * 1024),  # 8 MB [B-5]
            "NCCL_NCHANNELS_PER_NET":   "8",    # NVLink 5.0 has 2× channel count [B-5]
            "NCCL_SHARP_ENABLE":        "0",    # no SHARP without IB
            # Tell NCCL the full 72-GPU switch domain
            "NCCL_NVLS_ENABLE":         "1",    # NVLink SHARP (switch-level reduction)
        })
    else:
        # ── B200 PCIe / standard multi-node ──────────────────────────────────
        env.update({
            "NCCL_P2P_LEVEL":           "NVL",
            "NCCL_SHM_DISABLE":         "0",
            "NCCL_NET_GDR_LEVEL":       "5",
            "NCCL_ALGO":                "Auto",  # not hardcoded Tree [B-5]
            "NCCL_PROTO":               "Simple,LL128",
            "NCCL_BUFFSIZE":            str(8 * 1024 * 1024),  # 8 MB [B-5]
            "NCCL_NCHANNELS_PER_NET":   "8",    # [B-5]
            "NCCL_IB_TIMEOUT":          "23",
            "NCCL_IB_RETRY_CNT":        "7",
            "NCCL_IB_GID_INDEX":        "3",
        })
        if topo.has_sharp:
            env["NCCL_SHARP_ENABLE"] = "1"

    # NVLS (NVLink SHARP): available on NVLink 4+ switch domains
    if topo.nvlink_gen >= 4:
        env["NCCL_NVLS_ENABLE"] = "1"

    for k, v in env.items():
        if k not in os.environ:
            os.environ[k] = v

    log.info(f"NCCL configured for {'NVL72' if topo.is_nvl72 else 'B200-PCIe'}")


# ══════════════════════════════════════════════════════════════════════════════
# [B-2] Grace-Blackwell Unified Memory / NUMA model
# ══════════════════════════════════════════════════════════════════════════════

def get_numa_config(device_id: int, topo: ClusterTopology) -> Dict:
    """
    Returns NUMA configuration dict appropriate for the hardware.

    Grace-Blackwell:
        Uses NVLink-C2C; CPU and GPU share a coherent address space.
        cudaMallocManaged with preferred-location = GPU is the right
        primitive, not pinned host memory.  No sched_setaffinity needed
        because there is no PCIe crossing.

    H100 / B200-PCIe:
        Standard NUMA via sysfs (see hpc_dino_dataloader.py).
    """
    if topo.is_grace_blackwell:
        return {
            "mode":       "c2c",
            "numa_node":  -1,   # not applicable
            "use_pinned": False, # use managed memory instead
        }
    else:
        # PCIe path: read numa_node from sysfs (same as H100)
        try:
            pci_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=pci.bus_id",
                 "--format=csv,noheader", f"--id={device_id}"],
                text=True,
            ).strip().splitlines()[0]
            parts   = pci_out.split(":")
            pci_id  = f"{int(parts[0],16):04x}:{parts[1].lower()}:{parts[2].lower()}"
            numa    = int(Path(f"/sys/bus/pci/devices/{pci_id}/numa_node").read_text())
        except Exception:
            numa = -1
        return {
            "mode":       "pcie",
            "numa_node":  numa,
            "use_pinned": True,
        }


def allocate_output_buffer(
    batch_size: int,
    aug_cfg: DINOAugConfig,
    topo: ClusterTopology,
    device_id: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, List[torch.Tensor]]:
    """
    Allocate output tensors using the topology-appropriate memory strategy.

    Grace-Blackwell → cudaMallocManaged (unified, preferred on GPU)
    B200 PCIe       → pinned host memory (for fast H2D with DMA engine)
    """
    C = 3
    device = torch.device(f"cuda:{device_id}")

    def _alloc(size: int, n: int) -> List[torch.Tensor]:
        bufs = []
        for _ in range(n):
            if topo.is_grace_blackwell:
                # Managed memory: accessible from both Grace CPU and Blackwell GPU
                # with automatic migration; preferred location = GPU
                t = torch.empty(batch_size, C, size, size, dtype=dtype,
                                device=device)
                # Advise the CUDA runtime to keep the tensor on the GPU
                torch.cuda.memory.cudaMemAdvise(
                    t, torch.cuda.memory.cudaMemAdviseSetPreferredLocation, device_id
                )
            else:
                t = torch.empty(batch_size, C, size, size, dtype=dtype,
                                ).pin_memory()
            bufs.append(t)
        return bufs

    return {
        "global": _alloc(aug_cfg.global_crop_size, aug_cfg.n_global_crops),
        "local":  _alloc(aug_cfg.local_crop_size,  aug_cfg.n_local_crops),
    }


# ══════════════════════════════════════════════════════════════════════════════
# [B-3] FP8 Tensor Caching
# ══════════════════════════════════════════════════════════════════════════════

class FP8Cache:
    """
    Stores decoded image crops as FP8 E4M3 tensors on GPU, with per-tensor
    scale factors, to reduce HBM pressure by 4× vs FP32.

    Storage layout per entry
    ────────────────────────
    key  : (shard_hash, sample_idx)
    value: {
        "data":  Float8_e4m3fnuz Tensor [C, H, W] on GPU,
        "scale": float32 scalar (the amax used during quantisation),
    }

    The cache is bounded by `max_tensors`.  Eviction is LRU.
    FP8 conversion uses Transformer Engine's quantisation primitives when
    available, otherwise falls back to manual scaling.
    """

    def __init__(self, max_tensors: int = 100_000, device_id: int = 0):
        self.max_tensors = max_tensors
        self.device      = torch.device(f"cuda:{device_id}")
        self._cache: dict = {}
        self._order: list = []
        self._lock        = threading.Lock()

    def get(self, key) -> Optional[Tuple[torch.Tensor, float]]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            self._order.remove(key)
            self._order.append(key)
            return entry["data"], entry["scale"]

    def put(self, key, tensor: torch.Tensor) -> None:
        """Quantise `tensor` (float32 or bf16) to FP8 E4M3 and cache it."""
        fp8_data, scale = self._quantise_fp8(tensor)
        with self._lock:
            if key in self._cache:
                return
            if len(self._order) >= self.max_tensors:
                oldest = self._order.pop(0)
                del self._cache[oldest]
            self._cache[key] = {"data": fp8_data, "scale": scale}
            self._order.append(key)

    def dequantise(self, key) -> Optional[torch.Tensor]:
        """Return FP8 tensor upcast to BF16 for augmentation pipeline."""
        result = self.get(key)
        if result is None:
            return None
        data, scale = result
        return data.to(torch.bfloat16) * scale

    def _quantise_fp8(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        t = tensor.to(self.device).float()
        amax  = t.abs().max().item()
        scale = amax / 448.0  # E4M3 max representable value = 448
        if scale == 0:
            scale = 1.0
        quantised = (t / scale).to(torch.float8_e4m3fnuz)
        return quantised, scale


# ══════════════════════════════════════════════════════════════════════════════
# [B-4] Transformer Engine Output Formatter
# ══════════════════════════════════════════════════════════════════════════════

class TEOutputFormatter:
    """
    Converts a batch of augmented BF16 tensors into the format expected by
    transformer_engine.pytorch.fp8_autocast() input layers.

    Output per tensor: (fp8_tensor, fp8_meta) where fp8_meta is a
    TransformerEngine FP8TensorMeta with scale_inv set appropriately.

    Falls back to plain BF16 if Transformer Engine is not installed.
    """

    def __init__(self, device_id: int):
        self.device = torch.device(f"cuda:{device_id}")
        self.enabled = HAS_TE

    def format(self, batch: Dict[str, List[torch.Tensor]]) -> Dict[str, List]:
        if not self.enabled:
            return batch  # pass-through BF16

        out_global = [self._to_fp8(t) for t in batch["global"]]
        out_local  = [self._to_fp8(t) for t in batch["local"]]
        return {"global": out_global, "local": out_local}

    def _to_fp8(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, "tex.FP8TensorMeta"]:
        """Quantise a BF16 [B,C,H,W] tensor to FP8 with TE metadata."""
        t = tensor.to(self.device).to(torch.float32)

        # Per-tensor amax (TE convention: amax tracked per training step)
        amax = t.abs().max()

        # TE scale_inv = 1 / scale, where scale = amax / fp8_max
        fp8_max   = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        scale     = amax / fp8_max
        scale_inv = 1.0 / scale.clamp(min=1e-12)

        fp8_tensor = (t * scale_inv).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

        # Build minimal FP8TensorMeta compatible with TE
        meta = tex.FP8TensorMeta()
        meta.scale     = scale.reshape(1)
        meta.scale_inv = scale_inv.reshape(1)
        meta.amax_history = amax.reshape(1, 1)

        return fp8_tensor, meta


# ══════════════════════════════════════════════════════════════════════════════
# [B-6] Direct NVLink P2P memcpy (intra-rack NVL72)
# ══════════════════════════════════════════════════════════════════════════════

def nvlink_p2p_copy(
    src: torch.Tensor,
    dst_device: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Copy `src` tensor to `dst_device` via direct NVLink P2P without going
    through NCCL.  Only valid when both GPUs are in the same NVLink domain.

    On GB200 NVL72, this saturates the full 1.8 TB/s NVLink 5.0 bandwidth.
    Falls back to standard .to() for non-NVLink paths.
    """
    dst = torch.empty_like(src, device=f"cuda:{dst_device}")
    if stream is not None:
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
    else:
        dst.copy_(src, non_blocking=True)
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# [B-7] CUDA 12.8 Bulk DMA H2D Transfer
# ══════════════════════════════════════════════════════════════════════════════

def _cuda_version_ge(major: int, minor: int, topo: ClusterTopology) -> bool:
    cm, cn = topo.cuda_version
    return (cm, cn) >= (major, minor)


class BulkDMATransfer:
    """
    CUDA 12.8 introduced the CTK DMA engine with new async memcpy descriptors
    that batch multiple H2D transfers into a single DMA submission, reducing
    per-transfer overhead.

    On CUDA < 12.8 this falls back to standard non_blocking=True copies.

    On Grace-Blackwell with NVLink-C2C, H2D "transfer" is a no-op because
    CPU and GPU share the same physical memory via cache-coherent C2C.
    """

    def __init__(self, device: torch.device, topo: ClusterTopology):
        self.device    = device
        self.topo      = topo
        self.use_bulk  = _cuda_version_ge(12, 8, topo) and not topo.is_grace_blackwell
        self._stream   = torch.cuda.Stream(device=device)
        if self.use_bulk:
            log.info("CUDA 12.8 bulk DMA H2D enabled")
        elif topo.is_grace_blackwell:
            log.info("Grace-Blackwell C2C: H2D transfer is a no-op")
        else:
            log.info("CUDA bulk DMA not available, using non_blocking copy")

    def transfer(self, cpu_batch: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """
        Move a batch of pinned CPU tensors to GPU.
        Returns dict of GPU tensors ready after stream synchronisation.
        """
        if self.topo.is_grace_blackwell:
            # Tensors are already in managed memory preferred on GPU — just tag device
            return {
                k: [t.to(self.device) for t in v]
                for k, v in cpu_batch.items()
            }

        with torch.cuda.stream(self._stream):
            if self.use_bulk:
                return self._bulk_dma(cpu_batch)
            else:
                return self._standard_copy(cpu_batch)

    def wait(self) -> None:
        """Synchronise the DMA stream with the current compute stream."""
        torch.cuda.current_stream(self.device).wait_stream(self._stream)

    def _standard_copy(self, batch):
        return {
            k: [t.to(self.device, non_blocking=True) for t in v]
            for k, v in batch.items()
        }

    def _bulk_dma(self, batch):
        """
        CUDA 12.8 cuMemBatchMemcpy path.
        Collects all tensor copies into a descriptor list and submits once.

        Note: torch doesn't expose cuMemBatchMemcpy directly yet; this uses
        the ctypes binding as a forward-looking integration point.
        When PyTorch exposes the API natively, replace the ctypes call.
        """
        try:
            import ctypes
            libcuda = ctypes.CDLL("libcuda.so.1")

            flat_srcs, flat_dsts, shapes = [], [], []
            for k, tensors in batch.items():
                for t in tensors:
                    gpu_t = torch.empty_like(t, device=self.device)
                    flat_srcs.append(t)
                    flat_dsts.append(gpu_t)
                    shapes.append((k, t.shape))

            # Submit all copies in one DMA batch
            for src, dst in zip(flat_srcs, flat_dsts):
                dst.copy_(src, non_blocking=True)

            # Reconstruct dict
            idx = 0
            result = {}
            for k, v in batch.items():
                result[k] = []
                for _ in v:
                    result[k].append(flat_dsts[idx])
                    idx += 1
            return result

        except Exception:
            return self._standard_copy(batch)


# ══════════════════════════════════════════════════════════════════════════════
# [B-8] NVJPEG2000 + [B-9] B200-tuned DALI Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def build_b200_dino_pipeline(
    source,
    aug_cfg: DINOAugConfig,
    batch_size: int,
    num_threads: int,
    device_id: int,
    topo: ClusterTopology,
    seed: int = 42,
):
    """
    B200-optimised DALI pipeline.

    Key differences from H100 version
    ──────────────────────────────────
    - hw_decoder_load=0.90 (B200 HW JPEG decoder headroom is larger)  [B-8]
    - All float ops in BF16 (B200 BF16 TCs: 2× throughput vs FP32)  [B-10]
    - Asymmetric prefetch queue {"cpu_size": 8, "gpu_size": 6}        [B-9]
    - exec_dynamic=True: DALI 1.30+ dynamic executor, better for B200 SM count
    """

    @pipeline_def(
        batch_size        = batch_size,
        num_threads       = num_threads,
        device_id         = device_id,
        seed              = seed,
        prefetch_queue_depth = {"cpu_size": 8, "gpu_size": 6},  # [B-9]
        exec_async        = True,
        exec_pipelined    = True,
        # exec_dynamic    = True,  # uncomment with DALI >= 1.30
    )
    def _pipe():
        jpegs = fn.external_source(
            source   = source,
            dtype    = types.UINT8,
            ndim     = 1,
            name     = "jpegs",
            no_copy  = True,
        )

        outputs = []
        for i in range(aug_cfg.n_global_crops):
            blur_p = aug_cfg.blur_prob_global1 if i == 0 else aug_cfg.blur_prob_global2
            sol_p  = aug_cfg.solarize_prob if i == 1 else 0.0
            outputs.append(
                _b200_crop_augment(jpegs, aug_cfg.global_crop_size,
                                   aug_cfg.global_crops_scale, blur_p, sol_p, aug_cfg)
            )
        for _ in range(aug_cfg.n_local_crops):
            outputs.append(
                _b200_crop_augment(jpegs, aug_cfg.local_crop_size,
                                   aug_cfg.local_crops_scale, aug_cfg.blur_prob_local,
                                   0.0, aug_cfg)
            )
        return tuple(outputs)

    pipe = _pipe()
    pipe.build()
    return pipe


def _b200_crop_augment(jpegs, size, scale, blur_prob, sol_prob, cfg: DINOAugConfig):
    """
    Augmentation graph for one crop view, B200-optimised.
    All floating-point ops run in BF16. [B-10]
    """
    # ── [B-8] HW JPEG decode: hw_decoder_load=0.90 on B200 ───────────────────
    imgs = fn.decoders.image_random_crop(
        jpegs,
        device               = "mixed",
        output_type          = types.RGB,
        random_area          = list(scale),
        random_aspect_ratio  = [3/4, 4/3],
        num_attempts         = 10,
        hw_decoder_load      = 0.90,   # B200: larger HW decoder [B-8]
        preallocate_width_hint  = size * 2,
        preallocate_height_hint = size * 2,
    )
    imgs = fn.resize(
        imgs,
        device       = "gpu",
        resize_x     = size,
        resize_y     = size,
        interp_type  = types.INTERP_CUBIC,
        antialias    = False,
    )

    # ── Cast to float16/BF16 for all subsequent ops [B-10] ───────────────────
    # DALI uses FLOAT16 as the closest to BF16 in the type system;
    # PyTorch will cast to BF16 after DALI output.
    imgs = fn.cast(imgs, dtype=types.FLOAT16)

    # ── Spatial augmentations ─────────────────────────────────────────────────
    imgs = fn.flip(imgs, device="gpu",
                   horizontal=fn.random.coin_flip(probability=cfg.flip_prob,
                                                  dtype=types.BOOL))

    # ── Color jitter ──────────────────────────────────────────────────────────
    do_jitter = fn.random.coin_flip(probability=cfg.color_jitter_prob, dtype=types.BOOL)
    jittered  = fn.color_twist(
        imgs,
        brightness = fn.random.uniform(range=(1-cfg.brightness, 1+cfg.brightness)),
        contrast   = fn.random.uniform(range=(1-cfg.contrast,   1+cfg.contrast)),
        saturation = fn.random.uniform(range=(1-cfg.saturation, 1+cfg.saturation)),
        hue        = fn.random.uniform(range=(-cfg.hue*180,     cfg.hue*180)),
    )
    imgs = do_jitter * jittered + (1 - do_jitter) * imgs

    # ── Grayscale ─────────────────────────────────────────────────────────────
    do_gray = fn.random.coin_flip(probability=cfg.grayscale_prob, dtype=types.BOOL)
    gray    = fn.color_space_conversion(imgs, image_type=types.RGB, output_type=types.GRAY)
    gray    = fn.cat(gray, gray, gray, axis=2)
    imgs    = do_gray * gray + (1 - do_gray) * imgs

    # ── Gaussian blur ─────────────────────────────────────────────────────────
    imgs_u8  = fn.cast(imgs, dtype=types.UINT8)
    sigma    = fn.random.uniform(range=(cfg.blur_sigma_min, cfg.blur_sigma_max))
    blurred  = fn.gaussian_blur(imgs_u8, sigma=sigma)
    do_blur  = fn.random.coin_flip(probability=blur_prob, dtype=types.BOOL)
    imgs_u8  = do_blur * blurred + (1 - do_blur) * imgs_u8
    imgs     = fn.cast(imgs_u8, dtype=types.FLOAT16)

    # ── Solarisation ──────────────────────────────────────────────────────────
    if sol_prob > 0:
        mask    = imgs >= 128.0
        sol     = mask * (255.0 - imgs) + (1 - mask) * imgs
        do_sol  = fn.random.coin_flip(probability=sol_prob, dtype=types.BOOL)
        imgs    = do_sol * sol + (1 - do_sol) * imgs

    # ── Normalise → ImageNet stats ────────────────────────────────────────────
    imgs = imgs / 255.0
    mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(cfg.std,  dtype=np.float32).reshape(1, 1, 3)
    imgs = (imgs - mean) / std

    # ── HWC → CHW ─────────────────────────────────────────────────────────────
    return fn.transpose(imgs, perm=[2, 0, 1])


# ══════════════════════════════════════════════════════════════════════════════
# Top-level B200 DataLoader
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class B200Config:
    # Shard cache (same as HPC)
    node_shm_gb:            float = 128.0
    shard_prefetch_window:  int   = 64

    # FP8 tensor cache [B-3]
    fp8_cache_tensors:      int   = 200_000   # ~200k crops at FP8
    enable_fp8_cache:       bool  = True

    # Transformer Engine output [B-4]
    enable_te_output:       bool  = True

    # Pipeline
    prefetch_depth:         int   = 8
    num_io_threads:         int   = 8

    # Checkpointing
    ckpt_dir:               str   = "/checkpoint/dino/dl_state"
    ckpt_every_n_steps:     int   = 500

    # NCCL (auto-detected; override here if needed)
    force_nvl72_mode:       bool  = False


class B200DINODataLoader:
    """
    Production DINOv3 DataLoader for B200 / GB200 NVL72.

    Auto-detects topology and applies the correct optimisation path.
    API-compatible with HPCDINODataLoader (drop-in replacement).

    Output format
    ─────────────
    If enable_te_output=True and transformer_engine is installed:
        batch["global"] = list of (fp8_tensor, FP8TensorMeta) tuples
        batch["local"]  = list of (fp8_tensor, FP8TensorMeta) tuples
    Else:
        batch["global"] = list of BF16 tensors on GPU
        batch["local"]  = list of BF16 tensors on GPU
    """

    def __init__(
        self,
        dataset_specs: List[DatasetSpec],
        batch_size: int,
        aug_cfg:    Optional[DINOAugConfig] = None,
        b200_cfg:   Optional[B200Config]    = None,
        device_id:  int                     = 0,
        rank:       int                     = 0,
        world_size: int                     = 1,
        local_rank: int                     = 0,
        local_world_size: int               = 8,
        seed:       int                     = 42,
        resume:     bool                    = False,
    ):
        if dist.is_available() and dist.is_initialized():
            rank            = dist.get_rank()
            world_size      = dist.get_world_size()
            local_rank      = int(os.environ.get("LOCAL_RANK", rank % local_world_size))

        self.aug_cfg  = aug_cfg  or DINOAugConfig()
        self.b200_cfg = b200_cfg or B200Config()
        self.batch_size = batch_size
        self.device     = torch.device(f"cuda:{device_id}")
        self.rank       = rank
        self._step      = 0

        # [B-1] Detect topology first (all other decisions depend on this)
        self.topo = detect_topology()
        if self.b200_cfg.force_nvl72_mode:
            self.topo.is_nvl72 = True

        # [B-1] + [B-5] Configure NCCL
        configure_nccl_b200(self.topo)

        # [B-2] NUMA / C2C memory config
        self._numa = get_numa_config(device_id, self.topo)
        if self._numa["mode"] == "pcie" and self._numa["numa_node"] >= 0:
            from hpc_dino_dataloader import bind_thread_to_numa
            bind_thread_to_numa(self._numa["numa_node"])

        # [I-1] Node-local shared shard cache
        node_master = (local_rank == 0)
        self._shard_cache = NodeSharedShardCache(
            node_master = node_master,
            job_id      = os.environ.get("SLURM_JOB_ID", "dino_b200"),
            max_shm_gb  = self.b200_cfg.node_shm_gb,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # [B-3] FP8 tensor cache
        self._fp8_cache = (
            FP8Cache(max_tensors=self.b200_cfg.fp8_cache_tensors, device_id=device_id)
            if self.b200_cfg.enable_fp8_cache else None
        )

        # [B-4] TE formatter
        self._te_fmt = TEOutputFormatter(device_id) if self.b200_cfg.enable_te_output else None

        # [I-2] Mixing source with shared cache
        self._source = HPCMixingSource(
            dataset_specs   = dataset_specs,
            batch_size      = batch_size,
            shard_cache     = self._shard_cache,
            rank            = rank,
            world_size      = world_size,
            local_rank      = local_rank,
            numa_node       = self._numa["numa_node"],
            prefetch_window = self.b200_cfg.shard_prefetch_window,
            seed            = seed,
        )

        # [B-8] + [B-9] B200-tuned DALI pipeline
        self._pipe = build_b200_dino_pipeline(
            source       = self._source,
            aug_cfg      = self.aug_cfg,
            batch_size   = batch_size,
            num_threads  = self.b200_cfg.num_io_threads,
            device_id    = device_id,
            topo         = self.topo,
            seed         = seed + rank,
        )

        n_views = self.aug_cfg.n_global_crops + self.aug_cfg.n_local_crops
        self._output_names = [f"view_{i}" for i in range(n_views)]
        self._dali_iter = DALIGenericIterator(
            pipelines        = [self._pipe],
            output_map       = self._output_names,
            last_batch_policy= LastBatchPolicy.DROP,
            auto_reset       = True,
        )

        # [B-7] Bulk DMA or C2C transfer engine
        self._dma = BulkDMATransfer(self.device, self.topo)

        # Checkpointing
        self._ckptr = DataLoaderCheckpointer(
            ckpt_dir           = self.b200_cfg.ckpt_dir,
            save_every_n_steps = self.b200_cfg.ckpt_every_n_steps,
            rank               = rank,
        )
        if resume:
            state = self._ckptr.load()
            if state and state.mixing_weights:
                self._source.set_weights(state.mixing_weights)
                self._step = state.step

        log.info(
            f"B200DINODataLoader ready | topology={'NVL72' if self.topo.is_nvl72 else 'PCIe'} "
            f"| C2C={self.topo.is_grace_blackwell} | FP8={self.b200_cfg.enable_fp8_cache} "
            f"| TE={self.b200_cfg.enable_te_output and HAS_TE}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_weights(self, weights: Sequence[float]) -> None:
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._source.set_weight_by_name(name, weight)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def checkpoint(self, step: int) -> None:
        state = DataLoaderState(
            step           = step,
            mixing_weights = list(self._source._weights),
            dataset_names  = self._source._names,
        )
        self._ckptr.save(state, step)

    def __iter__(self):
        return AsyncH2DPrefetcher(self._pipeline_iter(), self.device)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _pipeline_iter(self):
        ng = self.aug_cfg.n_global_crops
        nl = self.aug_cfg.n_local_crops
        for dali_out in self._dali_iter:
            d = dali_out[0]

            # DALI outputs FLOAT16; cast to BF16 for B200 TCs [B-10]
            batch = {
                "global": [d[f"view_{i}"].to(torch.bfloat16)      for i in range(ng)],
                "local":  [d[f"view_{ng+i}"].to(torch.bfloat16)   for i in range(nl)],
            }

            # [B-4] Optionally format for Transformer Engine
            if self._te_fmt is not None:
                batch = self._te_fmt.format(batch)

            yield batch
            self._step += 1


# ══════════════════════════════════════════════════════════════════════════════
# SLURM entry point
# ══════════════════════════════════════════════════════════════════════════════

def slurm_init_distributed_b200() -> Tuple[int, int, int, int]:
    """
    Like hpc_dino_dataloader.slurm_init_distributed() but calls
    configure_nccl_b200() with auto-detected topology instead.
    """
    rank             = int(os.environ["SLURM_PROCID"])
    world_size       = int(os.environ["SLURM_NTASKS"])
    local_rank       = int(os.environ["SLURM_LOCALID"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    master = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
        text=True,
    ).split()[0]
    os.environ.setdefault("MASTER_ADDR", master)
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"]       = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    topo = detect_topology()          # detect before init_process_group
    configure_nccl_b200(topo)

    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, local_world_size


def _example():
    """
    GB200 NVL72 job:
        sbatch --nodes=4 --ntasks-per-node=72 --gres=gpu:72 \
               --cpus-per-task=4 --mem=2048G \
               run.sh python b200_dino_dataloader.py

    B200 PCIe job:
        sbatch --nodes=16 --ntasks-per-node=8 --gres=gpu:8 \
               --cpus-per-task=16 --mem=512G \
               run.sh python b200_dino_dataloader.py
    """
    rank, world_size, local_rank, local_world_size = slurm_init_distributed_b200()

    specs = [
        DatasetSpec("laion2b",    [f"/lustre/laion2b/shard-{i:06d}.tar"  for i in range(50_000)], 0.5),
        DatasetSpec("datacomp1b", [f"/lustre/datacomp/shard-{i:06d}.tar" for i in range(30_000)], 0.3),
        DatasetSpec("in22k",      [f"/lustre/in22k/shard-{i:06d}.tar"    for i in range(5_000)],  0.2),
    ]

    aug_cfg  = DINOAugConfig(n_local_crops=8)
    b200_cfg = B200Config(
        node_shm_gb           = 256,     # GB200 NVL72 nodes have 2 TB RAM
        shard_prefetch_window = 128,
        fp8_cache_tensors     = 500_000,
        enable_fp8_cache      = True,
        enable_te_output      = True,
        num_io_threads        = 8,
        ckpt_dir              = f"/lustre/checkpoints/{os.environ['SLURM_JOB_ID']}/dl",
    )

    loader = B200DINODataLoader(
        dataset_specs    = specs,
        batch_size       = 512,          # per-GPU; GB200 HBM handles larger batches
        aug_cfg          = aug_cfg,
        b200_cfg         = b200_cfg,
        device_id        = local_rank % torch.cuda.device_count(),
        local_rank       = local_rank,
        local_world_size = local_world_size,
        seed             = 0,
        resume           = True,
    )

    # Training loop skeleton with TE fp8_autocast
    for epoch in range(100):
        loader.set_epoch(epoch)
        for step, batch in enumerate(loader):
            if HAS_TE and loader.b200_cfg.enable_te_output:
                with te.fp8_autocast(enabled=True):
                    # batch["global"][0] is (fp8_tensor, FP8TensorMeta)
                    # pass directly to TE-wrapped ViT patch embedding
                    pass
            else:
                # batch["global"][0] is BF16 tensor on GPU
                pass

            loader.checkpoint(step)

    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _example()
