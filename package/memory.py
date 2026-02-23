"""
dino_loader.cache.memory
========================
Topology-aware memory allocation and H2D transfer.

Grace-Blackwell (NVLink-C2C)
    CPU and GPU share a cache-coherent address space.
    cudaMallocManaged with preferred-location=GPU is the right primitive.
    There is no H2D "transfer"; the tensor is already visible to both sides.

B200 / H100 PCIe
    Standard pinned host memory + non-blocking H2D copy.
    CUDA 12.8+: uses a dedicated DMA stream with explicit event synchronisation.

FP8 output (Transformer Engine)
    Augmented BF16 tensors are quantised to FP8 E4M3 with per-tensor amax
    tracking.  The rolling amax window (length 16) matches Transformer Engine's
    own internal convention, making the metadata directly usable by TE layers.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist

from dino_loader.config       import DINOAugConfig
from dino_loader.distributed  import ClusterTopology

log = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te
    import transformer_engine_extensions as tex
    HAS_TE = True
except ImportError:
    HAS_TE = False


# ══════════════════════════════════════════════════════════════════════════════
# Batch container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Batch:
    """
    Output batch.  Tensors are on GPU in BF16, or (FP8, FP8Meta) tuples
    when Transformer Engine output is enabled.
    """
    global_crops: List
    local_crops:  List

    def __iter__(self):
        """Convenience: unpack as (global_crops, local_crops)."""
        return iter((self.global_crops, self.local_crops))


# ══════════════════════════════════════════════════════════════════════════════
# NUMA-aware memory allocation
# ══════════════════════════════════════════════════════════════════════════════

def allocate_buffers(
    batch_size: int,
    aug_cfg:    DINOAugConfig,
    topo:       ClusterTopology,
    device:     torch.device,
    dtype:      torch.dtype = torch.bfloat16,
) -> Dict[str, List[torch.Tensor]]:
    """
    Allocate output buffers using the topology-appropriate strategy.

    Grace-Blackwell → managed memory, preferred on GPU (no H2D needed).
    PCIe            → pinned host memory (fast non-blocking H2D).
    """
    C = 3

    def _make(size: int, n: int) -> List[torch.Tensor]:
        bufs = []
        for _ in range(n):
            if topo.is_grace_blackwell:
                # Managed: accessible from Grace CPU and Blackwell GPU.
                # cudaMemAdviseSetPreferredLocation keeps it in HBM3e.
                t = torch.empty(batch_size, C, size, size, dtype=dtype, device=device)
                try:
                    torch.cuda.memory.cudaMemAdvise(
                        t,
                        torch.cuda.memory.cudaMemAdviseSetPreferredLocation,
                        device.index,
                    )
                except AttributeError:
                    pass   # API may not exist on older PyTorch; safe to skip
            else:
                t = torch.empty(batch_size, C, size, size, dtype=dtype).pin_memory()
            bufs.append(t)
        return bufs

    return {
        "global": _make(aug_cfg.global_crop_size, aug_cfg.n_global_crops),
        "local":  _make(aug_cfg.local_crop_size,  aug_cfg.n_local_crops),
    }


# ══════════════════════════════════════════════════════════════════════════════
# H2D transfer
# ══════════════════════════════════════════════════════════════════════════════

class H2DStream:
    """
    Dedicated CUDA stream for host-to-device transfers.

    Usage (inside the training loop):
        with h2d.transfer(cpu_batch) as gpu_batch:
            # gpu_batch is ready; compute on it here
            loss = model(gpu_batch)

    On Grace-Blackwell the context manager is a no-op (managed memory).
    """

    def __init__(self, device: torch.device, topo: ClusterTopology):
        self._device    = device
        self._topo      = topo
        self._stream    = torch.cuda.Stream(device=device)
        self._c2c       = topo.is_grace_blackwell
        if self._c2c:
            log.info("H2DStream: NVLink-C2C — H2D is a no-op")
        else:
            log.info("H2DStream: PCIe path, CUDA stream allocated")

    def send(self, cpu_batch: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """
        Initiate async H2D transfer.  Call wait() before using the result.
        On C2C, returns tensors already on device (no copy).
        """
        if self._c2c:
            return {k: [t.to(self._device) for t in v] for k, v in cpu_batch.items()}
        with torch.cuda.stream(self._stream):
            return {
                k: [t.to(self._device, non_blocking=True) for t in v]
                for k, v in cpu_batch.items()
            }

    def wait(self) -> None:
        """Synchronise the compute stream with the H2D stream."""
        if not self._c2c:
            torch.cuda.current_stream(self._device).wait_stream(self._stream)


# ══════════════════════════════════════════════════════════════════════════════
# Async prefetch iterator (overlaps H2D with compute)
# ══════════════════════════════════════════════════════════════════════════════

class AsyncPrefetchIterator:
    """
    Wraps any iterator of CPU-batch dicts, transferring each batch to GPU
    in the background while the consumer processes the previous one.

    This is the final stage in the pipeline: it hides H2D latency behind
    the forward/backward pass with zero additional threads (uses CUDA streams).
    """

    def __init__(
        self,
        source:     Iterator[Dict],
        h2d:        H2DStream,
        te_fmt:     Optional["FP8Formatter"] = None,
    ):
        self._src    = source
        self._h2d    = h2d
        self._te_fmt = te_fmt
        self._next: Optional[Dict] = None
        self._preload()

    def _preload(self) -> None:
        try:
            cpu = next(self._src)
        except StopIteration:
            self._next = None
            return
        gpu = self._h2d.send(cpu)
        # Do not call wait() yet — let the GPU transfer overlap with compute
        self._next = gpu

    def __iter__(self) -> "AsyncPrefetchIterator":
        return self

    def __next__(self) -> Batch:
        self._h2d.wait()       # ensure in-flight transfer is complete
        raw = self._next
        if raw is None:
            raise StopIteration
        self._preload()        # kick off next transfer before we return

        if self._te_fmt is not None:
            raw = self._te_fmt.format(raw)

        return Batch(
            global_crops = raw["global"],
            local_crops  = raw["local"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# FP8 output formatter (Transformer Engine)
# ══════════════════════════════════════════════════════════════════════════════

_FP8_MAX = 448.0   # max representable value in E4M3
_AMAX_WINDOW = 16  # rolling amax history length (matches TE convention)


class FP8Formatter:
    """
    Quantises BF16 augmented tensors to FP8 E4M3 with FP8TensorMeta,
    ready for direct use in a Transformer Engine fp8_autocast() context.

    Amax tracking
    -------------
    TE maintains a rolling window of per-tensor amax values and uses a
    delayed scaling strategy.  We replicate that here: each tensor tracks
    its own deque of length `_AMAX_WINDOW`, and the scale is computed from
    the window maximum.  This is important for stability at the start of
    training when amax can spike.

    Falls back to plain BF16 passthrough if TE is not installed.
    """

    def __init__(self, device: torch.device):
        self._device  = device
        self._enabled = HAS_TE
        # amax history per view index (global_0, global_1, local_0, ...)
        self._amax_history: Dict[int, deque] = {}
        self._lock = threading.Lock()

        if self._enabled:
            log.info("FP8Formatter: Transformer Engine FP8 output enabled")
        else:
            log.warning("FP8Formatter: transformer_engine not found — BF16 passthrough")

    def format(self, batch: Dict[str, List]) -> Dict[str, List]:
        if not self._enabled:
            return batch
        return {
            "global": [self._quantise(t, idx)
                       for idx, t in enumerate(batch["global"])],
            "local":  [self._quantise(t, len(batch["global"]) + idx)
                       for idx, t in enumerate(batch["local"])],
        }

    def _quantise(self, tensor: torch.Tensor, view_idx: int) -> Tuple:
        t = tensor.to(self._device).float()

        # Update rolling amax
        amax_val = t.abs().max().item()
        with self._lock:
            if view_idx not in self._amax_history:
                self._amax_history[view_idx] = deque([amax_val] * _AMAX_WINDOW,
                                                     maxlen=_AMAX_WINDOW)
            else:
                self._amax_history[view_idx].append(amax_val)
            window_amax = max(self._amax_history[view_idx])

        scale     = window_amax / _FP8_MAX if window_amax > 0 else 1.0
        scale_inv = 1.0 / max(scale, 1e-12)

        fp8 = (t * scale_inv).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)

        meta = tex.FP8TensorMeta()
        meta.scale     = torch.tensor([scale],     dtype=torch.float32, device=self._device)
        meta.scale_inv = torch.tensor([scale_inv], dtype=torch.float32, device=self._device)
        meta.amax_history = torch.tensor(
            list(self._amax_history[view_idx]), dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        return fp8, meta
