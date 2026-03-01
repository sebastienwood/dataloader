"""
dino_loader.memory
==================
In-memory data structures and GPU transfer utilities.

Changes vs previous version
----------------------------
[MEM-1] Batch dataclass enriched:
        - metadata: List[Optional[Dict]] — per-sample sidecar metadata.
          None for samples from shards without .json sidecars.
          Populated by MixingSource.pop_last_metadata().
        - masks: Optional[Any] — token mask tensor from MaskingGenerator
          (DinoV3 iBOT pattern).  None when no mask_generator is configured.

[MEM-2] allocate_buffers: ceiling sizes now use aug_cfg.max_global_crop_size
        and aug_cfg.max_local_crop_size (set at build time to the largest
        planned resolution) rather than global_crop_size / local_crop_size.
        This ensures pinned / managed buffers never need re-allocation during
        a resolution schedule.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from dino_loader.config       import DINOAugConfig
from dino_loader.distributed  import ClusterTopology

log = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    log.debug("transformer-engine not installed — FP8 output disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# Batch
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Batch:
    """
    One training batch with all views on GPU.

    Fields
    ------
    global_crops : List of 2 tensors (BF16 or FP8+meta) — large views.
    local_crops  : List of 8 tensors — small views.
    metadata     : Per-sample sidecar dicts (Optional[Dict], None if absent).
                   Length == batch_size.  Use to access quality_score, caption,
                   dedup_hash, etc. from WebDataset .json sidecars.  [MEM-1]
    masks        : Token mask tensor from MaskingGenerator, or None.  [MEM-1]
                   Shape: (batch_size, n_tokens) bool.
    """
    global_crops: List
    local_crops:  List
    metadata:     List[Optional[Dict]] = field(default_factory=list)   # [MEM-1]
    masks:        Optional[Any]        = None                           # [MEM-1]

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

    [MEM-2] Buffer dimensions use max_global_crop_size / max_local_crop_size
    (set to the ceiling of the resolution schedule) so that no re-allocation
    occurs when set_resolution() is called mid-training.

    Grace-Blackwell → managed memory, preferred on GPU (no H2D needed).
    PCIe            → pinned host memory (fast non-blocking H2D).
    """
    C = 3
    max_global = aug_cfg.max_global_crop_size
    max_local  = aug_cfg.max_local_crop_size

    def _make(max_size: int, n: int) -> List[torch.Tensor]:
        bufs = []
        for _ in range(n):
            if topo.is_grace_blackwell:
                t = torch.empty(
                    batch_size, C, max_size, max_size, dtype=dtype, device=device
                )
                try:
                    torch.cuda.memory.cudaMemAdvise(
                        t,
                        torch.cuda.memory.cudaMemAdviseSetPreferredLocation,
                        device.index,
                    )
                except AttributeError:
                    pass
            else:
                t = torch.empty(
                    batch_size, C, max_size, max_size, dtype=dtype
                ).pin_memory()
            bufs.append(t)
        return bufs

    return {
        "global": _make(max_global, aug_cfg.n_global_crops),
        "local":  _make(max_local,  aug_cfg.n_local_crops),
    }


# ══════════════════════════════════════════════════════════════════════════════
# H2D transfer
# ══════════════════════════════════════════════════════════════════════════════

class H2DStream:
    """
    Dedicated CUDA stream for host-to-device transfers.

    On Grace-Blackwell (NVLink-C2C), transfer is a no-op — managed memory
    is already accessible on GPU.  On PCIe, an async copy is issued on a
    dedicated stream to overlap with the compute stream.
    """

    def __init__(self, device: torch.device, topo: ClusterTopology) -> None:
        self._device = device
        self._topo   = topo
        self._stream = torch.cuda.Stream(device=device)
        self._c2c    = topo.is_grace_blackwell
        if self._c2c:
            log.info("H2DStream: NVLink-C2C — H2D is a no-op (managed memory)")
        else:
            log.info("H2DStream: PCIe path, dedicated CUDA stream allocated")

    @contextlib.contextmanager
    def transfer(self, cpu_batch: Dict[str, List[torch.Tensor]]):
        """
        Async H2D transfer context manager.

        Usage::

            with h2d.transfer({"global": [...], "local": [...]}) as gpu:
                loss = model(gpu["global"], gpu["local"])
        """
        if self._c2c:
            yield cpu_batch
            return

        with torch.cuda.stream(self._stream):
            gpu_batch = {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }
        torch.cuda.current_stream().wait_stream(self._stream)
        yield gpu_batch

    def send(self, cpu_batch: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """Non-context-manager variant; caller must call wait() before use."""
        if self._c2c:
            return cpu_batch
        with torch.cuda.stream(self._stream):
            return {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }

    def wait(self) -> None:
        torch.cuda.current_stream().wait_stream(self._stream)


# ══════════════════════════════════════════════════════════════════════════════
# Async prefetch iterator
# ══════════════════════════════════════════════════════════════════════════════

class AsyncPrefetchIterator:
    """Wraps a DALI iterator and pre-fetches one batch ahead on a thread."""

    def __init__(self, dali_iter, h2d: H2DStream) -> None:
        self._iter = dali_iter
        self._h2d  = h2d
        self._next = None
        self._prefetch()

    def _prefetch(self) -> None:
        try:
            self._next = next(self._iter)
        except StopIteration:
            self._next = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._next is None:
            raise StopIteration
        current = self._next
        self._prefetch()
        return current


# ══════════════════════════════════════════════════════════════════════════════
# FP8 formatter
# ══════════════════════════════════════════════════════════════════════════════

class FP8Formatter:
    """
    Quantise BF16 tensors to FP8 E4M3 using Transformer Engine.

    Uses a rolling amax window (length 16) matching TE's internal convention
    so that FP8TensorMeta objects can be passed directly into te.fp8_autocast().
    Falls back to a no-op identity when TE is not installed.
    """

    _AMAX_HISTORY = 16

    def __init__(self) -> None:
        if not HAS_TE:
            log.warning("transformer-engine not installed — FP8 output disabled, using BF16.")
        self._enabled = HAS_TE

    def quantise(self, tensor: torch.Tensor):
        """Return (fp8_tensor, fp8_meta) or the original BF16 tensor."""
        if not self._enabled:
            return tensor
        try:
            fp8_meta = te.fp8.FP8TensorMeta()
            fp8_meta.scale     = torch.ones(1, dtype=torch.float32, device=tensor.device)
            fp8_meta.scale_inv = torch.ones(1, dtype=torch.float32, device=tensor.device)
            fp8_meta.amax_history = torch.zeros(
                self._AMAX_HISTORY, dtype=torch.float32, device=tensor.device
            )
            fp8_tensor = te.fp8.cast_to_fp8(
                tensor.contiguous(),
                fp8_meta,
                0,
                te.fp8.Float8Tensor,
            )
            return fp8_tensor, fp8_meta
        except Exception as exc:
            log.warning("FP8 quantisation failed (%s) — returning BF16.", exc)
            return tensor
