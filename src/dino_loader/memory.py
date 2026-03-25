"""dino_loader.memory
==================
In-memory data structures and GPU transfer utilities.

[MEM-1] Batch dataclass with metadata and masks.
[MEM-4] FP8Formatter: no-op guard for dali_fp8_output=True path.

AsyncPrefetchIterator has been removed: DALI's internal cpu_queue / gpu_queue
pipeline (prefetch_queue_depth) already provides equivalent double-buffering
natively — no application-level thread is needed.  Increasing dali_cpu_queue
to ≥ 16 (see LoaderConfig) fully hides Lustre / extraction latency behind GPU
compute, which is what AsyncPrefetchIterator was manually replicating.

Note: Grace-Blackwell / NVL72 managed-memory paths have been removed.
      This loader targets B200, H200, H100 with standard PCIe topology.
      allocate_buffers() is retained as a utility but simplified to a single
      pinned-memory path.
"""

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from dino_env import ClusterTopology

from dino_loader.config import DINOAugConfig

log = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    log.debug("transformer-engine not installed — FP8 output disabled.")


@dataclass
class Batch:
    """One training batch with all views on GPU.

    Attributes:
        global_crops: List of global-crop tensors (BF16 or FP8).
        local_crops: List of local-crop tensors.
        metadata: Per-sample sidecar dicts. None when absent.
        masks: iBOT token mask tensor (bool, shape batch×n_tokens) or None.
            Generated on CPU post-DALI; cannot be fused into DALI because
            masking operates on ViT patch indices, not pixel values.

    """

    global_crops: list
    local_crops:  list
    metadata:     list[dict | None] = field(default_factory=list)
    masks:        Any | None        = None

    def __iter__(self):
        """Convenience unpack as (global_crops, local_crops)."""
        return iter((self.global_crops, self.local_crops))


def allocate_buffers(
    batch_size: int,
    aug_cfg:    DINOAugConfig,
    topo:       ClusterTopology,
    device:     torch.device,
    dtype:      torch.dtype = torch.bfloat16,
) -> dict[str, list[torch.Tensor]]:
    """Allocate pinned host buffers sized to max crop dimensions.

    Uses max_global_crop_size / max_local_crop_size so no re-allocation
    occurs when set_resolution() is called mid-training.

    Pinned host memory is used for efficient non-blocking DMA to GPU via
    the H2DStream. The topo parameter is accepted for API compatibility
    but all supported hardware (B200, H200, H100) uses the same PCIe path.

    Args:
        batch_size: Per-GPU batch size.
        aug_cfg: Augmentation config providing max crop dimensions.
        topo: Cluster topology (accepted for API compat, not used).
        device: Target CUDA device.
        dtype: Tensor dtype (default bfloat16).

    Returns:
        Dict with 'global' and 'local' keys, each a list of pinned tensors.

    """
    def _buf(size: int) -> list[torch.Tensor]:
        return [
            torch.zeros(batch_size, 3, size, size, dtype=dtype).pin_memory()
            for _ in range(2)
        ]

    return {
        "global": _buf(aug_cfg.max_global_crop_size),
        "local":  _buf(aug_cfg.max_local_crop_size),
    }


class H2DStream:
    """Async host-to-device transfer on a dedicated CUDA stream."""

    def __init__(self, device: torch.device, topo: ClusterTopology) -> None:
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        log.info("H2DStream: PCIe path, dedicated CUDA stream on %s", device)

    @contextlib.contextmanager
    def transfer(
        self, cpu_batch: dict[str, list[torch.Tensor]],
    ) -> Iterator[dict[str, list[torch.Tensor]]]:
        """Async H2D transfer context manager.

        Usage::

            with h2d.transfer({"global": [...], "local": [...]}) as gpu:
                loss = model(gpu["global"], gpu["local"])
        """
        with torch.cuda.stream(self._stream):
            gpu_batch = {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }
        torch.cuda.current_stream().wait_stream(self._stream)
        yield gpu_batch

    def send(
        self, cpu_batch: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        """Non-context-manager variant; caller must call wait() before use."""
        with torch.cuda.stream(self._stream):
            return {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }

    def wait(self) -> None:
        """Synchronise the dedicated stream with the current CUDA stream."""
        torch.cuda.current_stream().wait_stream(self._stream)


class FP8Formatter:
    """Quantise BF16 tensors to FP8 E4M3 using Transformer Engine.

    Uses a rolling amax window (length 16) matching TE's internal convention
    so that FP8TensorMeta objects can be passed directly into te.fp8_autocast().

    When LoaderConfig.dali_fp8_output=True, loader.py does NOT construct this
    class (self._fp8 = None), so quantise() is never called from the hot path.
    Falls back to identity when TE is not installed.
    """

    _AMAX_WINDOW = 16

    def __init__(self) -> None:
        if HAS_TE:
            self._meta = te.fp8.FP8TensorMeta()
            self._meta.scale     = torch.ones(1)
            self._meta.scale_inv = torch.ones(1)
            self._meta.amax_history = torch.zeros(self._AMAX_WINDOW, 1)
            log.info("FP8Formatter: Transformer Engine path active")
        else:
            self._meta = None
            log.info("FP8Formatter: TE not installed — identity (no-op)")

    def quantise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantise *tensor* to FP8 E4M3, or return it unchanged if already FP8."""
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            log.warning(
                "FP8Formatter.quantise called on already-FP8 tensor (dtype=%s) — "
                "returning unchanged. Check dali_fp8_output config.",
                tensor.dtype,
            )
            return tensor
        if not HAS_TE or self._meta is None:
            return tensor
        return te.fp8.cast_to_fp8(tensor, self._meta, 0, te.fp8.Float8Tensor)
