"""
dino_loader.memory
==================
In-memory data structures and GPU transfer utilities.

[MEM-1] Batch dataclass with metadata and masks.
[MEM-3] AsyncPrefetchIterator — genuine background prefetch with race-free
        exception handling (B1-FIX): future ownership transferred atomically
        under lock before result() is called outside it.
[MEM-4] FP8Formatter: no-op guard for dali_fp8_output=True path.

Note: Grace-Blackwell / NVL72 managed-memory paths have been removed.
      This loader targets B200, H200, H100 with standard PCIe topology.
      allocate_buffers() is retained as a utility but simplified to a single
      pinned-memory path.
"""

import contextlib
import logging
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from dino_loader.config import DINOAugConfig
from dino_env import ClusterTopology

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
        topo: Cluster topology (used for API compatibility only).
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
        self, cpu_batch: dict[str, list[torch.Tensor]]
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
        self, cpu_batch: dict[str, list[torch.Tensor]]
    ) -> dict[str, list[torch.Tensor]]:
        """Non-context-manager variant; caller must call wait() before use."""
        with torch.cuda.stream(self._stream):
            return {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }

    def wait(self) -> None:
        torch.cuda.current_stream().wait_stream(self._stream)


class AsyncPrefetchIterator:
    """Wraps a DALI iterator and pre-fetches the next batch on a background thread.

    Hides DALI decode latency behind GPU compute time.

    Thread model
    ------------
    One ThreadPoolExecutor(max_workers=1) thread is the sole consumer of
    next(self._iter); DALI iterators are not thread-safe.

    Error propagation
    -----------------
    The future is transferred to local ownership under lock (setting
    self._future = None atomically), then result() is called outside the lock.
    This prevents close() from racing with result() and ensures the iterator
    transitions cleanly to closed on any DALI decode failure.
    _submit() is only called on the successful path — never after an error.

    Shutdown
    --------
    Call close() explicitly or use as a context manager. The __del__ guard is
    intentionally limited to a best-effort non-blocking close to avoid deadlocks
    during interpreter shutdown.
    """

    _SENTINEL = object()

    def __init__(self, dali_iter: Any, h2d: H2DStream) -> None:
        self._iter     = dali_iter
        self._h2d      = h2d
        self._closed   = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dali-prefetch"
        )
        self._future: Future | None = None
        self._lock     = threading.Lock()
        self._submit()

    def __iter__(self) -> "AsyncPrefetchIterator":
        return self

    def __next__(self) -> Any:
        # Transfer future ownership atomically under lock.
        with self._lock:
            if self._closed or self._future is None:
                raise StopIteration
            fut          = self._future
            self._future = None

        # Wait for result outside the lock so close() can proceed concurrently.
        try:
            result = fut.result()
        except Exception:
            self.close()
            raise

        if result is self._SENTINEL:
            raise StopIteration

        # Only submit next fetch on success.
        self._submit()
        return result

    def __enter__(self) -> "AsyncPrefetchIterator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Signal shutdown and wait for the background thread to finish."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            fut          = self._future
            self._future = None

        if fut is not None:
            fut.cancel()
        # wait=False avoids blocking the caller (e.g. during error recovery).
        self._executor.shutdown(wait=False, cancel_futures=True)
        log.debug("AsyncPrefetchIterator closed")

    def __del__(self) -> None:
        # Best-effort, non-blocking cleanup only. Do not call shutdown(wait=True)
        # here — it can deadlock during interpreter teardown.
        with self._lock:
            if self._closed:
                return
            self._closed = True

    def _fetch_one(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            return self._SENTINEL

    def _submit(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._future = self._executor.submit(self._fetch_one)


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
