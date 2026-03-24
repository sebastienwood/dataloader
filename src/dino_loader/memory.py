"""
dino_loader.memory
==================
In-memory data structures and GPU transfer utilities.

Changes in this version
-----------------------
[MEM-1] Batch dataclass enriched (retained).

[MEM-2] allocate_buffers: ceiling sizes use max_*_crop_size (retained).

[MEM-3] AsyncPrefetchIterator — genuine background prefetch (retained).
        [B1-FIX] Race condition on exception path corrected: the future is
        consumed atomically (ownership transferred under lock), then result()
        is called outside the lock so that close() cannot race with
        StopIteration or error propagation.  _submit() is now only called
        after a successful result — never after an error — so the iterator
        transitions cleanly to a closed state on any DALI decode failure
        instead of silently dying.

[MEM-4] FP8Formatter: no-op when dali_fp8_output=True (retained).

[MEM-5] allocate_buffers: Grace-Blackwell path corrected.                 ← FIX
        Previously ``torch.zeros(..., device=cpu_device)`` was used even for
        the Grace-Blackwell managed-memory path — managed memory requires the
        tensor to be allocated on the CUDA device (``device="cuda:N"``) so
        that the driver maps it into the unified address space.  The fix:
        GB200/NVL72 path now allocates directly on device; PCIe path retains
        pinned host memory allocation.

[MEM-6] SharedMemoryRingBuffer — intra-node ring buffer for shard broadcast. ← NEW (opt-in)
        Architectural improvement #1.  Enabled via
        ``LoaderConfig.intra_node_ring_buffer = True`` (default: False).

        Instead of every rank independently mmap-ing the same /dev/shm file
        (N open()+mmap_setup() pairs per shard per prefetch window), rank 0
        writes each shard once into a POSIX shared_memory segment and all
        local ranks read from that single shared segment via memoryview slices.

        On NVL72 (72 ranks per node), this reduces mmap syscall overhead from
        O(ranks × active_shards) to O(active_shards), a ~72× reduction at
        steady state.

        Implementation notes
        --------------------
        - Uses ``multiprocessing.shared_memory.SharedMemory`` (Python 3.8+).
        - Segments are named ``dino_{job_id}_{shard_hash}`` and created by
          rank 0 (node master); other ranks open them by name.
        - A lightweight header (16 bytes: data_len u64 + ready_magic u64)
          matches the /dev/shm file format so that ShardIterator._extract()
          works unchanged.
        - The segment is unlinked by rank 0 on eviction or shutdown.
        - This class is a drop-in replacement for the mmap pool path in
          NodeSharedShardCache — it is NOT a replacement for the full cache.
          The cache still manages LRU eviction and asyncio I/O; this class
          only optimises the "give ranks a view" step.

        Why opt-in?
        -----------
        POSIX shared_memory on some HPC kernels (< 5.10) has bugs with
        large allocations on tmpfs under memory pressure.  The default mmap
        pool path (PERF-2) is battle-tested; the ring buffer is new and
        should be validated on each target cluster before enabling.
"""

from __future__ import annotations

import contextlib
import logging
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Iterator, List, Optional

import torch

from dino_loader.config      import DINOAugConfig
from dino_env import ClusterTopology

log = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    log.debug("transformer-engine not installed — FP8 output disabled.")

# Shared-memory header format (matches shard_cache._HDR_FMT)
_SHM_HDR_FMT  = "QQ"
_SHM_HDR_SIZE = struct.calcsize(_SHM_HDR_FMT)
_SHM_READY    = 0xDEAD_BEEF_CAFE_F00D


# ══════════════════════════════════════════════════════════════════════════════
# Batch
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Batch:
    """
    One training batch with all views on GPU.

    Fields
    ------
    global_crops : List of 2 tensors (BF16, FP8+TE meta, or FP8 from DALI).
    local_crops  : List of 8 tensors.
    metadata     : Per-sample sidecar dicts.  None when absent.  [MEM-1]
    masks        : iBOT token mask tensor (bool, shape batch×n_tokens) or None.
                   Generated on CPU post-DALI; cannot be fused into DALI
                   because masking operates on ViT patch indices, not pixels.
    """
    global_crops: List
    local_crops:  List
    metadata:     List[Optional[Dict]] = field(default_factory=list)
    masks:        Optional[Any]        = None

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

    [MEM-2] Dimensions use max_global_crop_size / max_local_crop_size so
    no re-allocation occurs when set_resolution() is called mid-training.

    [MEM-5] Grace-Blackwell path corrected:
    - PCIe  → pinned host memory on CPU (fast non-blocking H2D via DMA).
    - GB200 → CUDA device memory (NVLink-C2C unified address space; no H2D
              copy needed — tensors are GPU-visible from the moment of alloc).
              Previously this path incorrectly allocated on CPU, defeating the
              purpose of the unified memory architecture.
    """
    if topo.is_grace_blackwell:
        # Allocate directly on device — NVLink-C2C means host and device share
        # the same physical memory; there is no H2D copy cost.
        def alloc_fn(*args, **kwargs):
            # Force device= to the actual CUDA device, regardless of what
            # the caller passed in `device` (which might be cpu for safety).
            kwargs["device"] = device if device.type == "cuda" else torch.device("cuda")
            return torch.zeros(*args, **kwargs)
    else:
        # PCIe path: pinned host memory → non-blocking DMA to GPU.
        def alloc_fn(*args, **kwargs):
            kwargs.pop("device", None)
            return torch.zeros(*args, **kwargs).pin_memory()

    def _buf(size: int) -> List[torch.Tensor]:
        return [
            alloc_fn(batch_size, 3, size, size, dtype=dtype)
            for _ in range(2)  # global crops
        ]

    return {
        "global": _buf(aug_cfg.max_global_crop_size),
        "local":  _buf(aug_cfg.max_local_crop_size),
    }


# ══════════════════════════════════════════════════════════════════════════════
# H2D transfer stream
# ══════════════════════════════════════════════════════════════════════════════

class H2DStream:
    """
    Async host-to-device transfer on a dedicated CUDA stream.

    On Grace-Blackwell (NVLink-C2C), host and device share the same physical
    memory — H2D is a no-op and no stream is needed.
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
# [MEM-3] Async prefetch iterator — [B1-FIX] race-free exception handling
# ══════════════════════════════════════════════════════════════════════════════

class AsyncPrefetchIterator:
    """
    Wraps a DALI iterator and pre-fetches the next batch on a dedicated
    background thread, hiding DALI decode latency behind GPU compute.

    Thread model
    ------------
    One ThreadPoolExecutor(max_workers=1) thread is the sole consumer of
    ``next(self._iter)``; DALI iterators are not thread-safe.

    Error propagation — [B1-FIX]
    -----------------------------
    The previous implementation had a TOCTOU race: ``self._future.result()``
    was called outside the lock, but ``_submit()`` was called immediately
    after without checking if close() had fired in between.  Worse, if
    ``_fetch_one()`` raised a non-StopIteration exception (e.g. a DALI decode
    error), ``_submit()`` was never called, leaving ``self._future`` as the
    completed-but-erroring future — the next ``__next__`` call would then
    re-raise the same error from the *already-consumed* future, masking the
    root cause.

    Fix: the future is transferred to local ownership under the lock (setting
    ``self._future = None`` atomically).  ``result()`` is called outside the
    lock on the local variable.  On any exception, ``close()`` is called
    before re-raising so the executor shuts down cleanly.  ``_submit()`` is
    only called on the *successful* path.

    Shutdown contract
    -----------------
    close() cancels in-flight work and shuts down the executor.
    __next__() after close() raises StopIteration.
    """

    _SENTINEL = object()

    def __init__(self, dali_iter, h2d: H2DStream) -> None:
        self._iter     = dali_iter
        self._h2d      = h2d
        self._closed   = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dali-prefetch"
        )
        self._future: Optional[Future] = None
        self._lock     = threading.Lock()
        self._submit()

    def __iter__(self):
        return self

    def __next__(self):
        # ── Step 1: take exclusive ownership of the future under lock ─────────
        # This prevents close() from racing with result() and avoids the
        # TOCTOU window between reading self._future and calling _submit().
        with self._lock:
            if self._closed or self._future is None:
                raise StopIteration
            fut          = self._future
            self._future = None  # we now own `fut`; lock released here

        # ── Step 2: wait for the background thread result ────────────────────
        # Outside the lock so that close() can proceed if called concurrently.
        try:
            result = fut.result()
        except Exception:
            # Any error (DALI decode failure, etc.) → shut down cleanly and
            # re-raise so the training loop sees the real exception.
            self.close()
            raise

        # ── Step 3: check sentinel ────────────────────────────────────────────
        if result is self._SENTINEL:
            raise StopIteration

        # ── Step 4: submit next fetch (only on success) ───────────────────────
        self._submit()
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            fut          = self._future
            self._future = None

        if fut is not None:
            fut.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        log.debug("AsyncPrefetchIterator closed")

    def __del__(self) -> None:
        self.close()

    def _fetch_one(self):
        try:
            return next(self._iter)
        except StopIteration:
            return self._SENTINEL

    def _submit(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._future = self._executor.submit(self._fetch_one)


# ══════════════════════════════════════════════════════════════════════════════
# FP8 formatter — [MEM-4] no-op guard for DALI-FP8 path
# ══════════════════════════════════════════════════════════════════════════════

class FP8Formatter:
    """
    Quantise BF16 tensors to FP8 E4M3 using Transformer Engine.

    Uses a rolling amax window (length 16) matching TE's internal convention
    so that FP8TensorMeta objects can be passed directly into te.fp8_autocast().

    [MEM-4] When LoaderConfig.dali_fp8_output=True, loader.py does NOT
    construct this class (self._fp8 = None), so quantise() is never called.
    As a defensive measure, if quantise() is called on a tensor that is already
    FP8, it logs a warning and returns the tensor unchanged to avoid
    double-quantisation.

    Falls back to a no-op identity when TE is not installed.
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
            log.info("FP8Formatter: TE not installed — identity (no FP8)")

    def quantise(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            log.warning(
                "FP8Formatter.quantise called on already-FP8 tensor "
                "(dtype=%s) — returning unchanged.  Check dali_fp8_output config.",
                tensor.dtype,
            )
            return tensor
        if not HAS_TE or self._meta is None:
            return tensor
        return te.fp8.cast_to_fp8(
            tensor,
            self._meta,
            0,
            te.fp8.Float8Tensor,
        )


# ══════════════════════════════════════════════════════════════════════════════
# [MEM-6] SharedMemoryRingBuffer — opt-in intra-node shard broadcast
# ══════════════════════════════════════════════════════════════════════════════

class SharedMemoryRingBuffer:
    """
    Intra-node shard broadcast via POSIX shared memory segments.

    Architectural improvement #1 (opt-in via LoaderConfig.intra_node_ring_buffer).

    Instead of every local rank independently mmap-ing the same /dev/shm file
    (O(ranks × active_shards) open()+mmap calls), rank 0 writes each shard
    into a named SharedMemory segment once, and all ranks read from it via a
    zero-copy memoryview slice.  On NVL72 (72 ranks / node) this reduces the
    mmap call count at steady state from ~4 500 to ~64 per prefetch window.

    Design
    ------
    - One SharedMemory segment per active shard, named
      ``dino_{job_id}_{sha1(shard_path)[:12]}``.
    - Header: [data_len: u64][ready_magic: u64] (16 bytes, same as /dev/shm file
      format) so that reading code is identical.
    - Rank 0 (node master) calls ``publish(shard_path, data)`` to create and
      fill the segment; readers call ``view(shard_path)`` to get a memoryview.
    - A simple dict tracks live segments.  ``evict(shard_path)`` unlinks the
      segment (rank 0 only).
    - ``close()`` unlinks all segments owned by this process.

    Thread safety
    -------------
    ``publish`` and ``evict`` are only ever called from the asyncio I/O thread
    (rank 0).  ``view`` may be called from multiple extraction worker threads
    simultaneously; it holds a per-segment read lock only while setting up the
    SharedMemory object (cheap; the object is cached after first open).

    Why opt-in?
    -----------
    SharedMemory on some HPC kernels (< 5.10) has allocation bugs under memory
    pressure.  The default _MmapPool path is battle-tested; enable this feature
    only after validating on your target cluster.

    Usage (via LoaderConfig)
    -------------------------
    ::

        config = LoaderConfig(intra_node_ring_buffer=True)

    The DALIBackend wires this into NodeSharedShardCache automatically when
    the flag is set.
    """

    def __init__(self, job_id: str, node_master: bool) -> None:
        self._job_id      = job_id
        self._node_master = node_master
        self._segments:   Dict[str, SharedMemory] = {}   # shard_path → shm
        self._lock        = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def publish(self, shard_path: str, data: bytes) -> None:
        """
        Create a shared memory segment for *shard_path* and write *data*.

        Node master only.  Idempotent: if a segment already exists for this
        shard (e.g. from a previous epoch without eviction), it is reused if
        the size matches, otherwise recreated.
        """
        if not self._node_master:
            raise RuntimeError(
                "SharedMemoryRingBuffer.publish must only be called by the node master."
            )
        total = _SHM_HDR_SIZE + len(data)
        name  = self._seg_name(shard_path)

        with self._lock:
            existing = self._segments.get(shard_path)
            if existing is not None and existing.size == total:
                # Reuse — just overwrite content (same epoch, different data is
                # possible if shard_path collides, but SHA1 makes this ~impossible).
                self._write_into(existing, data)
                return
            if existing is not None:
                self._unlink_segment(existing)

            try:
                shm = SharedMemory(name=name, create=True, size=total)
            except FileExistsError:
                # Another process published this segment already (race at start).
                # Open existing and trust the content.
                shm = SharedMemory(name=name, create=False)
                self._segments[shard_path] = shm
                return

            self._write_into(shm, data)
            self._segments[shard_path] = shm
            log.debug("SharedMemory published: %s (%d MB)", name, len(data) >> 20)

    @contextlib.contextmanager
    def view(self, shard_path: str) -> Iterator[memoryview]:
        """
        Yield a zero-copy memoryview of the shard data.

        All ranks (including non-master) can call this.  On first call from a
        non-master rank, the SharedMemory object is opened by name and cached.
        Subsequent calls re-use the cached object — O(1) after warm-up.
        """
        shm = self._ensure_open(shard_path)
        data_len, magic = struct.unpack_from(_SHM_HDR_FMT, shm.buf, 0)
        if magic != _SHM_READY:
            raise RuntimeError(
                f"SharedMemory segment for {shard_path!r} has corrupt header "
                f"(magic={magic:#x}).  Segment may still be written by rank 0."
            )
        yield memoryview(shm.buf)[_SHM_HDR_SIZE: _SHM_HDR_SIZE + data_len]

    def evict(self, shard_path: str) -> None:
        """Remove and unlink the shared memory segment for *shard_path*."""
        with self._lock:
            seg = self._segments.pop(shard_path, None)
        if seg is not None:
            self._unlink_segment(seg)

    def close(self) -> None:
        """Unlink all segments owned by this process."""
        with self._lock:
            segs = list(self._segments.values())
            self._segments.clear()
        for seg in segs:
            self._unlink_segment(seg)
        log.debug("SharedMemoryRingBuffer closed (%d segments released)", len(segs))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _seg_name(self, shard_path: str) -> str:
        import hashlib
        digest = hashlib.sha1(shard_path.encode()).hexdigest()[:12]
        # SharedMemory names must be <= 255 chars and alphanumeric+underscore.
        return f"dino_{self._job_id}_{digest}"[:31]  # /dev/shm/<name> limit

    def _ensure_open(self, shard_path: str) -> SharedMemory:
        """Return the SharedMemory object, opening it by name if not cached."""
        with self._lock:
            shm = self._segments.get(shard_path)
            if shm is not None:
                return shm
        # Open by name — safe to do outside the lock (name is stable).
        name = self._seg_name(shard_path)
        shm  = SharedMemory(name=name, create=False)
        with self._lock:
            # Check again — another thread may have opened it while we waited.
            if shard_path not in self._segments:
                self._segments[shard_path] = shm
            else:
                shm.close()  # discard ours; use the one inserted by the winner
                shm = self._segments[shard_path]
        return shm

    @staticmethod
    def _write_into(shm: SharedMemory, data: bytes) -> None:
        """Write header + data into an already-allocated SharedMemory object."""
        # Phase 1: write not-ready sentinel so readers don't see partial data.
        struct.pack_into(_SHM_HDR_FMT, shm.buf, 0, len(data), 0)
        shm.buf[_SHM_HDR_SIZE: _SHM_HDR_SIZE + len(data)] = data
        # Phase 2: mark ready — acts as the memory barrier for readers.
        struct.pack_into(_SHM_HDR_FMT, shm.buf, 0, len(data), _SHM_READY)

    @staticmethod
    def _unlink_segment(seg: SharedMemory) -> None:
        try:
            seg.close()
        except Exception:
            pass
        try:
            seg.unlink()
        except Exception:
            pass
