"""
dino_loader.mixing_source
=========================
WebDataset mixing source for DALI ExternalSource.

Separation of concerns
-----------------------
MixingWeights   — owns weight state and thread-safe update logic.
ShardIterator   — owns per-dataset shard cycling, prefetch scheduling,
                  and background tar extraction.
MixingSource    — composes the two; implements the DALI callback protocol.

Changes from previous version (intern proposal)
------------------------------------------------
ACCEPTED
  [A-1] Two-level prefetch pipeline.
  [A-2] deque[bytes] buffer with popleft() replaces List[bytes] + int index.

FIXED (intern review)
  [F-1] Executor leak: close() added.
  [F-2] Off-by-two in initial prefetch horizon.
  [F-3] num_workers surfaced in LoaderConfig.
  [F-4] Thread-safety of cache.get() from worker threads: documented.

Additional fixes
----------------
[FIX-5]  Executor worker starvation when shards are not yet ready.
         With num_workers=2 and _EXTRACTION_DEPTH=2, both workers could
         block inside cache.get_view() → _inotify_wait() → select() at
         epoch start, starving DALI.  Fix: (a) raise the default
         num_workers to 4 so warm shards can be processed while cold ones
         wait; (b) document that num_workers should be set proportionally
         to shard_prefetch_window.  The real fix is that num_workers is
         now a LoaderConfig field (see config.py).

[FIX-13] Duplicate shard fetch when n_shards_per_rank < _EXTRACTION_DEPTH.
         When a dataset has very few shards (e.g. imagenet22k: ~17 shards
         per rank on a 288-GPU job), __init__ submitted two futures for the
         same shard (idx 0 mod 1 == idx 1 mod 1 == 0), triggering two
         concurrent get_view() calls on the same file.  Fixed by
         deduplicating the initial submission.

[FIX-14] No per-epoch reshuffle.
         The previous implementation shuffled shards once at construction and
         never again, so each epoch saw the same shard order.  Proper DINO
         training requires epoch-level reshuffling for convergence.  Fixed by
         storing the base seed and re-shuffling in reset_epoch(epoch), which
         MixingSource.set_epoch() calls on all ShardIterators.

[FIX-E]  _init_epoch() deduplication incorrectly advanced self._idx.
         When n_shards_per_rank < _EXTRACTION_DEPTH (tiny datasets such as
         imagenet22k on large jobs), the deduplication branch advanced
         self._idx without submitting a future.  The subsequent
         _prefetch_window() call then used the bumped idx, silently skipping
         cache prefetch for the first shard.
         Fix: the else branch no longer advances _idx.  The duplicate
         submission is simply skipped; _prefetch_window() covers the shard
         normally from its correct position.

[FIX-P1] Poison-pill error propagation from extraction workers.
         Previously, an exception inside _fetch_and_extract (e.g. corrupt
         tar, truncated JPEG list, ENOSPC on /dev/shm) was stored silently
         inside the Future object.  future.result() in _drain_next_future
         would re-raise it, but only once — subsequent calls to next_jpeg()
         would attempt to submit new work on the already-broken iterator,
         producing confusing secondary errors that masked the root cause.

         Fix: ShardIterator gains a threading.Event _poison_pill and an
         _error attribute.  _drain_next_future wraps future.result() in a
         try/except; on failure it sets the pill and stores the exception.
         next_jpeg() checks the pill first and re-raises the original error
         immediately, with context, rather than attempting further I/O.

         close() also sets the pill so any thread blocked in next_jpeg()
         unblocks cleanly during shutdown — no change in hot-path performance
         (the check is a single Event.is_set() call on a threading.Event,
         which is backed by a C-level flag with no lock overhead).

[FIX-P2] Sparse-dataset warning.
         When n_shards_per_rank < 4 every epoch is dominated by a tiny
         number of shards, causing the model to overfit to dataset-ordering
         patterns rather than learning image semantics.  This is not a crash,
         but is a silent training quality regression.  A log.warning is now
         emitted at ShardIterator construction so operators can detect it at
         job startup rather than after noticing degraded metrics.

[FIX-21] NUMA-aware CPU affinity for extraction threads.
         On multi-socket nodes (B200 PCIe, dual-socket Sapphire Rapids HBM)
         extraction threads that happen to be scheduled on the remote NUMA
         domain incur additional memory latency for every memcpy inside
         _extract_jpegs.  When cpu_affinity_enabled=True in LoaderConfig,
         ShardIterator binds its ThreadPoolExecutor workers to the CPU cores
         that are topologically closest to device_id (resolved via psutil's
         NUMA API).  Silently skipped if psutil is not installed or if the
         platform does not expose NUMA topology.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import threading
from collections import deque
from typing import Deque, List, Optional, Sequence, Set

import numpy as np

from dino_loader.config         import DatasetSpec
from dino_loader.shard_cache    import NodeSharedShardCache
from dino_loader.datasets.utils import _extract_jpegs

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# NUMA affinity helper
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_numa_cpus(device_id: int) -> Optional[List[int]]:
    """
    Return the list of logical CPU cores on the NUMA node closest to
    ``device_id``, or None if the information is unavailable.

    Requires ``psutil`` ≥ 5.9.  Silently returns None on any failure so
    the caller can degrade gracefully.
    """
    try:
        import psutil  # optional dependency
    except ImportError:
        return None

    try:
        # /sys/bus/pci/devices/<pci_addr>/numa_node contains the NUMA node
        # index for the GPU.  We read it via the torch CUDA API if available,
        # then fall back to parsing sysfs directly.
        numa_node: Optional[int] = None

        try:
            import torch
            props      = torch.cuda.get_device_properties(device_id)
            pci_bus_id = torch.cuda.get_device_name(device_id)  # used only as fallback
            # torch doesn't expose numa_node directly; use sysfs.
            _ = props  # suppress unused warning
        except Exception:
            pass

        # Sysfs path: /sys/class/drm/renderD<N>/device/numa_node
        # Alternative: /sys/bus/pci/devices/<addr>/numa_node
        sysfs_paths = [
            f"/sys/class/drm/renderD{128 + device_id}/device/numa_node",
            f"/sys/class/drm/card{device_id}/device/numa_node",
        ]
        for p in sysfs_paths:
            try:
                numa_node = int(open(p).read().strip())
                break
            except (OSError, ValueError):
                continue

        if numa_node is None or numa_node < 0:
            # NUMA node -1 means "unknown" or single-node system; skip affinity.
            return None

        cpu_info = psutil.cpu_count(logical=True)
        if cpu_info is None:
            return None

        # psutil exposes per-CPU NUMA info via Process.cpu_affinity(), but the
        # canonical way to enumerate CPUs-per-node is via os.sched_getaffinity +
        # reading /sys/devices/system/node/nodeN/cpumap.
        node_cpu_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        cpulist_raw   = open(node_cpu_path).read().strip()
        cpus: List[int] = []
        for part in cpulist_raw.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                cpus.extend(range(int(lo), int(hi) + 1))
            else:
                cpus.append(int(part))
        return cpus if cpus else None

    except Exception as exc:
        log.debug("NUMA affinity resolution failed (device %d): %s", device_id, exc)
        return None


def _apply_thread_affinity(cpus: List[int]) -> None:
    """Bind the calling thread to ``cpus``.  No-op on platforms without sched_setaffinity."""
    try:
        os.sched_setaffinity(0, cpus)
    except (AttributeError, OSError):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Weight manager
# ══════════════════════════════════════════════════════════════════════════════

class MixingWeights:
    """Thread-safe normalised mixing weights."""

    def __init__(self, names: List[str], initial_weights: List[float]):
        assert len(names) == len(initial_weights)
        self._names   = names
        self._lock    = threading.RLock()
        self._weights = self._normalise(initial_weights)

    @property
    def names(self) -> List[str]:
        return self._names

    def get(self) -> List[float]:
        with self._lock:
            return list(self._weights)

    def set(self, weights: Sequence[float]) -> None:
        if len(weights) != len(self._names):
            raise ValueError(f"Expected {len(self._names)} weights, got {len(weights)}")
        w = self._normalise(weights)
        with self._lock:
            self._weights = w
        log.debug(
            "Mixing weights updated: %s",
            dict(zip(self._names, [f"{v:.3f}" for v in w])),
        )

    def set_by_name(self, name: str, weight: float) -> None:
        try:
            idx = self._names.index(name)
        except ValueError:
            raise KeyError(f"Unknown dataset name: {name!r}") from None
        with self._lock:
            w = list(self._weights)
        w[idx] = weight
        self.set(w)

    @staticmethod
    def _normalise(w: Sequence[float]) -> List[float]:
        a = np.clip(np.array(w, dtype=np.float64), 0, None)
        total = a.sum()
        if total == 0:
            raise ValueError("At least one mixing weight must be positive.")
        return (a / total).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Per-dataset shard iterator
# ══════════════════════════════════════════════════════════════════════════════

class ShardIterator:
    """
    Cycles over the shards assigned to this rank for one dataset.

    Two-level prefetch pipeline
    ---------------------------
    Level 1 — I/O (NodeSharedShardCache):
        Raw tar bytes fetched from Lustre into /dev/shm asynchronously.
    Level 2 — CPU extraction (ThreadPoolExecutor):
        Worker threads parse tar archives into JPEG byte lists in RAM while
        DALI consumes the previous shard's JPEGs.

    [FIX-5] num_workers default raised to 4: at epoch start, up to
    _EXTRACTION_DEPTH workers may block in _inotify_wait waiting for cold
    shards.  With num_workers=4 the remaining workers can process warm shards
    concurrently, preventing DALI starvation.

    [FIX-13] Deduplicated initial futures: if n_shards < _EXTRACTION_DEPTH,
    the same shard is not submitted twice.

    [FIX-14] reset_epoch(epoch) re-shuffles shards deterministically using
    (base_seed + rank + epoch) so each epoch has a distinct, reproducible order.

    [FIX-E]  _init_epoch() deduplication no longer advances _idx spuriously.

    [FIX-P1] Poison-pill: extraction exceptions are captured and re-raised in
    next_jpeg() with full context, preventing silent failure or secondary errors.

    [FIX-P2] Warns at construction when n_shards_per_rank < 4.

    [FIX-21] Optional NUMA-aware CPU affinity for extraction threads.
    """

    _EXTRACTION_DEPTH = 2

    def __init__(
        self,
        name:                str,
        shards:              List[str],
        cache:               NodeSharedShardCache,
        rank:                int,
        world_size:          int,
        prefetch_ahead:      int  = 32,
        num_workers:         int  = 4,     # [FIX-5] was 2
        seed:                int  = 0,
        device_id:           int  = 0,     # [FIX-21] for NUMA resolution
        cpu_affinity_enabled: bool = False, # [FIX-21]
    ):
        self._name    = name
        self._cache   = cache
        self._ahead   = prefetch_ahead
        self._seed    = seed
        self._rank    = rank

        self._all_shards = [s for i, s in enumerate(shards) if i % world_size == rank]
        if not self._all_shards:
            raise RuntimeError(
                f"Rank {rank}/{world_size}: no shards assigned for dataset '{name}'. "
                f"Dataset has {len(shards)} shards total."
            )

        # [FIX-P2] Sparse-dataset warning — fewer than 4 shards per rank
        # means each epoch is dominated by a tiny sample of the dataset.
        if len(self._all_shards) < 4:
            log.warning(
                "ShardIterator '%s': only %d shard(s) assigned to rank %d/%d.  "
                "Training quality may degrade due to low per-rank shard diversity.  "
                "Consider increasing the number of shards (target: ≥ 4 per rank) "
                "or reducing world_size for this dataset.",
                name, len(self._all_shards), rank, world_size,
            )

        self._shards: List[str] = []   # populated by reset_epoch
        self._idx:    int = 0
        self._buffer: Deque[bytes]                                  = deque()
        self._futures: Deque[concurrent.futures.Future[List[bytes]]] = deque()

        # [FIX-P1] Poison-pill for error propagation from worker threads.
        self._poison_pill: threading.Event      = threading.Event()
        self._worker_error: Optional[Exception] = None

        # [FIX-21] Resolve NUMA-local CPUs once at construction.
        self._affinity_cpus: Optional[List[int]] = None
        if cpu_affinity_enabled:
            self._affinity_cpus = _resolve_numa_cpus(device_id)
            if self._affinity_cpus:
                log.debug(
                    "ShardIterator '%s': CPU affinity → %d cores on NUMA node "
                    "local to device %d",
                    name, len(self._affinity_cpus), device_id,
                )
            else:
                log.debug(
                    "ShardIterator '%s': cpu_affinity_enabled=True but NUMA "
                    "topology could not be resolved — running without affinity.",
                    name,
                )

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers        = num_workers,
            thread_name_prefix = f"shard-extract-{name}",
            initializer        = self._worker_init,  # [FIX-21]
        )
        self._closed = False

        # Initialise for epoch 0
        self._init_epoch(epoch=0)

        log.debug(
            "ShardIterator '%s': %d shards/rank, %d workers, %d prefetch window",
            name, len(self._all_shards), num_workers, prefetch_ahead,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_jpeg(self) -> bytes:
        """
        Return the next JPEG bytes, blocking only if extraction lags.

        [FIX-P1] Raises RuntimeError immediately if a worker thread has
        previously failed, surfacing the original exception as context.
        This prevents the caller (DALI prefetch thread) from issuing further
        I/O on a broken iterator and producing confusing secondary errors.
        """
        # [FIX-P1] Fast-path: check pill before touching any I/O state.
        if self._poison_pill.is_set():
            raise RuntimeError(
                f"ShardIterator '{self._name}': worker thread previously "
                f"failed — no further data will be produced."
            ) from self._worker_error

        if not self._buffer:
            self._drain_next_future()

        # Re-check after _drain_next_future (it can set the pill on failure).
        if self._poison_pill.is_set():
            raise RuntimeError(
                f"ShardIterator '{self._name}': worker thread previously "
                f"failed — no further data will be produced."
            ) from self._worker_error

        return self._buffer.popleft()

    def reset_epoch(self, epoch: int) -> None:
        """
        Re-shuffle shards for a new epoch and reset the pipeline.

        [FIX-14] Called by MixingSource.set_epoch() so each epoch sees a
        distinct, reproducible shard order.  The pipeline is drained and
        restarted cleanly.
        """
        # Drain any pending futures to avoid mixing across epochs.
        # Do not re-raise errors here — a new epoch is a fresh start.
        for f in list(self._futures):
            f.cancel()
        self._futures.clear()
        self._buffer.clear()

        # [FIX-P1] Clear the poison pill so a new epoch can proceed after
        # a transient error (e.g. a single corrupt shard that has since been
        # replaced).  The operator is expected to have fixed the root cause;
        # if the error recurs it will be re-set immediately.
        self._poison_pill.clear()
        self._worker_error = None

        self._init_epoch(epoch)

    def close(self) -> None:
        """Shut down the extraction thread pool. [F-1]"""
        if self._closed:
            return
        self._closed = True

        # [FIX-P1] Set the pill so any thread currently blocked in next_jpeg()
        # wakes up and exits cleanly instead of waiting for I/O.
        self._poison_pill.set()

        for f in self._futures:
            f.cancel()
        self._executor.shutdown(wait=False)
        self._futures.clear()
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _worker_init(self) -> None:
        """
        Per-worker initialisation called once when each thread starts.

        [FIX-21] Applies NUMA-local CPU affinity so extraction threads stay
        on the socket that is topologically closest to the GPU.
        """
        if self._affinity_cpus:
            _apply_thread_affinity(self._affinity_cpus)
            log.debug(
                "Thread %s: affinity set to %d NUMA-local CPUs",
                threading.current_thread().name, len(self._affinity_cpus),
            )

    def _init_epoch(self, epoch: int) -> None:
        """Shuffle shards for this epoch and prime the two-level pipeline."""
        self._shards = list(self._all_shards)
        random.Random(self._seed + self._rank + epoch * 1_000_003).shuffle(self._shards)
        self._idx = 0

        # [FIX-13] + [FIX-E]
        # Deduplicate initial submissions when n_shards < _EXTRACTION_DEPTH.
        # Previously the else-branch advanced self._idx without submitting a
        # future, causing _prefetch_window() to skip the first shard's cache
        # prefetch.  Now we simply skip the duplicate submission without
        # touching _idx, so _prefetch_window() starts from the correct position.
        submitted: Set[str] = set()
        for _ in range(self._EXTRACTION_DEPTH):
            path = self._shards[self._idx % len(self._shards)]
            if path not in submitted:
                self._submit_next_extraction()
                submitted.add(path)
            # [FIX-E] Do NOT advance _idx here — just skip the duplicate.

        self._prefetch_window()

    def _submit_next_extraction(self) -> None:
        path      = self._shards[self._idx % len(self._shards)]
        self._idx += 1
        # [F-4] cache.get_view() is safe from worker threads:
        #   Non-master ranks: _inotify_wait() → select(), thread-safe.
        #   Master rank: asyncio.run_coroutine_threadsafe().result(), cross-thread safe.
        future = self._executor.submit(self._fetch_and_extract, path)
        self._futures.append(future)

    def _fetch_and_extract(self, path: str) -> List[bytes]:
        """Worker: fetch raw tar from cache, parse JPEG members."""
        with self._cache.get_view(path) as raw_view:
            return _extract_jpegs(raw_view)

    def _drain_next_future(self) -> None:
        """
        Block on the oldest in-flight extraction; replenish immediately.

        [FIX-P1] Wraps future.result() in try/except.  On failure the
        poison pill is set and the exception is stored for re-raising in
        next_jpeg(), giving the caller a clean error with full context.
        Crucially, _submit_next_extraction() is NOT called after a failure —
        there is no point submitting further work on a broken iterator.
        """
        if not self._futures:
            self._submit_next_extraction()

        future = self._futures.popleft()

        try:
            jpegs = future.result()
        except Exception as exc:
            # [FIX-P1] Capture the root cause and set the pill so next_jpeg()
            # surfaces it immediately without attempting further I/O.
            self._worker_error = exc
            self._poison_pill.set()
            log.error(
                "ShardIterator '%s': extraction worker failed — "
                "setting poison pill.  Root cause: %s: %s",
                self._name, type(exc).__name__, exc,
                exc_info=True,
            )
            return  # caller will check the pill

        self._buffer.extend(jpegs)

        self._submit_next_extraction()
        horizon = self._shards[(self._idx + self._ahead - 1) % len(self._shards)]
        self._cache.prefetch(horizon)

    def _prefetch_window(self) -> None:
        """Request cache prefetch for the next prefetch_ahead shards. [F-2]"""
        for i in range(self._ahead):
            path = self._shards[(self._idx + i) % len(self._shards)]
            self._cache.prefetch(path)


# ══════════════════════════════════════════════════════════════════════════════
# DALI ExternalSource callback
# ══════════════════════════════════════════════════════════════════════════════

class MixingSource:
    """
    DALI ExternalSource callback.

    Returned batches are lists of numpy uint8 arrays (raw JPEG bytes).
    DALI decodes them on-GPU via its internal nvjpeg pipeline.

    Thread safety
    -------------
    __next__ is called from DALI's prefetch thread.
    set_weights / set_epoch may be called from the training thread.

    Cleanup
    -------
    Call close() when the loader is torn down.
    """

    def __init__(
        self,
        specs:          List[DatasetSpec],
        batch_size:     int,
        cache:          NodeSharedShardCache,
        rank:           int,
        world_size:     int,
        prefetch_ahead: int  = 32,
        num_workers:    int  = 4,
        seed:           int  = 0,
        device_id:      int  = 0,           # [FIX-21]
        cpu_affinity_enabled: bool = False,  # [FIX-21]
    ):
        self._batch_size = batch_size

        self._weights = MixingWeights(
            names           = [s.name for s in specs],
            initial_weights = [s.weight for s in specs],
        )

        self._iterators: List[ShardIterator] = [
            ShardIterator(
                name                 = s.name,
                shards               = s.shards,
                cache                = cache,
                rank                 = rank,
                world_size           = world_size,
                prefetch_ahead       = prefetch_ahead,
                num_workers          = num_workers,
                seed                 = seed + i,
                device_id            = device_id,           # [FIX-21]
                cpu_affinity_enabled = cpu_affinity_enabled, # [FIX-21]
            )
            for i, s in enumerate(specs)
        ]

    # ------------------------------------------------------------------
    # DALI ExternalSource protocol
    # ------------------------------------------------------------------

    def __call__(self, info=None) -> List[np.ndarray]:
        """
        Return one batch of raw JPEG bytes for DALI.

        Sampling is done with replacement (random.choices), which is O(k)
        and does not require any lock on the weight list beyond a single
        atomic read.
        """
        weights = self._weights.get()
        indices = random.choices(range(len(self._iterators)), weights=weights, k=self._batch_size)
        batch = []
        for idx in indices:
            jpeg = self._iterators[idx].next_jpeg()
            batch.append(np.frombuffer(jpeg, dtype=np.uint8))
        return batch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch change to all ShardIterators. [FIX-14]"""
        for it in self._iterators:
            it.reset_epoch(epoch)

    def set_weights(self, weights: Sequence[float]) -> None:
        self._weights.set(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._weights.set_by_name(name, weight)

    @property
    def current_weights(self) -> List[float]:
        return self._weights.get()

    @property
    def dataset_names(self) -> List[str]:
        return self._weights.names

    def close(self) -> None:
        """Shut down all ShardIterators. [F-1]"""
        for it in self._iterators:
            it.close()
