"""
dino_loader.io.mixing_source
============================
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
        The previous version blocked the DALI prefetch thread on both
        cache.get() (Lustre I/O) and _extract_jpegs() (CPU tarfile parse)
        sequentially when the buffer emptied.  The intern's insight is
        correct: tar extraction is CPU-bound and can be overlapped with
        JPEG consumption using a ThreadPoolExecutor.  This eliminates the
        extraction stall from the DALI hot path.

  [A-2] deque[bytes] buffer with popleft() replaces List[bytes] + int index.
        No index bookkeeping; consumed entries are immediately eligible for
        GC rather than holding references until the whole shard is replaced.

FIXED
  [F-1] Executor leak: the intern's ShardIterator had no shutdown path.
        With n_datasets * num_workers threads per rank and no cleanup,
        threads accumulate across epochs and hold references that prevent GC.
        Fixed by adding close() on ShardIterator, called by MixingSource.close(),
        which DINODataLoader calls on __del__ / explicit cleanup.

  [F-2] Off-by-two in initial prefetch horizon.
        The intern's __init__ called cache.prefetch for shards [0, ahead) then
        called _queue_next_extraction() twice, advancing self._idx to 2.
        The prefetch horizon for shards submitted to the executor was then
        ahead of the cache prefetch window — shards 0 and 1 were being
        extracted while shards [0, ahead) were being fetched, but the
        horizon shard requested from the executor was (2 + ahead), which
        had not been prefetched.
        Fixed by computing the cache prefetch window from self._idx after
        extraction futures are queued, not before.

  [F-3] num_workers surfaced in LoaderConfig.
        The intern's num_workers=2 was a magic constant with no path to
        LoaderConfig.  Threaded through MixingSource → ShardIterator via
        LoaderConfig.shard_extraction_workers.

  [F-4] Thread-safety of cache.get() from worker threads: documented.
        cache.get() on non-master ranks calls _inotify_wait() → select(),
        which is safe from any thread.  On the master rank it calls
        asyncio.run_coroutine_threadsafe().result(), also safe cross-thread.
        Added an explicit comment so the next reader does not have to
        re-derive this.
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import random
import tarfile
import threading
from collections import deque
from typing import Deque, List, Optional, Sequence

import numpy as np

from dino_loader.config         import DatasetSpec
from dino_loader.io.shard_cache import NodeSharedShardCache

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Weight manager  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class MixingWeights:
    """
    Thread-safe normalised mixing weights.
    Weights can be updated at any time from any thread.
    """

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
            raise ValueError(
                f"Expected {len(self._names)} weights, got {len(weights)}"
            )
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
# Per-dataset shard iterator  (two-level prefetch pipeline)
# ══════════════════════════════════════════════════════════════════════════════

class ShardIterator:
    """
    Cycles over the shards assigned to this rank for one dataset.

    Two-level prefetch pipeline
    ---------------------------
    Level 1 — I/O (NodeSharedShardCache):
        Raw tar bytes are fetched from Lustre into /dev/shm asynchronously
        by the node master.  This level is driven by cache.prefetch() calls
        issued here, keeping a window of `prefetch_ahead` shards warm.

    Level 2 — CPU extraction (ThreadPoolExecutor):
        While DALI is consuming JPEGs from the current shard buffer, worker
        threads are simultaneously parsing the next tar archive(s) into
        lists of JPEG bytes in RAM.  The DALI prefetch thread only blocks
        when the extraction future has not yet completed — which should be
        rare once both shards in the extraction queue are warm.

    Buffer lifecycle
    ----------------
    self._buffer    : deque[bytes] — JPEGs ready for immediate consumption.
    self._futures   : deque[Future[List[bytes]]] — in-flight extractions.

    next_jpeg() drains self._buffer.  When empty it calls
    _drain_next_future(), which blocks on the oldest future, extends the
    buffer, and immediately queues the next extraction so the pipeline
    stays full.

    Ordering
    --------
    Futures are appended and consumed in FIFO order (deque), so shard
    order is deterministic.  A later shard that happens to complete
    extraction first (e.g., cache hit) waits in the Future until it is
    at the head of the queue — this preserves reproducibility at the
    cost of occasionally waiting for a slower shard.  With num_workers=2
    the maximum wait is one shard's extraction time, which is 50-200 ms.
    """

    # How many extraction futures to keep in flight simultaneously.
    # Two is sufficient: one being consumed, one being extracted.
    _EXTRACTION_DEPTH = 2

    def __init__(
        self,
        name:           str,
        shards:         List[str],
        cache:          NodeSharedShardCache,
        rank:           int,
        world_size:     int,
        prefetch_ahead: int  = 32,
        num_workers:    int  = 2,    # tar extraction threads [F-3]
        shuffle:        bool = True,
        seed:           int  = 0,
    ):
        self._name   = name
        self._cache  = cache
        self._ahead  = prefetch_ahead

        self._shards = [s for i, s in enumerate(shards) if i % world_size == rank]
        if not self._shards:
            raise RuntimeError(
                f"Rank {rank}/{world_size}: no shards assigned for dataset '{name}'. "
                f"Dataset has {len(shards)} shards total."
            )
        if shuffle:
            random.Random(seed + rank).shuffle(self._shards)

        # _idx tracks the next shard to submit for extraction (not the one
        # currently being consumed — that is implicit in the future queue).
        self._idx:     int                                          = 0
        self._buffer:  Deque[bytes]                                 = deque()   # [A-2]
        self._futures: Deque[concurrent.futures.Future[List[bytes]]] = deque()

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers = num_workers,
            thread_name_prefix = f"shard-extract-{name}",
        )
        self._closed = False

        # Prime the pipeline:
        #   1. Queue the initial extraction futures (this advances self._idx).
        #   2. Prefetch the I/O window starting from the current self._idx so
        #      the horizon is correctly aligned. [F-2]
        for _ in range(self._EXTRACTION_DEPTH):
            self._submit_next_extraction()
        self._prefetch_window()

        log.debug(
            "ShardIterator '%s': %d shards, %d workers, %d prefetch window",
            name, len(self._shards), num_workers, prefetch_ahead,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_jpeg(self) -> bytes:
        """
        Return the next JPEG bytes.
        Blocks only if the background extractor has not yet finished the
        next shard — which should be rare once the pipeline is warm.
        """
        if not self._buffer:
            self._drain_next_future()
        return self._buffer.popleft()

    def close(self) -> None:
        """
        Shut down the extraction thread pool. [F-1]
        Must be called when the iterator is no longer needed to avoid
        thread leaks across epochs / dataset reloads.
        """
        if self._closed:
            return
        self._closed = True
        # Cancel pending futures where possible, then shut down without
        # waiting for in-progress work to complete (non-blocking teardown).
        for f in self._futures:
            f.cancel()
        self._executor.shutdown(wait=False)
        self._futures.clear()
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _submit_next_extraction(self) -> None:
        """
        Submit the next shard for tar extraction in the thread pool.
        Advances self._idx and requests a new cache prefetch at the horizon.
        """
        path      = self._shards[self._idx % len(self._shards)]
        self._idx += 1

        # [F-4] cache.get() is safe to call from a worker thread:
        #   - Non-master ranks: _inotify_wait() → select(), thread-safe.
        #   - Master rank: asyncio.run_coroutine_threadsafe().result(),
        #     explicitly designed for cross-thread use.
        future = self._executor.submit(self._fetch_and_extract, path)
        self._futures.append(future)

    def _fetch_and_extract(self, path: str) -> List[bytes]:
        """Worker function: fetch raw tar from cache, parse JPEG members."""
        raw = self._cache.get(path)
        return _extract_jpegs(raw)

    def _drain_next_future(self) -> None:
        """
        Block until the oldest in-flight extraction completes, extend the
        buffer with its results, then immediately submit the next extraction
        to keep the pipeline full.
        """
        if not self._futures:
            # Should not happen in normal operation, but handle gracefully
            self._submit_next_extraction()

        future = self._futures.popleft()
        jpegs  = future.result()   # blocks here if extraction not yet done
        self._buffer.extend(jpegs)

        # Replenish: submit the next extraction and advance the I/O horizon
        self._submit_next_extraction()
        # Prefetch the shard that just moved into the horizon
        horizon = self._shards[(self._idx + self._ahead - 1) % len(self._shards)]
        self._cache.prefetch(horizon)

    def _prefetch_window(self) -> None:
        """
        Request cache prefetch for the next `prefetch_ahead` shards starting
        from the current self._idx (i.e., ahead of the extraction queue).
        Called once after __init__ primes self._idx. [F-2]
        """
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
    set_weights / set_weight_by_name may be called from the training thread.
    MixingWeights uses an RLock; ShardIterator is single-producer per dataset.

    Cleanup
    -------
    Call close() when the loader is torn down to shut down ShardIterator
    thread pools.  DINODataLoader calls this on __del__.
    """

    def __init__(
        self,
        specs:          List[DatasetSpec],
        batch_size:     int,
        cache:          NodeSharedShardCache,
        rank:           int,
        world_size:     int,
        prefetch_ahead: int = 32,
        num_workers:    int = 2,   # extraction workers per dataset [F-3]
        seed:           int = 0,
    ):
        self._batch_size = batch_size
        self._rng        = random.Random(seed + rank)

        self._weights = MixingWeights(
            names           = [s.name for s in specs],
            initial_weights = [s.weight for s in specs],
        )
        self._iterators = [
            ShardIterator(
                name           = s.name,
                shards         = s.shards,
                cache          = cache,
                rank           = rank,
                world_size     = world_size,
                prefetch_ahead = prefetch_ahead,
                num_workers    = num_workers,
                seed           = seed,
            )
            for s in specs
        ]

    # ------------------------------------------------------------------
    # Mixing weight control (thread-safe)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # DALI callback
    # ------------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> List[np.ndarray]:
        weights = self._weights.get()
        indices = self._rng.choices(
            range(len(self._iterators)), weights=weights, k=self._batch_size
        )
        return [
            np.frombuffer(self._iterators[i].next_jpeg(), dtype=np.uint8)
            for i in indices
        ]

    # ------------------------------------------------------------------
    # Cleanup  [F-1]
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down all ShardIterator thread pools."""
        for it in self._iterators:
            it.close()

    def __del__(self):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# Tar parsing
# ══════════════════════════════════════════════════════════════════════════════

_JPEG_EXTS = frozenset({".jpg", ".jpeg"})


def _extract_jpegs(tar_bytes: bytes) -> List[bytes]:
    """
    Extract all JPEG members from a tar archive held in memory.
    Uses a single BytesIO wrapper to avoid double-buffering.
    Called from ShardIterator worker threads — stateless and thread-safe.
    """
    results: List[bytes] = []
    buf = io.BytesIO(tar_bytes)
    try:
        with tarfile.open(fileobj=buf, mode="r|*") as tf:
            for member in tf:
                ext = (
                    "." + member.name.rsplit(".", 1)[-1].lower()
                    if "." in member.name
                    else ""
                )
                if ext not in _JPEG_EXTS:
                    continue
                f = tf.extractfile(member)
                if f is not None:
                    results.append(f.read())
    except tarfile.TarError as e:
        raise RuntimeError(f"Corrupt tar shard: {e}") from e

    if not results:
        raise RuntimeError(
            "Shard contained no JPEG files. "
            "Check that shards are WebDataset tars with .jpg/.jpeg members."
        )
    return results