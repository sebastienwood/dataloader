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
         concurrent get_view() calls on the same file.  Fixed by deduplicating
         the initial submission: if the path was already submitted, skip.

[FIX-14] No per-epoch reshuffle.
         The previous implementation shuffled shards once at construction and
         never again, so each epoch saw the same shard order.  Proper DINO
         training requires epoch-level reshuffling for convergence.  Fixed by
         storing the base seed and re-shuffling in reset_epoch(epoch), which
         MixingSource.set_epoch() calls on all ShardIterators.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import threading
from collections import deque
from typing import Deque, List, Optional, Sequence

import numpy as np

from dino_loader.config         import DatasetSpec
from dino_loader.shard_cache    import NodeSharedShardCache
from dino_loader.datasets.utils import _extract_jpegs

log = logging.getLogger(__name__)


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
    """

    _EXTRACTION_DEPTH = 2

    def __init__(
        self,
        name:           str,
        shards:         List[str],
        cache:          NodeSharedShardCache,
        rank:           int,
        world_size:     int,
        prefetch_ahead: int  = 32,
        num_workers:    int  = 4,     # [FIX-5] was 2
        seed:           int  = 0,
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

        self._shards: List[str] = []   # populated by reset_epoch
        self._idx:    int = 0
        self._buffer: Deque[bytes]                                  = deque()
        self._futures: Deque[concurrent.futures.Future[List[bytes]]] = deque()

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers        = num_workers,
            thread_name_prefix = f"shard-extract-{name}",
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
        """Return the next JPEG bytes, blocking only if extraction lags."""
        if not self._buffer:
            self._drain_next_future()
        return self._buffer.popleft()

    def reset_epoch(self, epoch: int) -> None:
        """
        Re-shuffle shards for a new epoch and reset the pipeline.

        [FIX-14] Called by MixingSource.set_epoch() so each epoch sees a
        distinct, reproducible shard order.  The pipeline is drained and
        restarted cleanly.
        """
        # Drain any pending futures to avoid mixing across epochs
        for f in list(self._futures):
            f.cancel()
        self._futures.clear()
        self._buffer.clear()

        self._init_epoch(epoch)

    def close(self) -> None:
        """Shut down the extraction thread pool. [F-1]"""
        if self._closed:
            return
        self._closed = True
        for f in self._futures:
            f.cancel()
        self._executor.shutdown(wait=False)
        self._futures.clear()
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _init_epoch(self, epoch: int) -> None:
        """Shuffle shards for this epoch and prime the two-level pipeline."""
        self._shards = list(self._all_shards)
        random.Random(self._seed + self._rank + epoch * 1_000_003).shuffle(self._shards)
        self._idx = 0

        # [FIX-13] Deduplicate initial submissions when n_shards < _EXTRACTION_DEPTH
        submitted: set[str] = set()
        for _ in range(self._EXTRACTION_DEPTH):
            path = self._shards[self._idx % len(self._shards)]
            if path not in submitted:
                self._submit_next_extraction()
                submitted.add(path)
            else:
                # Same shard would be submitted twice (tiny dataset); advance
                # idx without a duplicate future so we don't double-consume.
                self._idx += 1

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
        """Block on the oldest in-flight extraction; replenish immediately."""
        if not self._futures:
            self._submit_next_extraction()

        future = self._futures.popleft()
        jpegs  = future.result()
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
    Call close() when the loader is torn down. DINODataLoader calls this
    on __del__.
    """

    def __init__(
        self,
        specs:          List[DatasetSpec],
        batch_size:     int,
        cache:          NodeSharedShardCache,
        rank:           int,
        world_size:     int,
        prefetch_ahead: int = 32,
        num_workers:    int = 4,    # [FIX-5] was 2; exposed via LoaderConfig
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
    # Epoch control
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """
        Re-shuffle all dataset shard iterators for this epoch. [FIX-14]

        Must be called at the start of each epoch (before iterating).
        DINODataLoader.set_epoch() calls this.
        """
        for it in self._iterators:
            it.reset_epoch(epoch)
        log.debug("MixingSource: shards reshuffled for epoch %d", epoch)

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
    # Cleanup [F-1]
    # ------------------------------------------------------------------

    def close(self) -> None:
        for it in self._iterators:
            it.close()

    def __del__(self):
        self.close()
