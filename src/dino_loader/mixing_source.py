"""
dino_loader.mixing_source
=========================
DALI ExternalSource callback and per-dataset shard cycling.

[PRED-1] Early sample filtering via SamplePredicate — evaluated before JPEG
         decode, eliminating DALI decode cost for rejected samples.
[DB-1]   Strict double-buffering in ShardIterator (Stage A: I/O thread,
         Stage B: extraction workers).
[FIX-RNG] numpy.random.Generator is not thread-safe. _rng accesses in
          MixingSource.__call__() are now protected by _rng_lock. ShardIterator
          creates a fresh per-extraction-worker RNG from the seed+rank base to
          avoid sharing state across threads.
[FIX-CYCLE] ShardIterator._shard_cycle reset race: a threading.Event now
            signals the I/O thread to stop before the cycle is replaced in
            reset_epoch(), preventing a mid-yield generator replacement.
[FIX-KEYLOG] Removed fcntl usage for debug key logging. LOCK_EX|LOCK_NB on
             a file opened for append never released the lock, silently
             breaking key logging in multi-rank jobs. Replaced with a
             threading.Lock around writes.
"""

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from dino_loader.augmentation import SampleMeta, SamplePredicate
from dino_datasets import DatasetSpec

log = logging.getLogger(__name__)

try:
    from dino_loader.monitor.metrics import get_registry, MetricField
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

try:
    from webdataset.shardlists import ResampledShards
    HAS_WDS = True
except ImportError:
    HAS_WDS = False
    log.warning(
        "webdataset not installed — shard_sampling='resampled' will fall back "
        "to 'epoch' mode. Install with: pip install webdataset"
    )


class MixingWeights:
    """Normalised, thread-safe weight vector for dataset mixing."""

    def __init__(self, specs: list[DatasetSpec]) -> None:
        self.names    = [s.name for s in specs]
        raw           = [s.weight for s in specs]
        self._lock    = threading.Lock()
        self._weights = self._normalise(raw)

    def get(self) -> list[float]:
        """Return a copy of the current normalised weight vector."""
        with self._lock:
            return list(self._weights)

    def set(self, weights: Sequence[float]) -> None:
        """Replace the weight vector (re-normalised automatically)."""
        if len(weights) != len(self.names):
            msg = (
                f"MixingWeights.set: expected {len(self.names)} weights, "
                f"got {len(weights)}."
            )
            raise ValueError(msg)
        with self._lock:
            self._weights = self._normalise(list(weights))

    def set_by_name(self, name: str, weight: float) -> None:
        """Update one dataset's weight by name (re-normalised automatically)."""
        try:
            idx = self.names.index(name)
        except ValueError:
            msg = f"Dataset '{name}' not found. Available: {self.names}"
            raise KeyError(msg) from None
        with self._lock:
            w     = list(self._weights)
            total = sum(w) or 1.0
            raw   = [v * total for v in w]
            raw[idx] = weight
            self._weights = self._normalise(raw)

    @staticmethod
    def _normalise(weights: list[float]) -> list[float]:
        total = sum(weights)
        if total <= 0:
            msg = f"Weights must sum to a positive number, got {weights}."
            raise ValueError(msg)
        return [w / total for w in weights]


class ResolutionSource:
    """Thread-safe holder for the current crop resolution.

    Acts as a DALI ExternalSource callback (batch=False). Calling set() is
    immediately visible to the next DALI prefetch.
    """

    def __init__(self, global_size: int, local_size: int) -> None:
        self._global = global_size
        self._local  = local_size
        self._lock   = threading.Lock()

    def set(self, global_size: int, local_size: int) -> None:
        """Update both dimensions atomically."""
        with self._lock:
            self._global = global_size
            self._local  = local_size

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current (global_size, local_size) as numpy scalars."""
        with self._lock:
            return (
                np.array(self._global, dtype=np.int32),
                np.array(self._local,  dtype=np.int32),
            )


class SampleRecord:
    """Decoded sample ready for the DALI pipeline."""

    __slots__ = ("jpeg", "metadata", "key")

    def __init__(
        self,
        jpeg:     bytes,
        metadata: dict | None = None,
        key:      str         = "",
    ) -> None:
        self.jpeg     = jpeg
        self.metadata = metadata
        self.key      = key


class _Sentinel:
    """Passed through _io_queue to signal I/O thread shutdown."""
    __slots__ = ()


_STOP = _Sentinel()


class ShardIterator:
    """Per-dataset, per-rank shard cycling with strict double-buffering.

    Stage A — I/O thread (single daemon):
        Reads shard bytes from cache into _io_queue.

    Stage B — Extraction workers (ThreadPoolExecutor):
        Parses tar, evaluates predicate, pushes accepted SampleRecords
        into _sample_queue.

    [FIX-CYCLE] reset_epoch() sets _stop_io_event before replacing the
    generator, then waits for the I/O thread to acknowledge via _io_stopped_event.
    This prevents the I/O thread from being mid-next() when the generator is
    replaced, avoiding orphaned generator state.

    [FIX-RNG] Each extraction worker gets its own seeded RNG derived from the
    base seed, rank, and worker index to avoid sharing mutable Generator state
    across threads.
    """

    _IO_BUFFER: int = 2

    def __init__(
        self,
        spec:                 DatasetSpec,
        cache:                Any,
        rank:                 int,
        world_size:           int,
        prefetch_ahead:       int   = 32,
        num_workers:          int   = 4,
        seed:                 int   = 0,
        device_id:            int   = 0,
        shuffle_buffer_size:  int   = 512,
        min_sample_quality:   float | None = None,
        sample_predicate:     SamplePredicate | None = None,
    ) -> None:
        self._name             = spec.name
        self._cache            = cache
        self._seed             = seed
        self._rank             = rank
        self._sampling         = spec.shard_sampling
        self._sample_predicate = sample_predicate
        self._num_workers      = max(1, num_workers)

        self._all_shards: list[str] = [
            s for i, s in enumerate(spec.shards) if i % world_size == rank
        ]
        self._shard_weights: list[float] | None = None
        if spec.shard_quality_scores is not None:
            raw   = [
                spec.shard_quality_scores[i]
                for i, _ in enumerate(spec.shards)
                if i % world_size == rank
            ]
            total = sum(raw) or 1.0
            self._shard_weights = [w / total for w in raw]

        if not self._all_shards:
            msg = (
                f"Rank {rank}/{world_size}: no shards assigned for dataset "
                f"'{spec.name}' ({len(spec.shards)} shards total)."
            )
            raise RuntimeError(msg)

        if len(self._all_shards) < 4:
            log.warning(
                "ShardIterator '%s': only %d shard(s) assigned to rank %d/%d. "
                "Consider using shard_sampling='resampled' for small datasets.",
                self._name, len(self._all_shards), rank, world_size,
            )

        self._resampled_iter = None
        if self._sampling == "resampled":
            if HAS_WDS:
                self._resampled_iter = iter(
                    ResampledShards(
                        urls          = self._all_shards,
                        seed          = seed + rank,
                        deterministic = True,
                    )
                )
            else:
                log.warning(
                    "ShardIterator '%s': shard_sampling='resampled' requested "
                    "but webdataset is not installed — falling back to 'epoch'.",
                    self._name,
                )
                self._sampling = "epoch"

        self._shuffle_buffer_size = shuffle_buffer_size
        self._min_sample_quality  = (
            min_sample_quality
            if min_sample_quality is not None
            else spec.min_sample_quality
        )

        self._io_queue:     queue.Queue[Any]          = queue.Queue(maxsize=self._IO_BUFFER)
        self._sample_queue: queue.Queue[SampleRecord] = queue.Queue()
        self._closed        = False

        # [FIX-CYCLE] Events to coordinate clean generator replacement.
        self._stop_io_event    = threading.Event()
        self._io_stopped_event = threading.Event()

        # Epoch-specific RNG for shard shuffling — accessed only from the I/O thread.
        self._io_rng = np.random.default_rng(seed + rank)
        self._shard_cycle = self._make_shard_cycle(self._io_rng)

        self._io_thread = threading.Thread(
            target=self._io_loop,
            name=f"shard-io-{self._name}",
            daemon=True,
        )
        self._io_thread.start()

        self._executor = ThreadPoolExecutor(
            max_workers        = self._num_workers,
            thread_name_prefix = f"shard-extract-{self._name}",
        )
        self._extract_futures: deque = deque()

        for _ in range(self._num_workers):
            self._submit_extract()

    def next_sample(self) -> SampleRecord:
        """Block until a sample passes all filters, then return it."""
        while not self._closed:
            try:
                return self._sample_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                continue
        msg = f"ShardIterator '{self._name}' has been closed."
        raise StopIteration(msg)

    def reset_epoch(self, epoch: int) -> None:
        """Restart the shard cycle for a new epoch.

        [FIX-CYCLE] Signals the I/O thread to stop its current cycle, waits
        for acknowledgement, then atomically replaces the generator.
        """
        # Signal the I/O thread to pause.
        self._stop_io_event.set()
        self._io_stopped_event.wait(timeout=5.0)
        self._io_stopped_event.clear()

        # Drain queues.
        for q in (self._io_queue, self._sample_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Replace generator with new epoch seed.
        self._io_rng      = np.random.default_rng(self._seed + self._rank + epoch * 997)
        self._shard_cycle = self._make_shard_cycle(self._io_rng)

        # Resume the I/O thread.
        self._stop_io_event.clear()

        for _ in range(self._num_workers):
            self._submit_extract()

    def close(self) -> None:
        """Signal shutdown and cancel in-flight work."""
        self._closed = True
        self._stop_io_event.set()
        try:
            self._io_queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def reservoir_size(self) -> int:
        """Approximate number of samples currently buffered. Thread-safe, O(1)."""
        return self._sample_queue.qsize()

    def _passes_predicate(self, record: SampleRecord, shard_path: str) -> bool:
        """Return True if the sample passes all early filters.

        Checks (cheapest first):
        1. min_sample_quality threshold (float compare, ~10 ns).
        2. User-supplied SamplePredicate (~50 ns–µs).
        """
        meta = record.metadata

        if self._min_sample_quality is not None and meta is not None:
            score = meta.get("quality_score")
            if score is not None and score < self._min_sample_quality:
                return False

        if self._sample_predicate is not None:
            sample_meta = SampleMeta(
                key        = record.key,
                shard_path = shard_path,
                metadata   = meta,
            )
            if not self._sample_predicate(sample_meta):
                return False

        return True

    # Stage A: I/O loop

    def _io_loop(self) -> None:
        try:
            import contextvars
            ctx = contextvars.copy_context()
            ctx.run(self._io_loop_inner)
        except Exception as exc:
            log.error(
                "ShardIterator '%s': I/O thread crashed: %s",
                self._name, exc, exc_info=True,
            )

    def _io_loop_inner(self) -> None:
        while not self._closed:
            # [FIX-CYCLE] Acknowledge stop request before yielding from generator.
            if self._stop_io_event.is_set():
                self._io_stopped_event.set()
                # Spin-wait until the event is cleared by reset_epoch() or close().
                self._stop_io_event.wait()
                if self._closed:
                    return
                continue

            try:
                shard_path = next(self._shard_cycle)
            except StopIteration:
                break

            # Prefetch the next shard while we queue the current one.
            try:
                next_path = next(self._shard_cycle)
                self._cache.prefetch(next_path)
                self._shard_cycle = self._chain_path(next_path, self._shard_cycle)
            except StopIteration:
                pass

            try:
                with self._cache.get_view(shard_path) as mv:
                    data = bytes(mv)
            except Exception as exc:
                log.error(
                    "ShardIterator '%s': I/O read failed for %s: %s",
                    self._name, shard_path, exc,
                )
                continue

            if self._closed:
                return

            try:
                self._io_queue.put((shard_path, data), block=True, timeout=5.0)
            except queue.Full:
                if self._closed:
                    return
                log.warning(
                    "ShardIterator '%s': _io_queue full after 5 s — "
                    "extraction workers may be stalled.",
                    self._name,
                )
                if not self._closed:
                    self._io_queue.put((shard_path, data), block=True)

        if not self._closed:
            self._io_queue.put(_STOP)

    @staticmethod
    def _chain_path(
        path: str,
        gen: Generator,
    ) -> Generator[str, None, None]:
        yield path
        yield from gen

    # Stage B: extraction workers

    def _submit_extract(self) -> None:
        if self._closed:
            return
        fut = self._executor.submit(self._extract_worker)
        self._extract_futures.append(fut)

    def _extract_worker(self) -> None:
        """Pop shard from _io_queue, parse, filter, push accepted samples.

        [FIX-RNG] Each worker uses an independent RNG seeded from the thread
        identity, avoiding shared mutable state across worker threads.
        """
        from dino_loader.datasets.utils import _extract_jpegs_with_meta

        # Per-worker RNG — no shared state between threads.
        worker_rng = np.random.default_rng(
            self._seed + self._rank + threading.get_ident()
        )

        while not self._closed:
            try:
                item = self._io_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue

            if isinstance(item, _Sentinel) or self._closed:
                try:
                    self._io_queue.put_nowait(_STOP)
                except queue.Full:
                    pass
                return

            shard_path, data = item

            try:
                records = _extract_jpegs_with_meta(
                    memoryview(data),
                    metadata_key   = None,
                    min_quality    = None,  # handled below via _passes_predicate
                    shuffle_buffer = self._shuffle_buffer_size,
                    rng            = worker_rng,
                )
                for record in records:
                    if self._closed:
                        return
                    if not self._passes_predicate(record, shard_path):
                        continue
                    self._sample_queue.put_nowait(record)
            except Exception as exc:
                log.error(
                    "ShardIterator '%s': extraction failed for %s: %s",
                    self._name, shard_path, exc,
                )

    def _make_shard_cycle(
        self, rng: np.random.Generator
    ) -> Generator[str, None, None]:
        """Generate shard paths in shuffled order, cycling on exhaustion."""
        if self._sampling == "resampled" and self._resampled_iter is not None:
            while not self._closed:
                item = next(self._resampled_iter)
                yield item["url"]
        else:
            while not self._closed:
                shards = list(self._all_shards)
                if self._shard_weights:
                    ordered = rng.choice(
                        shards,
                        size    = len(shards),
                        replace = False,
                        p       = self._shard_weights,
                    ).tolist()
                else:
                    rng.shuffle(shards)
                    ordered = shards
                yield from ordered


class MixingSource:
    """DALI ExternalSource callback that mixes samples from multiple datasets.

    Called once per batch; returns a list of np.ndarray (JPEG bytes).

    [FIX-RNG] _rng is protected by _rng_lock. numpy.random.Generator is not
    thread-safe; DALI's prefetch thread and any external reset_epoch() calls
    could race without this lock.

    [FIX-KEYLOG] Debug key logging uses a threading.Lock instead of fcntl,
    which was never released and silently failed in multi-rank jobs.

    [PRED-1] sample_predicate is forwarded to each ShardIterator.
    """

    def __init__(
        self,
        specs:               list[DatasetSpec],
        batch_size:          int,
        cache:               Any,
        rank:                int,
        world_size:          int,
        num_workers:         int   = 4,
        seed:                int   = 0,
        device_id:           int   = 0,
        shuffle_buffer_size: int   = 512,
        debug_log_keys:      str | None = None,
        sample_predicate:    SamplePredicate | None = None,
    ) -> None:
        self._batch_size  = batch_size
        self._debug_log   = debug_log_keys
        self._lock        = threading.Lock()

        # [FIX-RNG] Separate lock for the RNG to avoid conflating it with the
        # metadata lock, minimising contention on the hot path.
        self._rng      = np.random.default_rng(seed + rank * 1000 + 7)
        self._rng_lock = threading.Lock()

        self._ds_index_callbacks: list[Callable[[list[int]], None]] = []

        self._weights = MixingWeights(specs)
        self.names    = self._weights.names

        self._iters: list[ShardIterator] = []
        for spec in specs:
            it = ShardIterator(
                spec                = spec,
                cache               = cache,
                rank                = rank,
                world_size          = world_size,
                num_workers         = num_workers,
                seed                = seed,
                device_id           = device_id,
                shuffle_buffer_size = shuffle_buffer_size,
                sample_predicate    = sample_predicate,
            )
            self._iters.append(it)

        self._last_metadata: list[dict | None] = []

        # [FIX-KEYLOG] Simple threading.Lock replaces broken fcntl approach.
        self._debug_log_file   = None
        self._debug_log_lock   = threading.Lock()
        if debug_log_keys:
            try:
                self._debug_log_file = open(debug_log_keys, "a", buffering=1)  # noqa: SIM115
            except Exception as exc:
                log.warning(
                    "MixingSource: could not open debug_log_keys '%s': %s",
                    debug_log_keys, exc,
                )
                self._debug_log_file = None

    def __call__(self) -> list[np.ndarray]:
        """Return one batch of JPEG byte arrays (called by DALI per step)."""
        weights = self._weights.get()
        ds_indices: list[int] = []
        records:    list[SampleRecord] = []

        # [FIX-RNG] Protect all rng accesses with the dedicated lock.
        with self._rng_lock:
            chosen_indices = self._rng.choice(
                len(self._iters),
                size    = self._batch_size,
                replace = True,
                p       = weights,
            ).tolist()

        for ds_idx in chosen_indices:
            rec = self._iters[ds_idx].next_sample()
            records.append(rec)
            ds_indices.append(ds_idx)

        if self._ds_index_callbacks:
            for cb in self._ds_index_callbacks:
                cb(ds_indices)

        if self._debug_log_file is not None:
            with self._debug_log_lock:
                try:
                    for rec in records:
                        self._debug_log_file.write(rec.key + "\n")
                except Exception:
                    pass

        with self._lock:
            self._last_metadata = [r.metadata for r in records]

        if HAS_METRICS:
            reg = get_registry()
            if reg is not None:
                # MIXING_QUEUE_DEPTH is an absolute gauge (current queue depth),
                # not a cumulative counter. Reset to 0 then add the current value.
                m = reg.metrics
                if m is not None:
                    total_depth = sum(it.reservoir_size for it in self._iters)
                    current     = m.mixing_source_queue_depth
                    if total_depth != current:
                        reg.inc(MetricField.MIXING_QUEUE_DEPTH, total_depth - current)

        return [np.frombuffer(rec.jpeg, dtype=np.uint8) for rec in records]

    def pop_last_metadata(self) -> list[dict | None]:
        """Return metadata from the last __call__. Thread-safe."""
        with self._lock:
            return list(self._last_metadata)

    def register_dataset_index_callback(
        self,
        cb: Callable[[list[int]], None],
    ) -> None:
        """Register a callback that receives per-sample dataset indices."""
        self._ds_index_callbacks.append(cb)

    def set_epoch(self, epoch: int) -> None:
        """Reset all ShardIterators for a new epoch."""
        for it in self._iters:
            it.reset_epoch(epoch)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update mixing weights (re-normalised automatically)."""
        self._weights.set(weights)

    def set_by_name(self, name: str, weight: float) -> None:
        """Update one dataset's weight by name."""
        self._weights.set_by_name(name, weight)

    @property
    def current_weights(self) -> list[float]:
        return self._weights.get()

    @property
    def dataset_names(self) -> list[str]:
        return list(self.names)

    def close(self) -> None:
        """Shut down all ShardIterators and release resources."""
        for it in self._iters:
            it.close()
        if self._debug_log_file is not None:
            with self._debug_log_lock:
                try:
                    self._debug_log_file.close()
                except Exception:
                    pass
