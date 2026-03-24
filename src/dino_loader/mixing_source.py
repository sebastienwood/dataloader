"""dino_loader.mixing_source.

DALI ExternalSource callback and per-dataset shard cycling.

Changes in this version
-----------------------
[PRED-1] **Early sample filtering via SamplePredicate**

    A ``SamplePredicate`` callable is now accepted by ``ShardIterator`` and
    ``MixingSource``.  It is called by the extraction worker on the
    lightweight ``SampleMeta`` (metadata dict + key + shard path) *before*
    the sample JPEG enters the DALI pipeline.

    This eliminates DALI decode cost for samples that would have been
    discarded by a post-pipeline ``select()`` call.  The cost of the
    predicate itself is negligible: a simple dict lookup is ~50 ns vs.
    ~2 ms for DALI JPEG decode.

    Interaction with ``min_sample_quality``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``DatasetSpec.min_sample_quality`` is evaluated at the same stage (before
    DALI) and is preserved as a zero-overhead fast path.  ``SamplePredicate``
    is the extension point for logic that cannot be expressed as a float
    threshold.

    Thread safety
    ~~~~~~~~~~~~~
    The predicate is called from extraction worker threads.  Implementations
    must be read-only (stateless or using only thread-safe reads).  Mutable
    state (e.g. per-class counters for balanced sampling) must be protected
    by a threading.Lock.

[DB-1] Strict double-buffering in ShardIterator (retained).
[DB-2] otel integration (retained).
[MS-Q1..R3, MS-8, MS-9] All previous improvements retained.
"""

import fcntl
import logging
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Deque, Sequence

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
        "to 'epoch' mode.  Install with: pip install webdataset"
    )

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ══════════════════════════════════════════════════════════════════════════════
# MixingWeights — thread-safe weight vector with named access
# ══════════════════════════════════════════════════════════════════════════════

class MixingWeights:
    """Normalised, thread-safe weight vector for dataset mixing."""

    def __init__(self, specs: list[DatasetSpec]) -> None:
        self.names  = [s.name for s in specs]
        raw         = [s.weight for s in specs]
        self._lock  = threading.Lock()
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


# ══════════════════════════════════════════════════════════════════════════════
# ResolutionSource — thread-safe (global_size, local_size) provider
# ══════════════════════════════════════════════════════════════════════════════

class ResolutionSource:
    """Thread-safe holder for the current crop resolution.

    Acts as a DALI ExternalSource callback (batch=False).
    Calling set() is immediately visible to the next DALI prefetch.
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


# ══════════════════════════════════════════════════════════════════════════════
# SampleRecord
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Sentinel for I/O thread shutdown
# ══════════════════════════════════════════════════════════════════════════════

class _Sentinel:
    """Passed through _io_queue to signal I/O thread shutdown."""
    __slots__ = ()


_STOP = _Sentinel()


# ══════════════════════════════════════════════════════════════════════════════
# [DB-1] ShardIterator — per-dataset, per-rank shard cycling
# ══════════════════════════════════════════════════════════════════════════════

class ShardIterator:
    """Per-dataset, per-rank shard cycling with strict double-buffering.

    [PRED-1] Early filtering
    -------------------------
    An optional ``sample_predicate`` is evaluated inside the extraction worker
    on the lightweight ``SampleMeta`` *before* the JPEG bytes are queued for
    DALI.  This is the earliest possible rejection point — no JPEG decode, no
    DALI pipeline invocation, no GPU memory allocation.

    Stage A — I/O thread (single daemon):
        Reads shard bytes from cache into ``_io_queue``.

    Stage B — Extraction workers (ThreadPoolExecutor):
        Parses tar, evaluates predicate, pushes accepted ``SampleRecord``
        objects into ``_sample_queue``.
    """

    _IO_BUFFER: int = 2

    def __init__(
        self,
        spec:                 DatasetSpec,
        cache:                object,
        rank:                 int,
        world_size:           int,
        prefetch_ahead:       int   = 32,
        num_workers:          int   = 4,
        seed:                 int   = 0,
        device_id:            int   = 0,
        cpu_affinity_enabled: bool  = False,
        shuffle_buffer_size:  int   = 512,
        min_sample_quality:   float | None = None,
        sample_predicate:     SamplePredicate | None = None,  # [PRED-1]
    ) -> None:
        self._name             = spec.name
        self._cache            = cache
        self._ahead            = prefetch_ahead
        self._seed             = seed
        self._rank             = rank
        self._rng              = np.random.default_rng(seed + rank)
        self._sampling         = spec.shard_sampling
        self._sample_predicate = sample_predicate  # [PRED-1]

        self._all_shards: list[str] = [
            s for i, s in enumerate(spec.shards) if i % world_size == rank
        ]
        self._shard_weights: list[float] | None = None
        if spec.shard_quality_scores is not None:
            raw   = [spec.shard_quality_scores[i]
                     for i, _ in enumerate(spec.shards)
                     if i % world_size == rank]
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
            min_sample_quality if min_sample_quality is not None
            else spec.min_sample_quality
        )

        self._io_queue:     queue.Queue[object]       = queue.Queue(maxsize=self._IO_BUFFER)
        self._sample_queue: queue.Queue[SampleRecord] = queue.Queue()
        self._closed = False

        self._shard_cycle = self._make_shard_cycle()
        self._io_thread = threading.Thread(
            target = self._io_loop,
            name   = f"shard-io-{self._name}",
            daemon = True,
        )
        self._io_thread.start()

        self._executor = ThreadPoolExecutor(
            max_workers        = max(1, num_workers),
            thread_name_prefix = f"shard-extract-{self._name}",
        )
        self._extract_futures: Deque = deque()

        for _ in range(max(1, num_workers)):
            self._submit_extract()

    # ── Public API ─────────────────────────────────────────────────────────────

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
        """Re-seed the RNG and restart the shard cycle for a new epoch."""
        self._rng         = np.random.default_rng(self._seed + self._rank + epoch * 997)
        self._shard_cycle = self._make_shard_cycle()

        for q in (self._io_queue, self._sample_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        for _ in range(self._executor._max_workers):  # type: ignore[attr-defined]
            self._submit_extract()

    def close(self) -> None:
        """Signal shutdown and cancel in-flight work."""
        self._closed = True
        try:
            self._io_queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def reservoir_size(self) -> int:
        """Approximate number of samples currently buffered (O(1), thread-safe)."""
        return self._sample_queue.qsize()

    # ── [PRED-1] Predicate evaluation ─────────────────────────────────────────

    def _passes_predicate(self, record: SampleRecord, shard_path: str) -> bool:
        """Return True if the sample passes all early filters.

        Evaluated before the sample enters the DALI pipeline.

        Checks (in order, cheapest first):
        1. ``min_sample_quality`` threshold (float comparison, ~10 ns).
        2. User-supplied ``SamplePredicate`` callable (~50 ns–µs).
        """
        meta = record.metadata

        # Fast path: quality threshold (no Python call overhead).
        if self._min_sample_quality is not None and meta is not None:
            score = meta.get("quality_score")
            if score is not None and score < self._min_sample_quality:
                return False

        # User predicate (arbitrary logic).
        if self._sample_predicate is not None:
            sample_meta = SampleMeta(
                key        = record.key,
                shard_path = shard_path,
                metadata   = meta,
            )
            if not self._sample_predicate(sample_meta):
                return False

        return True

    # ── Stage A: I/O loop ──────────────────────────────────────────────────────

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
        try:
            from dino_loader.monitor.otel import StageTimer
            _otel_ok = True
        except ImportError:
            _otel_ok = False

        while not self._closed:
            try:
                shard_path = next(self._shard_cycle)
            except StopIteration:
                break

            try:
                next_path = next(self._shard_cycle)
                self._cache.prefetch(next_path)
                self._shard_cycle = self._chain_path(next_path, self._shard_cycle)
            except StopIteration:
                pass

            if _otel_ok:
                timer = StageTimer("shard_wait")
                timer.start()

            try:
                with self._cache.get_view(shard_path) as mv:
                    data = bytes(mv)
            except Exception as exc:
                log.error(
                    "ShardIterator '%s': I/O read failed for %s: %s",
                    self._name, shard_path, exc,
                )
                if _otel_ok:
                    timer.stop()
                continue

            if _otel_ok:
                timer.stop()

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
    def _chain_path(path: str, gen):
        yield path
        yield from gen

    # ── Stage B: extraction workers ────────────────────────────────────────────

    def _submit_extract(self) -> None:
        if self._closed:
            return
        fut = self._executor.submit(self._extract_worker)
        self._extract_futures.append(fut)

    def _extract_worker(self) -> None:
        """Pop shard from _io_queue, parse, filter, push accepted samples.

        [PRED-1] The predicate is evaluated here, inside the extraction
        worker thread, *before* the JPEG bytes are enqueued.  Rejected
        samples are discarded without ever reaching DALI.
        """
        from dino_loader.datasets.utils import _extract_jpegs_with_meta

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
                    min_quality    = None,   # [PRED-1] handled below via _passes_predicate
                    shuffle_buffer = self._shuffle_buffer_size,
                    rng            = self._rng,
                )
                for record in records:
                    if self._closed:
                        return
                    # [PRED-1] Early filter — before queuing for DALI.
                    if not self._passes_predicate(record, shard_path):
                        continue
                    self._sample_queue.put_nowait(record)
            except Exception as exc:
                log.error(
                    "ShardIterator '%s': extraction failed for %s: %s",
                    self._name, shard_path, exc,
                )

    # ── Shard cycling ──────────────────────────────────────────────────────────

    def _make_shard_cycle(self):
        if self._sampling == "resampled" and self._resampled_iter is not None:
            while not self._closed:
                item = next(self._resampled_iter)
                yield item["url"]
        else:
            while not self._closed:
                shards = list(self._all_shards)
                if self._shard_weights:
                    ordered = self._rng.choice(
                        shards,
                        size    = len(shards),
                        replace = False,
                        p       = self._shard_weights,
                    ).tolist()
                else:
                    self._rng.shuffle(shards)
                    ordered = shards
                yield from ordered


# ══════════════════════════════════════════════════════════════════════════════
# MixingSource — DALI ExternalSource callback
# ══════════════════════════════════════════════════════════════════════════════

class MixingSource:
    """DALI ExternalSource callback.

    Called once per batch; returns a list of np.ndarray (JPEG bytes).

    [PRED-1] The ``sample_predicate`` is forwarded to each ``ShardIterator``
    so that filtering happens at extraction time, not after DALI decode.

    [MS-R3] Tracks per-sample dataset indices and calls registered callbacks.
    [MS-R2] Optionally logs __key__ per sample for debug auditing.
    """

    def __init__(
        self,
        specs:               list[DatasetSpec],
        batch_size:          int,
        cache:               object,
        rank:                int,
        world_size:          int,
        num_workers:         int   = 4,
        seed:                int   = 0,
        device_id:           int   = 0,
        shuffle_buffer_size: int   = 512,
        debug_log_keys:      str | None = None,
        sample_predicate:    SamplePredicate | None = None,  # [PRED-1]
    ) -> None:
        self._batch_size  = batch_size
        self._rng         = np.random.default_rng(seed + rank * 1000 + 7)
        self._debug_log   = debug_log_keys
        self._lock        = threading.Lock()

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
                sample_predicate    = sample_predicate,  # [PRED-1]
            )
            self._iters.append(it)

        self._last_metadata: list[dict | None] = []

        self._debug_log_file = None
        if debug_log_keys:
            try:
                self._debug_log_file = open(debug_log_keys, "a", buffering=1)
                fcntl.flock(self._debug_log_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
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

        for _ in range(self._batch_size):
            ds_idx = int(self._rng.choice(len(self._iters), p=weights))
            rec    = self._iters[ds_idx].next_sample()
            records.append(rec)
            ds_indices.append(ds_idx)

        if self._ds_index_callbacks:
            for cb in self._ds_index_callbacks:
                cb(ds_indices)

        if self._debug_log_file is not None:
            for rec in records:
                try:
                    self._debug_log_file.write(rec.key + "\n")
                except Exception:
                    pass

        with self._lock:
            self._last_metadata = [r.metadata for r in records]

        if HAS_METRICS:
            reg = get_registry()
            if reg is not None:
                total_depth = sum(it.reservoir_size for it in self._iters)
                reg.set(MetricField.MIXING_QUEUE_DEPTH, total_depth)

        return [
            np.frombuffer(rec.jpeg, dtype=np.uint8) for rec in records
        ]

    def pop_last_metadata(self) -> list[dict | None]:
        """Return metadata list from the last ``__call__``.  Thread-safe."""
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
        """Current normalised mixing weights."""
        return self._weights.get()

    @property
    def dataset_names(self) -> list[str]:
        """Names of all datasets in the mix."""
        return list(self.names)

    def close(self) -> None:
        """Shut down all ShardIterators and release resources."""
        for it in self._iters:
            it.close()
        if self._debug_log_file is not None:
            try:
                self._debug_log_file.close()
            except Exception:
                pass
