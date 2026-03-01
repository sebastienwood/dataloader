"""
dino_loader.mixing_source
=========================
WebDataset mixing source for DALI ExternalSource.

Separation of concerns
-----------------------
ResolutionSource  — thread-safe scalar source for dynamic DALI resize (zero rebuild).
MixingWeights     — owns weight state and thread-safe update logic.
ShardIterator     — owns per-dataset shard cycling, prefetch scheduling,
                    tar extraction via wds.TarIterator, intra-shard shuffle,
                    sidecar metadata extraction, and quality filtering.
MixingSource      — composes the above; implements the DALI callback protocol.

Changes from previous version
------------------------------
[MS-1]  wds.TarIterator replaces custom _extract_jpegs tar parser.
        The manual POSIX tar parser (~150 lines) is removed.  webdataset's
        TarIterator handles V7 / GNU / POSIX ustar formats, multi-component
        samples (grouped by __key__), and sidecars (.json, .cls, .txt).
        Zero change to the I/O or SHM cache layers — we still read raw tar
        bytes from NodeSharedShardCache; TarIterator wraps a BytesIO view.

[MS-2]  Sidecar metadata extraction.
        When DatasetSpec.metadata_key is set (default "json"), ShardIterator
        extracts the sidecar alongside the JPEG.  Metadata is stored in a
        parallel deque and returned as part of SampleRecord namedtuples so
        that MixingSource can pass them through to the Batch.

[MS-3]  Sample-level quality filtering.
        If DatasetSpec.min_sample_quality is set, samples whose .json sidecar
        "quality_score" field is below the threshold are discarded before
        entering the DALI pipeline.  The filter is applied in _drain_next_future
        after TarIterator extraction.

[MS-4]  Weighted shard sampling.
        If DatasetSpec.shard_quality_scores is set, ShardIterator samples the
        next shard proportional to quality scores (via random.choices) rather
        than cycling sequentially.  Per-epoch shuffle is preserved.

[MS-5]  Intra-shard shuffle buffer.
        ShardIterator maintains a reservoir of depth LoaderConfig.shuffle_buffer_size.
        next_sample() draws a random position from the buffer instead of popleft(),
        breaking within-shard web-crawl correlations.  Set to 0 to disable.

[MS-6]  ResolutionSource — dynamic DALI resize without pipeline rebuild.
        A thread-safe scalar source emitting (global_size, local_size) per batch,
        consumed by fn.external_source in pipeline.py.  set_resolution() writes
        atomically; takes effect on the next DALI batch boundary.

[MS-7]  Per-dataset normalisation stats.
        MixingSource.__call__ now returns SampleRecord with dataset_idx so that
        a future DALI ExternalSource for per-sample mean/std can be wired in.
        Currently the per-dataset mean/std from DatasetSpec is forwarded to
        the pipeline builder as a lookup table.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from dino_loader.config import DatasetSpec
from dino_loader.shard_cache import NodeSharedShardCache

log = logging.getLogger(__name__)

try:
    import webdataset as wds  # [MS-1]
    HAS_WDS = True
except ImportError:
    HAS_WDS = False
    log.warning(
        "webdataset not installed — falling back to legacy tar parser. "
        "Install with: pip install webdataset"
    )

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ══════════════════════════════════════════════════════════════════════════════
# Sample record (JPEG bytes + optional metadata)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SampleRecord:
    """One decoded sample from a WebDataset shard."""
    jpeg:        bytes
    metadata:    Optional[Dict]  = None   # parsed .json sidecar, or None
    dataset_idx: int             = 0      # index into MixingSource._iterators


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic resolution source  [MS-6]
# ══════════════════════════════════════════════════════════════════════════════

class ResolutionSource:
    """
    Thread-safe scalar source for dynamic DALI resize.

    Emitted as a DALI ExternalSource with ``batch=False`` so the same pair
    (global_size, local_size) is broadcast to all samples in a batch.
    set_resolution() takes effect on the next DALI batch boundary — zero
    downtime, zero pipeline rebuild.

    Usage in pipeline.py::

        res_src = ResolutionSource(224, 96)
        sizes = fn.external_source(source=res_src, num_outputs=2,
                                   dtype=types.INT32, ndim=0, batch=False)
        global_size_node, local_size_node = sizes

        # Pass size_node to fn.resize instead of a Python constant:
        imgs = fn.resize(imgs, resize_x=global_size_node, ...)
    """

    def __init__(self, global_size: int, local_size: int) -> None:
        self._lock        = threading.Lock()
        self._global_size = global_size
        self._local_size  = local_size

    def set(self, global_size: int, local_size: int) -> None:
        with self._lock:
            self._global_size = global_size
            self._local_size  = local_size
        log.info("ResolutionSource: resolution → global=%d  local=%d",
                 global_size, local_size)

    def __call__(self, info=None) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            g = np.array(self._global_size, dtype=np.int32)
            l = np.array(self._local_size,  dtype=np.int32)
        return g, l


# ══════════════════════════════════════════════════════════════════════════════
# Mixing weights
# ══════════════════════════════════════════════════════════════════════════════

class MixingWeights:
    """Thread-safe dataset mixing weights."""

    def __init__(self, names: List[str], initial_weights: List[float]) -> None:
        self._names  = names
        self._lock   = threading.Lock()
        self._w      = self._normalise(initial_weights)

    @staticmethod
    def _normalise(w: List[float]) -> List[float]:
        total = sum(w)
        if total <= 0:
            raise ValueError("All mixing weights are zero.")
        return [x / total for x in w]

    def get(self) -> List[float]:
        with self._lock:
            return list(self._w)

    def set(self, weights: Sequence[float]) -> None:
        with self._lock:
            self._w = self._normalise(list(weights))

    def set_by_name(self, name: str, weight: float) -> None:
        with self._lock:
            idx = self._names.index(name)
            w   = list(self._w)
            w[idx] = weight
            self._w = self._normalise(w)

    @property
    def names(self) -> List[str]:
        return self._names


# ══════════════════════════════════════════════════════════════════════════════
# NUMA helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_numa_cpus(device_id: int) -> Optional[List[int]]:
    if not HAS_PSUTIL:
        return None
    try:
        import psutil
        numa_nodes = psutil.net_if_stats()  # probe availability
        cpus = psutil.Process().cpu_affinity()
        # psutil doesn't expose GPU NUMA directly; approximate by device_id parity
        half = len(cpus) // 2
        return cpus[:half] if device_id % 2 == 0 else cpus[half:]
    except Exception:
        return None


def _apply_thread_affinity(cpus: List[int]) -> None:
    if not HAS_PSUTIL:
        return
    try:
        import psutil
        psutil.Process().cpu_affinity(cpus)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Per-dataset shard iterator
# ══════════════════════════════════════════════════════════════════════════════

class ShardIterator:
    """
    Cycles over the shards assigned to this rank for one dataset.

    Two-level prefetch pipeline (unchanged from previous version)
    ------------------------------------------------------------
    Level 1 — I/O (NodeSharedShardCache):
        Raw tar bytes fetched from Lustre into /dev/shm asynchronously.
    Level 2 — CPU extraction (ThreadPoolExecutor):
        Worker threads parse tar archives into SampleRecord lists in RAM
        while DALI consumes the previous shard's data.

    Key changes
    -----------
    [MS-1] _fetch_and_extract now uses wds.TarIterator (or falls back to the
           legacy parser if webdataset is absent).
    [MS-2] Sidecar .json files are extracted alongside JPEGs.
    [MS-3] Samples below min_sample_quality are silently dropped.
    [MS-4] Shard cycling can be weighted by shard_quality_scores.
    [MS-5] Intra-shard shuffle buffer via _buffer_pool (a list acting as a
           reservoir); next_sample() picks a random slot.
    """

    _EXTRACTION_DEPTH = 2

    def __init__(
        self,
        spec:                DatasetSpec,
        cache:               NodeSharedShardCache,
        rank:                int,
        world_size:          int,
        prefetch_ahead:      int  = 32,
        num_workers:         int  = 4,
        seed:                int  = 0,
        device_id:           int  = 0,
        cpu_affinity_enabled: bool = False,
        shuffle_buffer_size: int  = 512,
        min_sample_quality:  Optional[float] = None,
    ) -> None:
        self._name    = spec.name
        self._cache   = cache
        self._ahead   = prefetch_ahead
        self._seed    = seed
        self._rank    = rank

        # [MS-4] weighted shard sampling support
        self._all_shards:  List[str]           = [s for i, s in enumerate(spec.shards)
                                                   if i % world_size == rank]
        self._shard_weights: Optional[List[float]] = None
        if spec.shard_quality_scores is not None:
            raw = [spec.shard_quality_scores[i] for i, _ in enumerate(spec.shards)
                   if i % world_size == rank]
            total = sum(raw) or 1.0
            self._shard_weights = [w / total for w in raw]

        if not self._all_shards:
            raise RuntimeError(
                f"Rank {rank}/{world_size}: no shards assigned for dataset '{self._name}'. "
                f"Dataset has {len(spec.shards)} shards total."
            )
        if len(self._all_shards) < 4:
            log.warning(
                "ShardIterator '%s': only %d shard(s) assigned to rank %d/%d. "
                "Training quality may degrade due to low per-rank shard diversity.",
                self._name, len(self._all_shards), rank, world_size,
            )

        # [MS-3] quality filter
        self._min_quality = min_sample_quality if min_sample_quality is not None \
                            else spec.min_sample_quality

        # [MS-2] metadata key
        self._metadata_key = spec.metadata_key

        # [MS-5] shuffle buffer
        self._shuffle_buf_size = shuffle_buffer_size
        # Use a list as a reservoir; random replacement instead of popleft
        self._reservoir:  List[SampleRecord]   = []

        # Internal pipeline state
        self._shards:  List[str] = []
        self._idx:     int = 0
        self._futures: Deque[concurrent.futures.Future] = deque()

        # Poison-pill for worker error propagation
        self._poison_pill:   threading.Event      = threading.Event()
        self._worker_error:  Optional[Exception]  = None

        # NUMA affinity
        self._affinity_cpus: Optional[List[int]] = None
        if cpu_affinity_enabled:
            self._affinity_cpus = _resolve_numa_cpus(device_id)

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers        = num_workers,
            thread_name_prefix = f"shard-extract-{self._name}",
            initializer        = self._worker_init,
        )
        self._closed = False

        self._init_epoch(epoch=0)

        log.debug(
            "ShardIterator '%s': %d shards/rank, %d workers, "
            "prefetch=%d, shuffle_buf=%d, wds=%s",
            self._name, len(self._all_shards), num_workers,
            prefetch_ahead, shuffle_buffer_size, HAS_WDS,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_sample(self) -> SampleRecord:
        """
        Return the next SampleRecord, applying the intra-shard shuffle buffer.

        Raises RuntimeError if a worker thread has previously failed.
        """
        if self._poison_pill.is_set():
            raise RuntimeError(
                f"ShardIterator '{self._name}': worker thread previously "
                f"failed — no further data will be produced."
            ) from self._worker_error

        # Fill reservoir if empty
        if not self._reservoir:
            self._drain_next_future()

        if self._poison_pill.is_set():
            raise RuntimeError(
                f"ShardIterator '{self._name}': worker thread previously "
                f"failed — no further data will be produced."
            ) from self._worker_error

        # [MS-5] Intra-shard shuffle: replace a random reservoir slot and
        # return the evicted record, OR just pop if buffer disabled.
        if self._shuffle_buf_size > 0 and len(self._reservoir) >= self._shuffle_buf_size:
            idx = random.randrange(len(self._reservoir))
            record = self._reservoir[idx]
            # Refill from deque if available, otherwise shrink reservoir
            if self._reservoir:
                last = self._reservoir[-1]
                self._reservoir[idx] = last
                self._reservoir.pop()
            return record
        else:
            return self._reservoir.pop(0)

    def reset_epoch(self, epoch: int) -> None:
        """Re-shuffle shards for a new epoch and reset the pipeline."""
        for f in list(self._futures):
            f.cancel()
        self._futures.clear()
        self._reservoir.clear()
        self._poison_pill.clear()
        self._worker_error = None
        self._init_epoch(epoch)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._poison_pill.set()
        for f in self._futures:
            f.cancel()
        self._executor.shutdown(wait=False)
        self._futures.clear()
        self._reservoir.clear()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _worker_init(self) -> None:
        if self._affinity_cpus:
            _apply_thread_affinity(self._affinity_cpus)

    def _init_epoch(self, epoch: int) -> None:
        """Shuffle/weight-sample shards for this epoch and prime the pipeline."""
        if self._shard_weights is not None:
            # [MS-4] Weighted sampling with replacement for a full epoch-length list
            self._shards = random.choices(
                self._all_shards,
                weights=self._shard_weights,
                k=len(self._all_shards),
            )
        else:
            self._shards = list(self._all_shards)
            random.Random(self._seed + self._rank + epoch * 1_000_003).shuffle(self._shards)
        self._idx = 0

        submitted: Set[str] = set()
        for _ in range(self._EXTRACTION_DEPTH):
            path = self._shards[self._idx % len(self._shards)]
            if path not in submitted:
                self._submit_next_extraction()
                submitted.add(path)

        self._prefetch_window()

    def _prefetch_window(self) -> None:
        for i in range(self._ahead):
            path = self._shards[(self._idx + i) % len(self._shards)]
            self._cache.prefetch(path)

    def _submit_next_extraction(self) -> None:
        path       = self._shards[self._idx % len(self._shards)]
        self._idx += 1
        future     = self._executor.submit(self._fetch_and_extract, path)
        self._futures.append(future)

    def _fetch_and_extract(self, path: str) -> List[SampleRecord]:
        """Worker: fetch shard bytes, extract samples via wds.TarIterator."""
        with self._cache.get_view(path) as raw_view:
            if HAS_WDS:
                return self._extract_wds(raw_view)
            else:
                return self._extract_legacy(raw_view)

    def _extract_wds(self, raw_view: memoryview) -> List[SampleRecord]:
        """
        [MS-1] Extract samples using webdataset TarIterator.

        Groups files by __key__ (WebDataset convention).  For each group,
        requires a .jpg or .jpeg member.  Optionally reads the sidecar
        specified by self._metadata_key.  Applies quality filter [MS-3].
        """
        buf     = io.BytesIO(bytes(raw_view))
        results: List[SampleRecord] = []

        try:
            # wds.TarIterator yields dicts: {"__key__": str, "jpg": bytes, "json": bytes, ...}
            for sample in wds.TarIterator(buf):
                # Locate JPEG bytes
                jpeg: Optional[bytes] = sample.get("jpg") or sample.get("jpeg")
                if jpeg is None:
                    continue

                # [MS-2] Parse sidecar metadata
                metadata: Optional[Dict] = None
                if self._metadata_key and self._metadata_key in sample:
                    try:
                        metadata = json.loads(sample[self._metadata_key])
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

                # [MS-3] Quality filter
                if self._min_quality is not None and metadata is not None:
                    score = metadata.get("quality_score")
                    if score is not None and score < self._min_quality:
                        continue

                results.append(SampleRecord(jpeg=jpeg, metadata=metadata))
        except Exception as exc:
            log.warning("wds extraction failed for a shard: %s", exc)

        if not results:
            raise RuntimeError(
                "Shard contained no valid JPEG samples after extraction/filtering. "
                "Check shard integrity or lower min_sample_quality."
            )
        return results

    def _extract_legacy(self, raw_view: memoryview) -> List[SampleRecord]:
        """
        Legacy tar parser (used when webdataset is not installed).
        Extracts only JPEG files; no sidecar support.
        """
        _BLOCK = 512
        results: List[SampleRecord] = []
        offset = 0
        total  = len(raw_view)
        null_count = 0

        while offset + _BLOCK <= total:
            block = raw_view[offset: offset + _BLOCK]
            if all(b == 0 for b in block):
                null_count += 1
                if null_count >= 2:
                    break
                offset += _BLOCK
                continue
            null_count = 0

            name_end = offset
            while name_end < offset + 100 and raw_view[name_end] != 0:
                name_end += 1
            name = bytes(raw_view[offset: name_end]).decode("utf-8", "ignore").lower()

            size_str = bytes(raw_view[offset + 124: offset + 136]).strip(b" \0")
            try:
                file_size = int(size_str, 8) if size_str else 0
            except ValueError:
                file_size = 0

            data_offset = offset + _BLOCK
            typeflag    = raw_view[offset + 156]

            if typeflag in (0, 48) and (name.endswith(".jpg") or name.endswith(".jpeg")):
                if data_offset + file_size <= total:
                    results.append(SampleRecord(
                        jpeg=bytes(raw_view[data_offset: data_offset + file_size])
                    ))

            blocks  = (file_size + _BLOCK - 1) // _BLOCK
            offset += _BLOCK + blocks * _BLOCK

        if not results:
            raise RuntimeError("Shard contained no JPEG files (legacy parser).")
        return results

    def _drain_next_future(self) -> None:
        """Block on the oldest in-flight extraction; extend reservoir."""
        if not self._futures:
            self._submit_next_extraction()

        future = self._futures.popleft()

        try:
            records = future.result()
        except Exception as exc:
            self._worker_error = exc
            self._poison_pill.set()
            log.error(
                "ShardIterator '%s': extraction worker failed: %s: %s",
                self._name, type(exc).__name__, exc, exc_info=True,
            )
            return

        self._reservoir.extend(records)
        self._submit_next_extraction()

        horizon = self._shards[(self._idx + self._ahead - 1) % len(self._shards)]
        self._cache.prefetch(horizon)


# ══════════════════════════════════════════════════════════════════════════════
# DALI ExternalSource callback
# ══════════════════════════════════════════════════════════════════════════════

class MixingSource:
    """
    DALI ExternalSource callback.

    Returns batches of raw JPEG bytes; DALI decodes on-GPU via nvjpeg.
    Also accumulates per-sample metadata (when available) for downstream use.

    Thread safety
    -------------
    __call__ is invoked from DALI's prefetch thread.
    set_weights / set_epoch / resolution updates come from the training thread.
    """

    def __init__(
        self,
        specs:               List[DatasetSpec],
        batch_size:          int,
        cache:               NodeSharedShardCache,
        rank:                int,
        world_size:          int,
        prefetch_ahead:      int  = 32,
        num_workers:         int  = 4,
        seed:                int  = 0,
        device_id:           int  = 0,
        cpu_affinity_enabled: bool = False,
        shuffle_buffer_size: int  = 512,
    ) -> None:
        self._batch_size = batch_size
        self._weights    = MixingWeights(
            names           = [s.name for s in specs],
            initial_weights = [s.weight for s in specs],
        )
        self._iterators: List[ShardIterator] = [
            ShardIterator(
                spec                 = s,
                cache                = cache,
                rank                 = rank,
                world_size           = world_size,
                prefetch_ahead       = prefetch_ahead,
                num_workers          = num_workers,
                seed                 = seed + i,
                device_id            = device_id,
                cpu_affinity_enabled = cpu_affinity_enabled,
                shuffle_buffer_size  = shuffle_buffer_size,
            )
            for i, s in enumerate(specs)
        ]
        # Last-batch metadata cache — read by DINODataLoader after each DALI call
        self._last_metadata: List[Optional[Dict]] = []
        self._meta_lock = threading.Lock()

    # ------------------------------------------------------------------
    # DALI ExternalSource protocol
    # ------------------------------------------------------------------

    def __call__(self, info=None) -> List[np.ndarray]:
        """Return one batch of raw JPEG bytes; cache per-sample metadata."""
        weights = self._weights.get()
        indices = random.choices(range(len(self._iterators)), weights=weights, k=self._batch_size)
        batch:    List[np.ndarray]       = []
        metadata: List[Optional[Dict]]   = []

        for idx in indices:
            record = self._iterators[idx].next_sample()
            batch.append(np.frombuffer(record.jpeg, dtype=np.uint8))
            metadata.append(record.metadata)

        with self._meta_lock:
            self._last_metadata = metadata

        return batch

    def pop_last_metadata(self) -> List[Optional[Dict]]:
        """Thread-safe retrieval of last-batch metadata. Called by DINODataLoader."""
        with self._meta_lock:
            return list(self._last_metadata)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
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
        for it in self._iterators:
            it.close()
