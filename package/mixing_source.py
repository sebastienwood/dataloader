"""
dino_loader.io.mixing_source
============================
WebDataset mixing source for DALI ExternalSource.

Separation of concerns vs previous versions
--------------------------------------------
MixingWeights   — owns weight state and thread-safe update logic.
ShardIterator   — owns per-dataset shard cycling and prefetch scheduling.
MixingSource    — composes the two; implements the DALI callback protocol.

This makes it possible to update weights from an external thread without
touching any I/O state, and to unit-test each piece independently.
"""

from __future__ import annotations

import io
import logging
import random
import tarfile
import threading
from typing import Iterator, List, Optional, Sequence

import numpy as np

from dino_loader.config         import DatasetSpec
from dino_loader.io.shard_cache import NodeSharedShardCache

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Weight manager
# ══════════════════════════════════════════════════════════════════════════════

class MixingWeights:
    """
    Thread-safe normalised mixing weights.
    Weights can be updated at any time from any thread.
    """

    def __init__(self, names: List[str], initial_weights: List[float]):
        assert len(names) == len(initial_weights)
        self._names  = names
        self._lock   = threading.RLock()
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
        log.debug("Mixing weights updated: %s",
                  dict(zip(self._names, [f"{v:.3f}" for v in w])))

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
    Cycles over the shards assigned to this rank for one dataset,
    parses tar archives, and yields individual JPEG bytes.

    Prefetch scheduling is handled here: as each shard is consumed,
    the next `prefetch_ahead` shards are queued into the shard cache.
    """

    def __init__(
        self,
        name:           str,
        shards:         List[str],
        cache:          NodeSharedShardCache,
        rank:           int,
        world_size:     int,
        prefetch_ahead: int = 32,
        shuffle:        bool = True,
        seed:           int = 0,
    ):
        self._name   = name
        self._cache  = cache
        self._ahead  = prefetch_ahead

        # Rank-local shard assignment: deterministic, no coordination needed
        self._shards = [s for i, s in enumerate(shards) if i % world_size == rank]
        if not self._shards:
            raise RuntimeError(
                f"Rank {rank}/{world_size}: no shards assigned for dataset '{name}'. "
                f"Dataset has {len(shards)} shards total."
            )

        if shuffle:
            rng = random.Random(seed + rank)
            rng.shuffle(self._shards)

        self._idx:     int = 0
        self._buffer:  List[bytes] = []
        self._buf_pos: int = 0

        # Pre-warm the first window
        self._prefetch_window()

    def next_jpeg(self) -> bytes:
        """Return the next JPEG bytes, loading a new shard when the buffer empties."""
        if self._buf_pos >= len(self._buffer):
            self._advance_shard()
        jpeg = self._buffer[self._buf_pos]
        self._buf_pos += 1
        return jpeg

    # ------------------------------------------------------------------

    def _advance_shard(self) -> None:
        shard_path     = self._shards[self._idx % len(self._shards)]
        self._idx     += 1
        raw            = self._cache.get(shard_path)
        self._buffer   = _extract_jpegs(raw)
        self._buf_pos  = 0
        self._prefetch_window()

    def _prefetch_window(self) -> None:
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
    MixingWeights uses an RLock; ShardIterator is single-producer.
    """

    def __init__(
        self,
        specs:          List[DatasetSpec],
        batch_size:     int,
        cache:          NodeSharedShardCache,
        rank:           int,
        world_size:     int,
        prefetch_ahead: int = 32,
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
        indices = self._rng.choices(range(len(self._iterators)),
                                    weights=weights, k=self._batch_size)
        return [
            np.frombuffer(self._iterators[i].next_jpeg(), dtype=np.uint8)
            for i in indices
        ]


# ══════════════════════════════════════════════════════════════════════════════
# Tar parsing
# ══════════════════════════════════════════════════════════════════════════════

_JPEG_EXTS = frozenset({".jpg", ".jpeg"})


def _extract_jpegs(tar_bytes: bytes) -> List[bytes]:
    """
    Extract all JPEG members from a tar archive held in memory.
    Avoids double-buffering by using a single BytesIO wrapper.
    """
    results: List[bytes] = []
    buf = io.BytesIO(tar_bytes)   # zero-copy view; tarfile reads from it
    try:
        with tarfile.open(fileobj=buf, mode="r|*") as tf:
            for member in tf:
                ext = "." + member.name.rsplit(".", 1)[-1].lower() if "." in member.name else ""
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
