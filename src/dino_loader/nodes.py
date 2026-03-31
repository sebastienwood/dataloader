"""dino_loader.nodes
===================
`torchdata.nodes`-compatible wrappers around the dino_loader pipeline stages.

Motivation
----------
`torchdata.nodes` is PyTorch's composable, stateful data-loading graph
abstraction.  Exposing dino_loader's stages as ``BaseNode`` subclasses gives
three concrete benefits:

1. **Composability** — users can plug any ``torchdata.nodes`` transform
   (``Mapper``, ``Prefetcher``, ``PinMemory``, …) directly onto the shard
   reader output without touching loader internals.

2. **Standardised state_dict** — ``tn.Loader`` automatically handles
   ``state_dict`` / ``load_state_dict`` for the whole graph, making
   mid-epoch resumption first-class.

3. **Standalone usage** — the most expensive part of dino_loader (the
   Lustre-aware shard I/O + mixing pipeline) can now be used without DALI,
   making it easy to front-end a plain PyTorch or custom augmentation stack.

Public API
----------
::

    from dino_loader.nodes import (
        ShardReaderNode,    # stages 1-2: shard I/O + sample mixing
        MetadataNode,       # pops per-sample JSON metadata from the reader
        MaskMapNode,        # attaches iBOT patch masks to Batch objects
        build_reader_graph, # convenience factory → tn.Loader
    )

Corrections apportées
---------------------
[FIX-RESET] ShardReaderNode.reset() ne recrée plus MixingSource à chaque
            appel.  ``tn.Loader(restart_on_stop_iteration=True)`` appelle
            reset() à chaque époque, recréer tous les threads ShardIterator
            était donc coûteux et inutile.  reset() appelle désormais
            set_epoch() sur la source existante, comme le fait DINODataLoader.
            La source n'est créée qu'une seule fois à la première construction.
[FIX-POOL]  Le pool d'extraction partagé (SharedExtractionPoolConfig) est
            transmis à MixingSource afin que le budget de threads soit borné
            même via le chemin torchdata.nodes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torchdata.nodes as tn
from torchdata.nodes import BaseNode
from dino_datasets import DatasetSpec

from dino_loader.config import SharedExtractionPoolConfig
from dino_loader.memory import Batch
from dino_loader.mixing_source import MixingSource, SamplePredicate

log = logging.getLogger(__name__)

# Type alias: un batch brut retourné par ShardReaderNode.
ReaderBatch = tuple[list[np.ndarray], list[dict[str, Any] | None]]


# ---------------------------------------------------------------------------
# ShardReaderNode
# ---------------------------------------------------------------------------


class ShardReaderNode(BaseNode):  # type: ignore[misc]
    """torchdata BaseNode wrapping stages 1-2 of the dino_loader pipeline.

    Stage 1 — ``NodeSharedShardCache`` / ``InProcessShardCache``:
        Reads shard bytes from Lustre (rank 0) or /dev/shm (other ranks).

    Stage 2 — ``MixingSource``:
        Weighted multi-dataset sample mixing with per-shard quality filtering.

    Each call to ``next()`` returns a ``ReaderBatch``:
    ``(list[np.ndarray], list[dict | None])`` — raw JPEG bytes and optional
    JSON metadata, both of length ``batch_size``.

    Note on reset()
    ---------------
    ``reset()`` ne recrée **pas** ``MixingSource`` à chaque appel.  Cela
    évite la reconstruction de tous les threads d'extraction à chaque époque.
    La source est construite une seule fois dans ``reset()`` lors de la
    première invocation, puis ``set_epoch()`` est appelé pour les suivantes.

    Args:
        specs: Dataset specifications (shards, weights, quality filters, …).
        batch_size: Number of samples per batch.
        cache: Shard cache instance.
        rank: Global rank of this process.
        world_size: Total number of processes.
        pool_cfg: Shared extraction pool config.  Controls total thread budget.
        seed: Base RNG seed.
        device_id: Local GPU index (forwarded to ``MixingSource``).
        shuffle_buffer_size: Reservoir size for per-shard sample shuffling.
        debug_log_keys: Optional path to write per-sample key audit log.
        sample_predicate: Optional early filter applied before DALI decode.

    """

    _KEY_EPOCH   = "epoch"
    _KEY_WEIGHTS = "mixing_weights"
    _KEY_NAMES   = "dataset_names"

    def __init__(
        self,
        specs:               list[DatasetSpec],
        batch_size:          int,
        cache:               Any,
        rank:                int,
        world_size:          int,
        *,
        pool_cfg:            SharedExtractionPoolConfig | None = None,
        seed:                int                    = 0,
        device_id:           int                    = 0,
        shuffle_buffer_size: int                    = 512,
        debug_log_keys:      str | None             = None,
        sample_predicate:    SamplePredicate | None = None,
    ) -> None:
        """Initialise a ShardReaderNode."""
        super().__init__()

        self._specs         = specs
        self._batch_size    = batch_size
        self._cache         = cache
        self._rank          = rank
        self._world_size    = world_size
        self._pool_cfg      = pool_cfg
        self._seed          = seed
        self._device_id     = device_id
        self._shuffle_buf   = shuffle_buffer_size
        self._debug_log     = debug_log_keys
        self._predicate     = sample_predicate

        self._epoch: int                  = 0
        self._source: MixingSource | None = None

    # ------------------------------------------------------------------
    # BaseNode protocol
    # ------------------------------------------------------------------

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Prépare la source pour une nouvelle époque.

        [FIX-RESET] La source n'est créée qu'à la première invocation.
        Pour les époques suivantes, set_epoch() est appelé sur la source
        existante, ce qui évite de recréer tous les threads d'extraction.

        Args:
            initial_state: Optional state dict from a prior :meth:`get_state` call.

        """
        super().reset(initial_state)

        if self._source is None:
            # Première construction : crée la source et son pool.
            self._source = MixingSource(
                specs               = self._specs,
                batch_size          = self._batch_size,
                cache               = self._cache,
                rank                = self._rank,
                world_size          = self._world_size,
                pool_cfg            = self._pool_cfg,
                seed                = self._seed,
                device_id           = self._device_id,
                shuffle_buffer_size = self._shuffle_buf,
                debug_log_keys      = self._debug_log,
                sample_predicate    = self._predicate,
            )
            log.debug(
                "ShardReaderNode: MixingSource created, rank=%d/%d datasets=%s",
                self._rank, self._world_size,
                [s.name for s in self._specs],
            )

        if initial_state is not None:
            self._restore_state(initial_state)
        else:
            # Réinitialise l'époque sur la source existante sans recréer de threads.
            self._source.set_epoch(self._epoch)
            log.debug("ShardReaderNode.reset: set_epoch(%d)", self._epoch)

    def next(self) -> ReaderBatch:
        """Return the next batch as (jpeg_list, metadata_list).

        Raises:
            AssertionError: If called before :meth:`reset`.

        """
        assert self._source is not None, "reset() must be called before next()"
        jpegs = self._source()
        meta  = self._source.pop_last_metadata()
        return jpegs, meta

    def get_state(self) -> dict[str, Any]:
        """Persist epoch + mixing weights (within-epoch position not saved)."""
        weights = self._source.current_weights if self._source is not None else []
        names   = self._source.dataset_names   if self._source is not None else []
        return {
            self._KEY_EPOCH:   self._epoch,
            self._KEY_WEIGHTS: weights,
            self._KEY_NAMES:   names,
        }

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Advance to a new epoch (reshuffles shard order)."""
        self._epoch = epoch
        if self._source is not None:
            self._source.set_epoch(epoch)

    def set_weights(self, weights: list[float]) -> None:
        """Update dataset mixing weights (re-normalised automatically)."""
        if self._source is not None:
            self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Update one dataset's weight by name."""
        if self._source is not None:
            self._source.set_by_name(name, weight)

    @property
    def current_weights(self) -> list[float]:
        """Current normalised mixing weights."""
        if self._source is None:
            return []
        return self._source.current_weights

    @property
    def dataset_names(self) -> list[str]:
        """Ordered list of dataset names."""
        if self._source is None:
            return [s.name for s in self._specs]
        return self._source.dataset_names

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _restore_state(self, state: dict[str, Any]) -> None:
        """Apply a previously persisted state dict."""
        assert self._source is not None

        saved_names: list[str] = state.get(self._KEY_NAMES, [])
        if saved_names and saved_names != self._source.dataset_names:
            log.warning(
                "ShardReaderNode: checkpoint dataset names %s differ from "
                "current specs %s — skipping weight restore.",
                saved_names, self._source.dataset_names,
            )
        else:
            saved_weights: list[float] = state.get(self._KEY_WEIGHTS, [])
            if saved_weights:
                self._source.set_weights(saved_weights)

        saved_epoch: int = state.get(self._KEY_EPOCH, 0)
        self._epoch = saved_epoch
        self._source.set_epoch(saved_epoch)

    def __del__(self) -> None:
        """Close the underlying MixingSource on GC."""
        source = getattr(self, "_source", None)
        if source is not None:
            try:
                source.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# MetadataNode
# ---------------------------------------------------------------------------


class MetadataNode(BaseNode):  # type: ignore[misc]
    """Splits a ``ReaderBatch`` stream into JPEG and metadata components."""

    def __init__(self, source: BaseNode) -> None:  # type: ignore[type-arg]
        """Initialise a MetadataNode."""
        super().__init__()
        self._source            = source
        self._last_meta: list[dict[str, Any] | None] = []

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Reset the upstream source and clear buffered metadata."""
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._last_meta = []

    def next(self) -> tuple[list[np.ndarray], list[dict[str, Any] | None]]:
        """Return the next batch, buffering metadata for later retrieval."""
        jpegs, meta     = self._source.next()
        self._last_meta = meta
        return jpegs, meta

    def get_state(self) -> dict[str, Any]:
        """Delegate state to the upstream source."""
        return self._source.get_state()

    def pop_last_metadata(self) -> list[dict[str, Any] | None]:
        """Return metadata from the last ``next()`` call and clear the buffer."""
        meta, self._last_meta = self._last_meta, []
        return meta


# ---------------------------------------------------------------------------
# MaskMapNode
# ---------------------------------------------------------------------------


class MaskMapNode(BaseNode):  # type: ignore[misc]
    """torchdata BaseNode that attaches iBOT patch masks to every batch."""

    def __init__(
        self,
        source,
        mask_generator,
        num_masking_patches: int | None = None,
    ) -> None:
        """Initialise a MaskMapNode."""
        super().__init__()
        self._source = source
        self._gen    = mask_generator
        self._n_mask = num_masking_patches

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Reset the upstream source."""
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self) -> Batch:
        """Return the next batch with ``Batch.masks`` populated."""
        batch: Batch = self._source.next()
        if not batch.global_crops:
            msg = (
                "MaskMapNode.next: batch.global_crops is empty. "
                "MaskMapNode requires at least one global crop to derive the "
                "batch dimension for mask expansion."
            )
            raise ValueError(msg)

        mask  = self._gen(flat=True)
        masks = torch.from_numpy(mask).unsqueeze(0)
        b     = batch.global_crops[0].shape[0]
        masks = masks.expand(b, -1)

        batch.masks = masks
        return batch

    def get_state(self) -> dict[str, Any]:
        """Delegate state to the upstream source."""
        return self._source.get_state()

    @staticmethod
    def as_transform(
        mask_generator,
        num_masking_patches: int | None = None,
    ) -> "Callable[[Batch], Batch]":  # noqa: F821
        """Return a ``Batch → Batch`` callable for use with ``.map()``."""

        def _apply(batch: Batch) -> Batch:
            if not batch.global_crops:
                msg = (
                    "MaskMapNode.as_transform: batch.global_crops is empty. "
                    "Masking requires at least one global crop."
                )
                raise ValueError(msg)

            mask  = mask_generator(flat=True)
            masks = torch.from_numpy(mask).unsqueeze(0)
            b     = batch.global_crops[0].shape[0]
            masks = masks.expand(b, -1)

            return Batch(
                global_crops = batch.global_crops,
                local_crops  = batch.local_crops,
                metadata     = batch.metadata,
                masks        = masks,
            )

        return _apply


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_reader_graph(
    specs:               list[DatasetSpec],
    batch_size:          int,
    cache:               Any,
    rank:                int,
    world_size:          int,
    *,
    pool_cfg:            SharedExtractionPoolConfig | None = None,
    seed:                int                    = 0,
    device_id:           int                    = 0,
    shuffle_buffer_size: int                    = 512,
    debug_log_keys:      str | None             = None,
    sample_predicate:    SamplePredicate | None = None,
    prefetch_factor:     int                    = 2,
) -> tuple[tn.Loader[ReaderBatch], ShardReaderNode]:
    """Build a ready-to-use ``tn.Loader`` over the shard reader pipeline.

    Args:
        specs: Dataset specifications.
        batch_size: Samples per batch.
        cache: Shard cache.
        rank: Global rank.
        world_size: Total ranks.
        pool_cfg: Shared extraction pool configuration.
        seed: Base RNG seed.
        device_id: GPU index.
        shuffle_buffer_size: Per-shard reservoir depth.
        debug_log_keys: Optional audit log path.
        sample_predicate: Optional early filter.
        prefetch_factor: ``tn.Prefetcher`` look-ahead depth.

    Returns:
        ``(loader, reader_node)`` — iterate over ``loader``, call
        ``reader_node.set_epoch(e)`` each epoch.

    """
    reader: ShardReaderNode = ShardReaderNode(
        specs               = specs,
        batch_size          = batch_size,
        cache               = cache,
        rank                = rank,
        world_size          = world_size,
        pool_cfg            = pool_cfg,
        seed                = seed,
        device_id           = device_id,
        shuffle_buffer_size = shuffle_buffer_size,
        debug_log_keys      = debug_log_keys,
        sample_predicate    = sample_predicate,
    )

    prefetched: tn.Prefetcher = tn.Prefetcher(reader, prefetch_factor=prefetch_factor)
    loader: tn.Loader         = tn.Loader(prefetched, restart_on_stop_iteration=True)

    return loader, reader
