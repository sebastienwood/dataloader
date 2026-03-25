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
        build_reader_graph, # convenience factory → tn.Loader
    )

Usage
-----
Standalone (no DALI)::

    from dino_loader.nodes import build_reader_graph
    import torchdata.nodes as tn

    loader = build_reader_graph(
        specs       = [DatasetSpec(...)],
        batch_size  = 512,
        cache       = my_shard_cache,
        rank        = env.rank,
        world_size  = env.world_size,
    )
    for batch_jpegs, batch_meta in loader:
        # batch_jpegs: list[np.ndarray]   — raw JPEG bytes, one per sample
        # batch_meta:  list[dict | None]  — parsed JSON sidecars
        ...

With a custom augmentation mapper::

    from dino_loader.nodes import ShardReaderNode
    import torchdata.nodes as tn

    reader = ShardReaderNode(specs=specs, batch_size=512, ...)
    augmented = tn.Mapper(reader, fn=my_aug_fn)
    prefetched = tn.Prefetcher(augmented, prefetch_factor=4)
    loader = tn.Loader(prefetched)

    for epoch in range(100):
        loader.state_dict()   # full graph state
        for batch in loader:
            ...

Notes
-----
- ``ShardReaderNode`` holds a live ``MixingSource`` internally; it is
  **not** safe to share one instance across processes.
- ``state_dict`` persists epoch number and mixing weights; within-epoch
  sample position is *not* restored (matching existing dino_loader semantics).
- ``torchdata`` is listed as an optional dependency; a clear ``ImportError``
  is raised on import if it is absent.

"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from dino_datasets import DatasetSpec

from dino_loader.mixing_source import MixingSource, SamplePredicate

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard — torchdata is not required at import time of the
# rest of dino_loader.
# ---------------------------------------------------------------------------
try:
    import torchdata.nodes as tn
    from torchdata.nodes import BaseNode

    _HAS_TORCHDATA = True
except ImportError:
    _HAS_TORCHDATA = False
    # Provide a stub so the module is still importable; the error surfaces only
    # when a Node is actually instantiated.
    tn = None  # type: ignore[assignment]
    BaseNode = object  # type: ignore[assignment,misc]


def _require_torchdata() -> None:
    if not _HAS_TORCHDATA:
        msg = (
            "torchdata is required for dino_loader.nodes.  "
            "Install it with:  pip install torchdata"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: One batch yielded by ShardReaderNode: parallel lists of JPEG bytes and
#: optional metadata dicts, one entry per sample.
ReaderBatch = tuple[list[np.ndarray], list[dict[str, Any] | None]]


# ---------------------------------------------------------------------------
# ShardReaderNode
# ---------------------------------------------------------------------------


class ShardReaderNode(BaseNode):  # type: ignore[misc]
    """torchdata BaseNode wrapping stages 1–2 of the dino_loader pipeline.

    Stage 1 — ``NodeSharedShardCache`` / ``InProcessShardCache``:
        Reads shard bytes from Lustre (rank 0) or /dev/shm (other ranks).

    Stage 2 — ``MixingSource``:
        Weighted multi-dataset sample mixing with per-shard quality filtering.

    Each call to ``next()`` returns a ``ReaderBatch``:
    ``(list[np.ndarray], list[dict | None])`` — raw JPEG bytes and optional
    JSON metadata, both of length ``batch_size``.

    Args:
        specs: Dataset specifications (shards, weights, quality filters, …).
        batch_size: Number of samples per batch.
        cache: Shard cache instance (``InProcessShardCache`` or
            ``NodeSharedShardCache``).  Callers are responsible for
            construction so that the cache lifecycle is managed externally.
        rank: Global rank of this process.
        world_size: Total number of processes.
        num_workers: Extraction thread-pool workers per dataset.
        seed: Base RNG seed.
        device_id: Local GPU index (forwarded to ``MixingSource``).
        shuffle_buffer_size: Reservoir size for per-shard sample shuffling.
        debug_log_keys: Optional path to write per-sample key audit log.
        sample_predicate: Optional early filter applied before DALI decode.

    State dict keys
    ---------------
    ``epoch``
        Current epoch number (set via :meth:`set_epoch`).
    ``mixing_weights``
        Normalised mixing weight vector.
    ``dataset_names``
        Ordered list of dataset names (sanity-check on restore).

    """

    # Keys used in get_state / reset(initial_state=...)
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
        num_workers:         int                    = 4,
        seed:                int                    = 0,
        device_id:           int                    = 0,
        shuffle_buffer_size: int                    = 512,
        debug_log_keys:      str | None             = None,
        sample_predicate:    SamplePredicate | None = None,
    ) -> None:
        _require_torchdata()
        super().__init__()

        self._specs       = specs
        self._batch_size  = batch_size
        self._cache       = cache
        self._rank        = rank
        self._world_size  = world_size
        self._num_workers = num_workers
        self._seed        = seed
        self._device_id   = device_id
        self._shuffle_buf = shuffle_buffer_size
        self._debug_log   = debug_log_keys
        self._predicate   = sample_predicate

        self._epoch: int             = 0
        self._source: MixingSource | None = None

    # ------------------------------------------------------------------
    # BaseNode protocol
    # ------------------------------------------------------------------

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """(Re-)initialise the MixingSource, optionally from a saved state."""
        super().reset(initial_state)

        # Close previous source if any (epoch boundary).
        if self._source is not None:
            self._source.close()

        self._source = MixingSource(
            specs               = self._specs,
            batch_size          = self._batch_size,
            cache               = self._cache,
            rank                = self._rank,
            world_size          = self._world_size,
            num_workers         = self._num_workers,
            seed                = self._seed,
            device_id           = self._device_id,
            shuffle_buffer_size = self._shuffle_buf,
            debug_log_keys      = self._debug_log,
            sample_predicate    = self._predicate,
        )

        if initial_state is not None:
            self._restore_state(initial_state)

        log.debug(
            "ShardReaderNode.reset: epoch=%d rank=%d/%d datasets=%s",
            self._epoch, self._rank, self._world_size,
            [s.name for s in self._specs],
        )

    def next(self) -> ReaderBatch:
        """Return the next batch as (jpeg_list, metadata_list)."""
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
        """Advance to a new epoch (reshuffles shard order).

        Must be called before the first ``next()`` of each epoch when using
        deterministic shuffling.  Safe to call concurrently with ``next()``
        via ``MixingSource.set_epoch`` which is itself thread-safe.
        """
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
        if saved_epoch != 0:
            self._epoch = saved_epoch
            self._source.set_epoch(saved_epoch)

    def __del__(self) -> None:
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
    """Splits a ``ReaderBatch`` stream into separate JPEG and metadata streams.

    Takes a source that yields ``(jpeg_list, meta_list)`` tuples and
    re-emits just the ``(jpeg_list, meta_list)`` pair with the metadata
    stored for retrieval via :meth:`pop_last_metadata`.

    In practice this node is useful when you want to pass only the raw
    JPEG bytes into a DALI or other augmentation node while retaining the
    metadata on the side for loss computation or logging.

    Args:
        source: Upstream node yielding ``ReaderBatch`` tuples.

    Example::

        reader   = ShardReaderNode(...)
        splitter = MetadataNode(reader)
        aug_node = tn.Mapper(splitter, fn=lambda jpegs_meta: augment(jpegs_meta[0]))
        loader   = tn.Loader(aug_node)

        for augmented_batch in loader:
            meta = splitter.pop_last_metadata()
            ...

    """

    def __init__(self, source: BaseNode) -> None:  # type: ignore[type-arg]
        _require_torchdata()
        super().__init__()
        self._source            = source
        self._last_meta: list[dict[str, Any] | None] = []

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._last_meta = []

    def next(self) -> tuple[list[np.ndarray], list[dict[str, Any] | None]]:
        jpegs, meta     = self._source.next()
        self._last_meta = meta
        return jpegs, meta

    def get_state(self) -> dict[str, Any]:
        return self._source.get_state()

    def pop_last_metadata(self) -> list[dict[str, Any] | None]:
        """Return metadata from the last ``next()`` call and clear the buffer."""
        meta, self._last_meta = self._last_meta, []
        return meta


class MaskMapNode:
    """torchdata BaseNode that attaches iBOT patch masks to every batch.

    Sits between the augmentation stage and the training loop.  On each call to
    ``next()`` it draws a fresh mask from ``MaskingGenerator`` and stores it on
    ``Batch.masks``.

    The mask shape is derived from ``img_size`` and ``patch_size`` at
    construction time, so the grid dimensions never change during training
    (even when ``set_resolution`` is called — resolution changes that affect
    masking should reconstruct the node).

    Args:
        source: Upstream node yielding ``Batch`` objects.
        mask_generator: A configured ``MaskingGenerator`` instance.
        num_masking_patches: Number of patches to mask per sample.  When
            ``None`` the generator's own default is used.

    Example::

        from dino_loader.masking import MaskingGenerator
        from dino_loader.nodes import MaskMapNode
        from dino_loader.pipeline_graph import wrap_loader

        gen     = MaskingGenerator(input_size=(16, 16), num_masking_patches=60)
        pipeline = wrap_loader(DINODataLoader(...)).map(
            MaskMapNode.as_transform(gen, num_masking_patches=60)
        )

    """

    # Provided as a torchdata BaseNode subclass when torchdata is available,
    # or as a plain callable wrapper usable with PostProcessPipeline.map().

    def __init__(
        self,
        source,              # BaseNode[Batch]
        mask_generator,      # MaskingGenerator
        num_masking_patches: int | None = None,
    ) -> None:
        """Initialise the node."""
        _require_torchdata()
        super().__init__()
        self._source = source
        self._gen    = mask_generator
        self._n_mask = num_masking_patches

    def reset(self, initial_state=None) -> None:
        """Reset the upstream source."""
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self):
        """Return the next batch with ``Batch.masks`` populated."""
        import torch  # noqa: PLC0415 — torch is an optional dep of this node
        batch: Batch = self._source.next()
        n_mask = self._n_mask if self._n_mask is not None else self._gen.num_masking_patches
        mask   = self._gen(flat=True)                  # shape (H*W,)
        masks  = torch.from_numpy(mask).unsqueeze(0)   # (1, H*W) bool
        # Expand to batch dimension if global_crops is populated.
        if batch.global_crops:
            b = batch.global_crops[0].shape[0]
            masks = masks.expand(b, -1)                # (B, H*W)
        batch.masks = masks
        return batch

    def get_state(self):
        """Delegate state to the upstream source."""
        return self._source.get_state()

    # ------------------------------------------------------------------
    # Convenience: use as a plain Batch → Batch transform
    # ------------------------------------------------------------------

    @staticmethod
    def as_transform(mask_generator, num_masking_patches: int | None = None):
        """Return a ``Batch → Batch`` callable for use with ``.map()``.

        Useful when you have a ``PostProcessPipeline`` or ``NodePipeline``
        and want to add masking without building a full ``BaseNode``.

        Args:
            mask_generator: A configured ``MaskingGenerator``.
            num_masking_patches: Patches to mask per sample.

        Returns:
            A callable ``(Batch) -> Batch``.

        Example::

            from dino_loader.pipeline_graph import wrap_loader
            from dino_loader.masking import MaskingGenerator
            from dino_loader.nodes import MaskMapNode

            gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
            pipeline = wrap_loader(loader).map(
                MaskMapNode.as_transform(gen)
            )

        """
        import torch  # noqa: PLC0415

        def _apply(batch):
            from dino_loader.memory import Batch  # noqa: PLC0415
            n_mask = num_masking_patches if num_masking_patches is not None else mask_generator.num_masking_patches
            mask   = mask_generator(flat=True)
            masks  = torch.from_numpy(mask).unsqueeze(0)
            if batch.global_crops:
                b = batch.global_crops[0].shape[0]
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
    num_workers:         int                    = 4,
    seed:                int                    = 0,
    device_id:           int                    = 0,
    shuffle_buffer_size: int                    = 512,
    debug_log_keys:      str | None             = None,
    sample_predicate:    SamplePredicate | None = None,
    prefetch_factor:     int                    = 2,
) -> tuple[tn.Loader[ReaderBatch], ShardReaderNode]:
    """Build a ready-to-use ``tn.Loader`` over the shard reader pipeline.

    Returns both the ``Loader`` (for iteration) and the ``ShardReaderNode``
    (for epoch/weight control).

    Args:
        specs: Dataset specifications.
        batch_size: Samples per batch.
        cache: Shard cache (``InProcessShardCache`` or ``NodeSharedShardCache``).
        rank: Global rank.
        world_size: Total ranks.
        num_workers: Extraction workers per dataset.
        seed: Base RNG seed.
        device_id: GPU index.
        shuffle_buffer_size: Per-shard reservoir depth.
        debug_log_keys: Optional audit log path.
        sample_predicate: Optional early filter.
        prefetch_factor: ``tn.Prefetcher`` look-ahead depth.

    Returns:
        ``(loader, reader_node)`` — iterate over ``loader``, call
        ``reader_node.set_epoch(e)`` each epoch.

    Example::

        loader, reader = build_reader_graph(specs, batch_size=512, ...)
        for epoch in range(100):
            reader.set_epoch(epoch)
            for jpegs, meta in loader:
                my_augment(jpegs)

    """
    _require_torchdata()

    reader: ShardReaderNode = ShardReaderNode(
        specs               = specs,
        batch_size          = batch_size,
        cache               = cache,
        rank                = rank,
        world_size          = world_size,
        num_workers         = num_workers,
        seed                = seed,
        device_id           = device_id,
        shuffle_buffer_size = shuffle_buffer_size,
        debug_log_keys      = debug_log_keys,
        sample_predicate    = sample_predicate,
    )

    prefetched: tn.Prefetcher = tn.Prefetcher(reader, prefetch_factor=prefetch_factor)
    loader: tn.Loader         = tn.Loader(prefetched, restart_on_stop_iteration=True)

    return loader, reader
