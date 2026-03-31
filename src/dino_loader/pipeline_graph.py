"""dino_loader.pipeline_graph
============================
``torchdata.nodes``-based post-processing pipeline.

[FIX-RESET] _DINOLoaderNode.reset() : la restauration de ``_num_yielded``
depuis initial_state était trompeuse car l'itérateur repart toujours de
zéro (état DALI non sérialisable).  Le commentaire et la doc sont maintenant
explicites sur ce point, et la restauration de position intra-époque est
retirée pour éviter une fausse impression de résumabilité fine.

Le chemin recommandé pour le post-processing est celui-ci. DINODataLoader
n'expose plus .map()/.select()/.with_epoch() directement.

Usage::

    from dino_loader.pipeline_graph import wrap_loader

    pipeline = (
        wrap_loader(DINODataLoader(...))
        .map(apply_ibot_masks)
        .select(quality_ok)
        .with_epoch(steps_per_epoch)
    )

    for epoch in range(100):
        pipeline.set_epoch(epoch)
        sd = pipeline.state_dict()
        for batch in pipeline:
            train_step(batch)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torchdata.nodes as tn
from torchdata.nodes import BaseNode

from dino_loader.memory import Batch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive nodes
# ---------------------------------------------------------------------------


class BatchMapNode(BaseNode):  # type: ignore[misc]
    """Apply a ``Batch → Batch`` function to every element of a node stream."""

    def __init__(
        self,
        source: BaseNode,  # type: ignore[type-arg]
        fn:     Callable[[Batch], Batch],
        *,
        label:  str = "<map>",
    ) -> None:
        """Initialise a BatchMapNode."""
        super().__init__()
        self._source = source
        self._fn     = fn
        self._label  = label

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Reset upstream source."""
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self) -> Batch:
        """Apply fn to the next batch from source."""
        return self._fn(self._source.next())

    def get_state(self) -> dict[str, Any]:
        """Delegate to source."""
        return self._source.get_state()

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return f"BatchMapNode({self._label!r})"


class BatchFilterNode(BaseNode):  # type: ignore[misc]
    """Skip ``Batch`` objects for which a predicate returns ``False``."""

    def __init__(
        self,
        source:    BaseNode,  # type: ignore[type-arg]
        predicate: Callable[[Batch], bool],
        *,
        label:     str = "<filter>",
    ) -> None:
        """Initialise a BatchFilterNode."""
        super().__init__()
        self._source    = source
        self._predicate = predicate
        self._label     = label
        self._n_skipped = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Reset upstream source and clear skip counter."""
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._n_skipped = 0

    def next(self) -> Batch:
        """Return the next batch accepted by the predicate."""
        while True:
            batch = self._source.next()
            if self._predicate(batch):
                return batch
            self._n_skipped += 1
            self._record_filtered()

    def get_state(self) -> dict[str, Any]:
        """Delegate to source."""
        return self._source.get_state()

    @property
    def n_skipped(self) -> int:
        """Running count of batches rejected by the predicate."""
        return self._n_skipped

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return f"BatchFilterNode({self._label!r}, skipped={self._n_skipped})"

    @staticmethod
    def _record_filtered() -> None:
        """Increment the batches_filtered metric if the registry is available."""
        try:
            from dino_loader.monitor.metrics import get_registry  # noqa: PLC0415
            reg = get_registry()
            if reg is not None:
                reg.inc("batches_filtered", 1)
        except Exception:  # noqa: BLE001
            pass


class _LimitNode(BaseNode):  # type: ignore[misc]
    """Yield at most ``max_steps`` batches per epoch, then raise StopIteration."""

    def __init__(self, source: BaseNode, max_steps: int) -> None:  # type: ignore[type-arg]
        """Initialise a _LimitNode."""
        super().__init__()
        self._source    = source
        self._max_steps = max_steps
        self._yielded   = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Reset upstream and clear yield counter."""
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._yielded = 0

    def next(self) -> Batch:
        """Return next batch or raise StopIteration when limit is reached."""
        if self._yielded >= self._max_steps:
            raise StopIteration
        batch = self._source.next()
        self._yielded += 1
        return batch

    def get_state(self) -> dict[str, Any]:
        """Return source state (limit counter not persisted)."""
        return self._source.get_state()


# ---------------------------------------------------------------------------
# NodePipeline
# ---------------------------------------------------------------------------


class NodePipeline:
    """Composable, stateful post-processing pipeline built on ``torchdata.nodes``.

    Do not instantiate directly; use :func:`wrap_loader` instead.
    """

    def __init__(
        self,
        root_node:   BaseNode,  # type: ignore[type-arg]
        dino_loader: Any,
        max_steps:   int | None = None,
    ) -> None:
        """Initialise a NodePipeline."""
        self._root      = root_node
        self._loader    = dino_loader
        self._max_steps = max_steps
        self._tn_loader: tn.Loader | None = None

    def map(self, fn: Callable[[Batch], Batch], *, label: str = "<map>") -> NodePipeline:
        """Append a ``Batch → Batch`` transform."""
        return NodePipeline(
            root_node   = BatchMapNode(self._root, fn, label=label),
            dino_loader = self._loader,
            max_steps   = self._max_steps,
        )

    def select(
        self,
        predicate: Callable[[Batch], bool],
        *,
        label: str = "<filter>",
    ) -> NodePipeline:
        """Drop batches for which ``predicate(batch)`` is ``False``."""
        return NodePipeline(
            root_node   = BatchFilterNode(self._root, predicate, label=label),
            dino_loader = self._loader,
            max_steps   = self._max_steps,
        )

    def with_epoch(self, n_steps: int) -> NodePipeline:
        """Limit iteration to ``n_steps`` batches per epoch."""
        return NodePipeline(
            root_node   = self._root,
            dino_loader = self._loader,
            max_steps   = n_steps,
        )

    def _build_tn_loader(self) -> tn.Loader:
        """Build the underlying tn.Loader, adding a _LimitNode if needed."""
        root = self._root
        if self._max_steps is not None:
            root = _LimitNode(root, self._max_steps)
        return tn.Loader(root, restart_on_stop_iteration=True)

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        if self._tn_loader is None:
            self._tn_loader = self._build_tn_loader()
        return iter(self._tn_loader)

    def __len__(self) -> int:
        """Return max_steps if set, else delegate to the underlying loader."""
        if self._max_steps is not None:
            return self._max_steps
        return len(self._loader)

    def state_dict(self) -> dict[str, Any]:
        """Return the full graph state dict.

        Note on intra-epoch resumption
        --------------------------------
        The DINODataLoader state (epoch, mixing weights, resolution) is fully
        persisted.  Within-epoch position is **not** resumable: DALI pipeline
        state is not serialisable, and the torchdata node graph position only
        records how many batches were yielded (``_num_yielded``), not the
        underlying shard position.  On restore, iteration restarts from the
        beginning of the epoch with the correct epoch seed.
        """
        loader_sd = self._loader.state_dict()
        tn_sd: dict[str, Any] = {}
        if self._tn_loader is not None:
            tn_sd = self._tn_loader.state_dict()
        return {"loader": loader_sd, "tn_graph": tn_sd}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore the full graph state."""
        if "loader" in sd:
            self._loader.load_state_dict(sd["loader"])
        if "tn_graph" in sd and self._tn_loader is not None:
            self._tn_loader.load_state_dict(sd["tn_graph"])

    def set_epoch(self, epoch: int) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.set_epoch(epoch)

    def checkpoint(self, step: int) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.checkpoint(step)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.set_weight_by_name(name, weight)

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.set_resolution(global_size, local_size)

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Current crop resolution as ``(global_size, local_size)``."""
        return self._loader.current_resolution


# ---------------------------------------------------------------------------
# _DINOLoaderNode — bridges DINODataLoader into the node graph
# ---------------------------------------------------------------------------


class _DINOLoaderNode(BaseNode):  # type: ignore[misc]
    """Wraps a ``DINODataLoader`` as a ``torchdata.nodes.BaseNode``.

    Restauration de position intra-époque
    ---------------------------------------
    ``get_state()`` persiste ``_num_yielded`` pour information, mais
    ``reset()`` ne tente **pas** de sauter des batches pour retrouver la
    position.  La raison : DALI pipeline state is not serialisable.  Sur
    restore, l'itération repart du début de l'époque avec le bon seed.
    Cela est cohérent avec le comportement de DINODataLoader sans torchdata.
    """

    def __init__(self, dino_loader: Any) -> None:
        """Initialise a _DINOLoaderNode."""
        super().__init__()
        self._dino_loader  = dino_loader
        self._it: Iterator[Batch] | None = None
        self._num_yielded  = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Re-enter the loader iterator for a new epoch.

        La position intra-époque n'est pas restaurée (voir docstring classe).
        """
        super().reset(initial_state)
        self._it          = iter(self._dino_loader)
        self._num_yielded = 0

    def next(self) -> Batch:
        """Return the next Batch from the loader."""
        assert self._it is not None, "reset() must be called before next()"
        batch = next(self._it)
        self._num_yielded += 1
        return batch

    def get_state(self) -> dict[str, Any]:
        """Return loader state dict plus the number of batches yielded this epoch."""
        loader_sd: dict[str, Any] = {}
        try:
            loader_sd = self._dino_loader.state_dict()
        except RuntimeError:
            # stateful_dataloader=False — skip gracefully.
            pass
        return {**loader_sd, "_num_yielded": self._num_yielded}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def wrap_loader(dino_loader: Any) -> NodePipeline:
    """Wrap a ``DINODataLoader`` in a composable ``NodePipeline``.

    Args:
        dino_loader: A ``DINODataLoader`` instance.

    Returns:
        ``NodePipeline`` — iterate directly or compose further with
        ``.map()``, ``.select()``, ``.with_epoch()``.

    Example::

        from dino_loader.pipeline_graph import wrap_loader

        pipeline = (
            wrap_loader(DINODataLoader(...))
            .map(apply_ibot_masks)
            .select(lambda b: any(m is not None for m in b.metadata))
            .with_epoch(steps_per_epoch)
        )

        for epoch in range(100):
            pipeline.set_epoch(epoch)
            for batch in pipeline:
                train_step(batch)
            sd = pipeline.state_dict()

    """
    root = _DINOLoaderNode(dino_loader)
    return NodePipeline(root_node=root, dino_loader=dino_loader)
