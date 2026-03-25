"""dino_loader.pipeline_graph
============================
``torchdata.nodes``-based post-processing pipeline that replaces the ad-hoc
``PostProcessPipeline`` for the Batch → Batch transformation stage.

Background
----------
The original ``PostProcessPipeline`` in ``loader.py`` is a hand-rolled lazy
iterator with ``.map()``, ``.select()``, and ``.with_epoch()`` methods.  It
works, but lacks:

- Proper ``state_dict`` across the full graph (only the underlying loader was
  checkpointed, not pipeline position).
- Backpressure between transforms.
- Composability with ecosystem tools (e.g. ``tn.PinMemory``).

This module re-implements the same API on top of ``torchdata.nodes``.  The
original ``PostProcessPipeline`` is preserved in ``loader.py`` for backward
compatibility; users who want the improved version can opt in via
:func:`wrap_loader`.

Public API
----------
::

    from dino_loader.pipeline_graph import wrap_loader, BatchFilterNode, BatchMapNode

    loader, reader = build_reader_graph(...)
    pipeline = (
        wrap_loader(base_dino_loader)
        .map(apply_ibot_masks)
        .select(quality_ok)
        .with_epoch(steps_per_epoch)
    )

    for epoch in range(100):
        pipeline.set_epoch(epoch)
        sd = pipeline.state_dict()          # full graph state
        for batch in pipeline:
            train_step(batch)

Notes
-----
- ``torchdata`` is an optional dependency; a clear ``ImportError`` is raised
  if it is absent.
- ``BatchFilterNode`` is the ``select()`` equivalent; it does *not* rebatch —
  rejected batches are simply skipped.
- ``state_dict`` / ``load_state_dict`` are delegated to the underlying
  ``tn.Loader`` which recursively serialises the whole node graph.

"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from dino_loader.memory import Batch

log = logging.getLogger(__name__)

try:
    import torchdata.nodes as tn
    from torchdata.nodes import BaseNode

    _HAS_TORCHDATA = True
except ImportError:
    _HAS_TORCHDATA = False
    tn = None  # type: ignore[assignment]
    BaseNode = object  # type: ignore[assignment,misc]


def _require_torchdata() -> None:
    if not _HAS_TORCHDATA:
        msg = (
            "torchdata is required for dino_loader.pipeline_graph.  "
            "Install it with:  pip install torchdata"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Primitive nodes
# ---------------------------------------------------------------------------


class BatchMapNode(BaseNode):  # type: ignore[misc]
    """Apply a ``Batch → Batch`` function to every element of a node stream.

    Equivalent to ``tn.Mapper`` but typed for ``Batch`` objects and with a
    more descriptive name in node graphs.

    Args:
        source: Upstream node yielding ``Batch`` objects.
        fn: Transform function ``(Batch) -> Batch``.
        label: Human-readable label shown in logs/repr.

    """

    def __init__(
        self,
        source: BaseNode,  # type: ignore[type-arg]
        fn:     Callable[[Batch], Batch],
        *,
        label:  str = "<map>",
    ) -> None:
        _require_torchdata()
        super().__init__()
        self._source = source
        self._fn     = fn
        self._label  = label

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self) -> Batch:
        return self._fn(self._source.next())

    def get_state(self) -> dict[str, Any]:
        return self._source.get_state()

    def __repr__(self) -> str:
        return f"BatchMapNode({self._label!r})"


class BatchFilterNode(BaseNode):  # type: ignore[misc]
    """Skip ``Batch`` objects for which a predicate returns ``False``.

    Unlike ``select()`` in the original ``PostProcessPipeline``, this node
    keeps pulling from ``source`` until it finds an accepted batch — so it
    never yields ``None``.  Rejected batches are counted via the metrics
    registry if available.

    Args:
        source: Upstream node yielding ``Batch`` objects.
        predicate: ``(Batch) -> bool`` — return ``True`` to keep, ``False`` to skip.
        label: Human-readable label for logs.

    """

    def __init__(
        self,
        source:    BaseNode,  # type: ignore[type-arg]
        predicate: Callable[[Batch], bool],
        *,
        label:     str = "<filter>",
    ) -> None:
        _require_torchdata()
        super().__init__()
        self._source    = source
        self._predicate = predicate
        self._label     = label
        self._n_skipped = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._n_skipped = 0

    def next(self) -> Batch:
        while True:
            batch = self._source.next()
            if self._predicate(batch):
                return batch
            self._n_skipped += 1
            self._record_filtered()

    def get_state(self) -> dict[str, Any]:
        return self._source.get_state()

    @property
    def n_skipped(self) -> int:
        """Running count of batches rejected by the predicate."""
        return self._n_skipped

    def __repr__(self) -> str:
        return f"BatchFilterNode({self._label!r}, skipped={self._n_skipped})"

    @staticmethod
    def _record_filtered() -> None:
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
        _require_torchdata()
        super().__init__()
        self._source    = source
        self._max_steps = max_steps
        self._yielded   = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._yielded = initial_state.get("_yielded", 0) if initial_state else 0

    def next(self) -> Batch:
        if self._yielded >= self._max_steps:
            raise StopIteration
        batch = self._source.next()
        self._yielded += 1
        return batch

    def get_state(self) -> dict[str, Any]:
        return {**self._source.get_state(), "_yielded": self._yielded}


# ---------------------------------------------------------------------------
# NodePipeline — the composable wrapper
# ---------------------------------------------------------------------------


class NodePipeline:
    """Composable, stateful post-processing pipeline built on ``torchdata.nodes``.

    Mirrors the ``PostProcessPipeline`` API in ``loader.py`` but delegates
    state management to ``tn.Loader``, which recursively serialises the
    entire node graph.

    Do not instantiate directly; use :func:`wrap_loader` instead.

    Methods
    -------
    map(fn) → NodePipeline
        Append a ``Batch → Batch`` transform.
    select(pred) → NodePipeline
        Drop batches for which ``pred(batch)`` is ``False``.
    with_epoch(n) → NodePipeline
        Limit iteration to ``n`` steps per epoch.
    set_epoch(e)
        Delegate to the underlying ``DINODataLoader``.
    checkpoint(step)
        Delegate to the underlying ``DINODataLoader``.
    state_dict() / load_state_dict(sd)
        Full graph state via ``tn.Loader``.

    """

    def __init__(
        self,
        root_node:   BaseNode,  # type: ignore[type-arg]
        dino_loader: Any,
        max_steps:   int | None = None,
    ) -> None:
        _require_torchdata()
        self._root      = root_node
        self._loader    = dino_loader
        self._max_steps = max_steps
        self._tn_loader: tn.Loader | None = None  # built lazily

    # ------------------------------------------------------------------
    # Fluent composition API
    # ------------------------------------------------------------------

    def map(self, fn: Callable[[Batch], Batch], *, label: str = "<map>") -> NodePipeline:
        """Append a ``Batch → Batch`` transform.

        Args:
            fn: Transform applied to every batch.
            label: Optional label for debugging.

        Returns:
            A new ``NodePipeline`` with ``fn`` appended.

        """
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
        """Drop batches for which ``predicate(batch)`` is ``False``.

        Args:
            predicate: Return ``True`` to keep a batch, ``False`` to skip it.
            label: Optional label for debugging.

        Returns:
            A new ``NodePipeline`` with the filter appended.

        """
        return NodePipeline(
            root_node   = BatchFilterNode(self._root, predicate, label=label),
            dino_loader = self._loader,
            max_steps   = self._max_steps,
        )

    def with_epoch(self, n_steps: int) -> NodePipeline:
        """Limit iteration to ``n_steps`` batches per epoch.

        Args:
            n_steps: Maximum number of batches yielded before ``StopIteration``.

        Returns:
            A new ``NodePipeline`` bounded to ``n_steps``.

        """
        return NodePipeline(
            root_node   = self._root,
            dino_loader = self._loader,
            max_steps   = n_steps,
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _build_tn_loader(self) -> tn.Loader:
        root = self._root
        if self._max_steps is not None:
            root = _LimitNode(root, self._max_steps)
        return tn.Loader(root, restart_on_stop_iteration=True)

    def __iter__(self) -> Iterator[Batch]:
        if self._tn_loader is None:
            self._tn_loader = self._build_tn_loader()
        return iter(self._tn_loader)

    def __len__(self) -> int:
        if self._max_steps is not None:
            return self._max_steps
        return len(self._loader)

    # ------------------------------------------------------------------
    # State management (full graph via tn.Loader)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return the full graph state dict.

        Includes the underlying ``DINODataLoader`` state *and* the
        ``torchdata`` node graph position.
        """
        loader_sd = self._loader.state_dict()
        if self._tn_loader is not None:
            tn_sd = self._tn_loader.state_dict()
        else:
            tn_sd = {}
        return {"loader": loader_sd, "tn_graph": tn_sd}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore the full graph state.

        Args:
            sd: State dict previously produced by :meth:`state_dict`.

        """
        if "loader" in sd:
            self._loader.load_state_dict(sd["loader"])
        if "tn_graph" in sd and self._tn_loader is not None:
            self._tn_loader.load_state_dict(sd["tn_graph"])

    # ------------------------------------------------------------------
    # DINODataLoader delegation
    # ------------------------------------------------------------------

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
# IterableWrapperNode — bridges DINODataLoader into the node graph
# ---------------------------------------------------------------------------


class _DINOLoaderNode(BaseNode):  # type: ignore[misc]
    """Wraps a ``DINODataLoader`` as a ``torchdata.nodes.BaseNode``.

    This is an internal bridge node; end users access it via :func:`wrap_loader`.

    The ``DINODataLoader`` already manages its own async stages internally.
    This node simply drives its ``__iter__`` and forwards ``set_epoch``
    calls via the ``NodePipeline`` control API.

    State dict key
    --------------
    ``_num_yielded`` — number of batches yielded in the current epoch, used
    to detect within-epoch position on restore (best-effort; DALI state is
    not fully serialisable).
    """

    def __init__(self, dino_loader: Any) -> None:
        _require_torchdata()
        super().__init__()
        self._dino_loader  = dino_loader
        self._it: Iterator[Batch] | None = None
        self._num_yielded  = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        # Re-enter the loader iterator (triggers set_epoch internally via
        # the caller's set_epoch → loader.set_epoch chain).
        self._it          = iter(self._dino_loader)
        self._num_yielded = 0
        if initial_state:
            self._num_yielded = initial_state.get("_num_yielded", 0)

    def next(self) -> Batch:
        assert self._it is not None, "reset() must be called before next()"
        batch = next(self._it)
        self._num_yielded += 1
        return batch

    def get_state(self) -> dict[str, Any]:
        loader_sd: dict[str, Any] = {}
        try:
            loader_sd = self._dino_loader.state_dict()
        except RuntimeError:
            # stateful_dataloader=False — skip gracefully
            pass
        return {**loader_sd, "_num_yielded": self._num_yielded}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def wrap_loader(dino_loader: Any) -> NodePipeline:
    """Wrap a ``DINODataLoader`` in a composable ``NodePipeline``.

    This is the recommended entry point for Phase-3 usage.  The returned
    ``NodePipeline`` supports the same fluent API as ``PostProcessPipeline``
    but with full ``state_dict`` support across the entire graph.

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
            # Save full graph state
            sd = pipeline.state_dict()

    """
    _require_torchdata()
    root = _DINOLoaderNode(dino_loader)
    return NodePipeline(root_node=root, dino_loader=dino_loader)
