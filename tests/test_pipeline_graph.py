"""tests/test_pipeline_graph.py
================================
Unit tests for :mod:`dino_loader.pipeline_graph` (Phase 3 — composable
stateful post-processing pipeline built on torchdata.nodes).

Note: ``_DALINode`` lives in ``dino_loader.dali_node`` and is re-exported
from ``pipeline_graph`` for backward compatibility.

Coverage
--------
BatchMapNode
- applies fn to every batch
- fn receives correct batch object
- get_state delegates to source

BatchFilterNode
- keeps only batches passing the predicate
- n_skipped counter tracks rejected batches
- reset clears n_skipped
- all-rejected stream raises StopIteration

NodePipeline (via wrap_loader)
- iterable, yields all batches
- .map() applies to every batch
- .select() drops non-matching batches
- .with_epoch() limits steps per epoch
- len() returns max_steps when set
- chaining map + select works correctly
- set_epoch delegates to underlying loader
- state_dict contains 'loader' key
- load_state_dict restores epoch
- current_resolution delegates to underlying loader
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torchdata.nodes as tn

from dino_loader.memory import Batch
from dino_loader.pipeline_graph import (
    BatchFilterNode,
    BatchMapNode,
    NodePipeline,
    wrap_loader,
)


# ── Fake loader stub ──────────────────────────────────────────────────────────


class _FakeLoader:
    """Minimal DINODataLoader stub for NodePipeline tests."""

    def __init__(self, batches: list[Batch], steps_per_epoch: int | None = None) -> None:
        self._batches        = batches
        self._steps_per_epoch = steps_per_epoch
        self._epoch          = 0
        self._step           = 0
        self.set_epoch_calls: list[int] = []

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        if self._steps_per_epoch is None:
            raise TypeError
        return self._steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self.set_epoch_calls.append(epoch)

    def checkpoint(self, step: int) -> None:
        self._step = step

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "step": self._step}

    def load_state_dict(self, sd: dict) -> None:
        self._epoch = sd.get("epoch", 0)
        self._step  = sd.get("step", 0)

    def set_weights(self, weights) -> None:
        pass

    def set_weight_by_name(self, name: str, weight: float) -> None:
        pass

    def set_resolution(self, g: int, l: int) -> None:
        pass

    @property
    def current_resolution(self) -> tuple[int, int]:
        return (224, 96)


def _fake_loader(n: int = 8) -> _FakeLoader:
    batches = [Batch([], [], [{"idx": i}]) for i in range(n)]
    return _FakeLoader(batches, steps_per_epoch=n)


# ══════════════════════════════════════════════════════════════════════════════
# BatchMapNode
# ══════════════════════════════════════════════════════════════════════════════


class TestBatchMapNode:

    def test_applies_fn_to_every_batch(self) -> None:
        seen: list[int] = []

        def _tag(b: Batch) -> Batch:
            seen.append(id(b))
            return b

        batches = [Batch([], [], []) for _ in range(3)]
        src  = tn.IterableWrapper(iter(batches))
        node = BatchMapNode(src, _tag)
        node.reset()
        for _ in range(3):
            node.next()
        assert len(seen) == 3

    def test_fn_receives_correct_batch_metadata(self) -> None:
        seen: list[list] = []

        def _inspect(b: Batch) -> Batch:
            seen.append(list(b.metadata))
            return b

        batches = [Batch([], [], [{"i": i}]) for i in range(4)]
        src  = tn.IterableWrapper(iter(batches))
        node = BatchMapNode(src, _inspect)
        node.reset()
        for _ in range(4):
            node.next()
        assert [s[0]["i"] for s in seen] == [0, 1, 2, 3]

    def test_get_state_delegates_to_source(self) -> None:
        src  = tn.IterableWrapper(iter([Batch([], [], [])]))
        node = BatchMapNode(src, lambda b: b)
        node.reset()
        assert isinstance(node.get_state(), dict)


# ══════════════════════════════════════════════════════════════════════════════
# BatchFilterNode
# ══════════════════════════════════════════════════════════════════════════════


class TestBatchFilterNode:

    def test_keeps_only_passing_batches(self) -> None:
        batches = [Batch([], [], [{"score": float(i)}]) for i in range(6)]
        src  = tn.IterableWrapper(iter(batches))
        node = BatchFilterNode(src, lambda b: b.metadata[0]["score"] >= 3.0)
        node.reset()
        kept: list[Batch] = []
        try:
            while True:
                kept.append(node.next())
        except StopIteration:
            pass
        assert len(kept) == 3
        assert all(b.metadata[0]["score"] >= 3.0 for b in kept)

    def test_n_skipped_counts_rejected(self) -> None:
        batches = [Batch([], [], [{"ok": i % 2 == 0}]) for i in range(8)]
        src  = tn.IterableWrapper(iter(batches))
        node = BatchFilterNode(src, lambda b: b.metadata[0]["ok"])
        node.reset()
        try:
            while True:
                node.next()
        except StopIteration:
            pass
        assert node.n_skipped == 4

    def test_reset_clears_n_skipped(self) -> None:
        batches = [Batch([], [], [{"ok": False}]) for _ in range(4)] + \
                  [Batch([], [], [{"ok": True}])]
        src  = tn.IterableWrapper(iter(batches))
        node = BatchFilterNode(src, lambda b: b.metadata[0]["ok"])
        node.reset()
        try:
            while True:
                node.next()
        except StopIteration:
            pass
        assert node.n_skipped == 4
        node.reset()
        assert node.n_skipped == 0

    def test_all_rejected_raises_stop_iteration(self) -> None:
        batches = [Batch([], [], []) for _ in range(3)]
        src  = tn.IterableWrapper(iter(batches))
        node = BatchFilterNode(src, lambda b: False)
        node.reset()
        with pytest.raises(StopIteration):
            node.next()


# ══════════════════════════════════════════════════════════════════════════════
# NodePipeline via wrap_loader
# ══════════════════════════════════════════════════════════════════════════════


class TestNodePipelineIteration:

    def test_iterates_all_batches(self) -> None:
        assert len(list(wrap_loader(_fake_loader(4)))) == 4

    def test_map_applied_to_every_batch(self) -> None:
        seen: list[int] = []

        def _tag(b: Batch) -> Batch:
            seen.append(id(b))
            return b

        list(wrap_loader(_fake_loader(4)).map(_tag))
        assert len(seen) == 4

    def test_select_drops_non_matching(self) -> None:
        pipeline = (
            wrap_loader(_fake_loader(8))
            .select(lambda b: b.metadata[0]["idx"] % 2 == 0)
        )
        kept = list(pipeline)
        assert all(b.metadata[0]["idx"] % 2 == 0 for b in kept)

    def test_with_epoch_limits_steps(self) -> None:
        batches = list(wrap_loader(_fake_loader(20)).with_epoch(3))
        assert len(batches) == 3

    def test_len_with_max_steps(self) -> None:
        pipeline = wrap_loader(_fake_loader(20)).with_epoch(7)
        assert len(pipeline) == 7

    def test_chaining_map_and_select(self) -> None:
        mutated: list[dict] = []

        def _mutate(b: Batch) -> Batch:
            b.metadata[0]["mutated"] = True
            mutated.append(b.metadata[0])
            return b

        kept = list(
            wrap_loader(_fake_loader(8))
            .map(_mutate)
            .select(lambda b: b.metadata[0]["idx"] % 2 == 0),
        )
        assert all(b.metadata[0].get("mutated") for b in kept)


class TestNodePipelineDelegation:

    def test_set_epoch_delegates_to_loader(self) -> None:
        loader = _fake_loader(4)
        wrap_loader(loader).set_epoch(3)
        assert loader.set_epoch_calls == [3]

    def test_state_dict_contains_loader_key(self) -> None:
        pipeline = wrap_loader(_fake_loader(4))
        next(iter(pipeline))
        sd = pipeline.state_dict()
        assert "loader" in sd

    def test_load_state_dict_restores_epoch(self) -> None:
        loader = _fake_loader(4)
        wrap_loader(loader).load_state_dict({"loader": {"epoch": 7, "step": 0}})
        assert loader._epoch == 7

    def test_current_resolution_delegation(self) -> None:
        assert wrap_loader(_fake_loader(4)).current_resolution == (224, 96)