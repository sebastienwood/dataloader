"""tests/test_pipeline_graph.py
================================
Unit tests for :mod:`dino_loader.pipeline_graph`.

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

_LimitNode
- stops after max_steps

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
- backend and aug_spec properties delegate correctly
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


# ── Fake loader stub ──────────────────────────────────────────────────────────


class _FakeDALINode:
    """Minimal _DALINode stub — provides reset/next/get_state protocol."""

    def __init__(self, batches: list[Batch]) -> None:
        self._batches = batches
        self._idx = 0

    def reset(self, initial_state=None):
        self._idx = 0

    def next(self) -> Batch:
        if self._idx >= len(self._batches):
            raise StopIteration
        b = self._batches[self._idx]
        self._idx += 1
        return b

    def get_state(self) -> dict:
        return {"_num_yielded": self._idx}

    def reset_iter(self):
        pass


class _FakeLoader:
    """Minimal DINODataLoader stub for NodePipeline tests.

    Must expose _dali_node for wrap_loader() to work.
    """

    def __init__(self, batches: list[Batch], steps_per_epoch: int | None = None) -> None:
        self._batches = batches
        self._steps_per_epoch = steps_per_epoch
        self._epoch = 0
        self._step = 0
        self.set_epoch_calls: list[int] = []

        # wrap_loader() reads _dali_node
        self._dali_node = _FakeDALINode(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        if self._steps_per_epoch is None:
            raise TypeError
        return self._steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self.set_epoch_calls.append(epoch)
        self._dali_node.reset()

    def checkpoint(self, step: int) -> None:
        self._step = step

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "step": self._step}

    def load_state_dict(self, sd: dict) -> None:
        self._epoch = sd.get("epoch", 0)
        self._step = sd.get("step", 0)

    def set_weights(self, weights) -> None:
        pass

    def set_weight_by_name(self, name: str, weight: float) -> None:
        pass

    def set_resolution(self, g: int, l: int) -> None:
        pass

    @property
    def current_resolution(self) -> tuple[int, int]:
        return (224, 96)

    @property
    def current_weights(self) -> list[float]:
        return [1.0]

    @property
    def backend(self):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.name = "cpu"
        return m

    @property
    def aug_spec(self):
        from dino_loader.augmentation import DinoV2AugSpec
        from dino_loader.config import DINOAugConfig
        return DinoV2AugSpec(aug_cfg=DINOAugConfig())


def _fake_loader(n: int = 8) -> _FakeLoader:
    batches = [Batch([], [], [{"idx": i}]) for i in range(n)]
    return _FakeLoader(batches, steps_per_epoch=n)


# ══════════════════════════════════════════════════════════════════════════════
# BatchMapNode
# ══════════════════════════════════════════════════════════════════════════════


class TestBatchMapNode:

    def test_applies_fn_to_every_batch(self) -> None:
        from dino_loader.pipeline_graph import BatchMapNode
        seen: list[int] = []

        def _tag(b: Batch) -> Batch:
            seen.append(id(b))
            return b

        batches = [Batch([], [], []) for _ in range(3)]
        src = tn.IterableWrapper(iter(batches))
        node = BatchMapNode(src, _tag)
        node.reset()
        for _ in range(3):
            node.next()
        assert len(seen) == 3

    def test_fn_receives_correct_batch_metadata(self) -> None:
        from dino_loader.pipeline_graph import BatchMapNode
        seen: list[list] = []

        def _inspect(b: Batch) -> Batch:
            seen.append(list(b.metadata))
            return b

        batches = [Batch([], [], [{"i": i}]) for i in range(4)]
        src = tn.IterableWrapper(iter(batches))
        node = BatchMapNode(src, _inspect)
        node.reset()
        for _ in range(4):
            node.next()
        assert [s[0]["i"] for s in seen] == [0, 1, 2, 3]

    def test_get_state_delegates_to_source(self) -> None:
        from dino_loader.pipeline_graph import BatchMapNode
        src = tn.IterableWrapper(iter([Batch([], [], [])]))
        node = BatchMapNode(src, lambda b: b)
        node.reset()
        assert isinstance(node.get_state(), dict)

    def test_label_in_repr(self) -> None:
        from dino_loader.pipeline_graph import BatchMapNode
        src = tn.IterableWrapper(iter([]))
        node = BatchMapNode(src, lambda b: b, label="my_transform")
        assert "my_transform" in repr(node)


# ══════════════════════════════════════════════════════════════════════════════
# BatchFilterNode
# ══════════════════════════════════════════════════════════════════════════════


class TestBatchFilterNode:

    def test_keeps_only_passing_batches(self) -> None:
        from dino_loader.pipeline_graph import BatchFilterNode
        batches = [Batch([], [], [{"score": float(i)}]) for i in range(6)]
        src = tn.IterableWrapper(iter(batches))
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
        from dino_loader.pipeline_graph import BatchFilterNode
        batches = [Batch([], [], [{"ok": i % 2 == 0}]) for i in range(8)]
        src = tn.IterableWrapper(iter(batches))
        node = BatchFilterNode(src, lambda b: b.metadata[0]["ok"])
        node.reset()
        try:
            while True:
                node.next()
        except StopIteration:
            pass
        assert node.n_skipped == 4

    def test_reset_clears_n_skipped(self) -> None:
        from dino_loader.pipeline_graph import BatchFilterNode
        batches = [Batch([], [], [{"ok": False}]) for _ in range(4)] + \
                  [Batch([], [], [{"ok": True}])]
        src = tn.IterableWrapper(iter(batches))
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
        from dino_loader.pipeline_graph import BatchFilterNode
        batches = [Batch([], [], []) for _ in range(3)]
        src = tn.IterableWrapper(iter(batches))
        node = BatchFilterNode(src, lambda b: False)
        node.reset()
        with pytest.raises(StopIteration):
            node.next()


# ══════════════════════════════════════════════════════════════════════════════
# NodePipeline via wrap_loader
# ══════════════════════════════════════════════════════════════════════════════


class TestNodePipelineIteration:

    def test_iterates_all_batches(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        loader = _fake_loader(4)
        loader.set_epoch(0)
        assert len(list(wrap_loader(loader))) == 4

    def test_map_applied_to_every_batch(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        seen: list[int] = []

        def _tag(b: Batch) -> Batch:
            seen.append(id(b))
            return b

        loader = _fake_loader(4)
        loader.set_epoch(0)
        list(wrap_loader(loader).map(_tag))
        assert len(seen) == 4

    def test_select_drops_non_matching(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        loader = _fake_loader(8)
        loader.set_epoch(0)
        pipeline = wrap_loader(loader).select(
            lambda b: b.metadata[0]["idx"] % 2 == 0
        )
        pipeline.set_epoch(0)
        kept = list(pipeline)
        assert all(b.metadata[0]["idx"] % 2 == 0 for b in kept)

    def test_with_epoch_limits_steps(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        loader = _fake_loader(20)
        pipeline = wrap_loader(loader).with_epoch(3)
        pipeline.set_epoch(0)
        batches = list(pipeline)
        assert len(batches) == 3

    def test_len_with_max_steps(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        pipeline = wrap_loader(_fake_loader(20)).with_epoch(7)
        assert len(pipeline) == 7

    def test_chaining_map_and_select(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        mutated: list[dict] = []

        def _mutate(b: Batch) -> Batch:
            b.metadata[0]["mutated"] = True
            mutated.append(b.metadata[0])
            return b

        loader = _fake_loader(8)
        pipeline = wrap_loader(loader).map(_mutate).select(
            lambda b: b.metadata[0]["idx"] % 2 == 0
        )
        pipeline.set_epoch(0)
        kept = list(pipeline)
        assert all(b.metadata[0].get("mutated") for b in kept)

    def test_wrap_loader_raises_without_dali_node(self) -> None:
        """wrap_loader must raise TypeError if _dali_node is absent."""
        from dino_loader.pipeline_graph import wrap_loader

        class _BadLoader:
            pass

        with pytest.raises(TypeError, match="_dali_node"):
            wrap_loader(_BadLoader())


class TestNodePipelineDelegation:

    def test_set_epoch_delegates_to_loader(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        loader = _fake_loader(4)
        wrap_loader(loader).set_epoch(3)
        assert 3 in loader.set_epoch_calls

    def test_state_dict_contains_loader_key(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        loader = _fake_loader(4)
        pipeline = wrap_loader(loader)
        pipeline.set_epoch(0)
        next(iter(pipeline))
        sd = pipeline.state_dict()
        assert "loader" in sd

    def test_state_dict_contains_tn_graph_key(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        loader = _fake_loader(4)
        pipeline = wrap_loader(loader)
        pipeline.set_epoch(0)
        next(iter(pipeline))
        sd = pipeline.state_dict()
        assert "tn_graph" in sd

    def test_load_state_dict_restores_epoch(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        loader = _fake_loader(4)
        wrap_loader(loader).load_state_dict({"loader": {"epoch": 7, "step": 0}})
        assert loader._epoch == 7

    def test_current_resolution_delegation(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        assert wrap_loader(_fake_loader(4)).current_resolution == (224, 96)

    def test_current_weights_delegation(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        assert wrap_loader(_fake_loader(4)).current_weights == [1.0]

    def test_backend_delegation(self) -> None:
        from dino_loader.pipeline_graph import wrap_loader
        assert wrap_loader(_fake_loader(4)).backend.name == "cpu"

    def test_aug_spec_delegation(self) -> None:
        from dino_loader.augmentation import DinoV2AugSpec
        from dino_loader.pipeline_graph import wrap_loader
        assert isinstance(wrap_loader(_fake_loader(4)).aug_spec, DinoV2AugSpec)

    def test_state_dict_max_steps(self) -> None:
        """max_steps is persisted in state_dict [FIX-STATE-MAX-STEPS]."""
        from dino_loader.pipeline_graph import wrap_loader
        loader = _fake_loader(20)
        pipeline = wrap_loader(loader).with_epoch(5)
        pipeline.set_epoch(0)
        next(iter(pipeline))
        sd = pipeline.state_dict()
        assert sd["max_steps"] == 5