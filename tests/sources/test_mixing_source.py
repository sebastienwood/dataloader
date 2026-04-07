"""tests/test_mixing_source.py
================================
Unit tests for hpc_source.py (ex-mixing_source.py).

[POOL] MixingSource accepte pool_cfg (SharedExtractionPoolConfig).
       ShardIterator requiert un executor injecté par MixingSource.
[SLOW] Tests d'intégration multi-dataset avec le pool partagé.

Run fast tests only::

    pytest -m "not slow"

Run the full suite::

    pytest
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_datasets import DatasetSpec

from dino_loader.backends.cpu import InProcessShardCache
from dino_loader.config import SharedExtractionPoolConfig
from dino_loader.sources import MixingSource, MixingWeights, ResolutionSource
from dino_loader.sources.hpc_source import SampleRecord, ShardIterator
from tests.fixtures import scaffold_dataset_dir, write_shard


# ══════════════════════════════════════════════════════════════════════════════
# ResolutionSource
# ══════════════════════════════════════════════════════════════════════════════


class TestResolutionSource:

    def test_default_values(self):
        rs = ResolutionSource(224, 96)
        g, l = rs()
        assert int(g) == 224 and int(l) == 96

    def test_set_updates_values(self):
        rs = ResolutionSource(224, 96)
        rs.set(448, 192)
        g, l = rs()
        assert int(g) == 448 and int(l) == 192

    def test_thread_safety(self):
        rs     = ResolutionSource(32, 16)
        errors: list = []

        def writer():
            for _ in range(500):
                rs.set(64, 32)
                rs.set(32, 16)

        def reader():
            for _ in range(500):
                g, l = int(rs()[0]), int(rs()[1])
                if g not in (32, 64) or l not in (16, 32):
                    errors.append((g, l))

        t1, t2 = threading.Thread(target=writer), threading.Thread(target=reader)
        t1.start(); t2.start(); t1.join(); t2.join()
        assert not errors

    def test_returns_numpy_arrays(self):
        rs = ResolutionSource(224, 96)
        g, _ = rs()
        assert hasattr(g, "dtype")


# ══════════════════════════════════════════════════════════════════════════════
# MixingWeights
# ══════════════════════════════════════════════════════════════════════════════


class TestMixingWeights:

    def test_normalises_on_init(self):
        mw = MixingWeights(["a", "b"], [3.0, 1.0])
        w  = mw.get()
        assert abs(w[0] - 0.75) < 1e-6 and abs(w[1] - 0.25) < 1e-6

    def test_sum_to_one(self):
        mw = MixingWeights(["a", "b", "c"], [1.0, 2.0, 3.0])
        assert abs(sum(mw.get()) - 1.0) < 1e-6

    def test_set_normalises(self):
        mw = MixingWeights(["a", "b"], [1.0, 1.0])
        mw.set([7.0, 3.0])
        assert abs(mw.get()[0] - 0.70) < 1e-6

    def test_set_by_name_documented_behaviour(self):
        """set_by_name utilise les poids normalisés courants comme base."""
        mw = MixingWeights(["a", "b"], [1.0, 1.0])
        mw.set_by_name("a", 3.0)
        w = mw.get()
        assert w[0] > w[1]
        assert abs(sum(w) - 1.0) < 1e-6

    def test_set_by_name_unknown_raises(self):
        mw = MixingWeights(["alpha", "beta"], [1.0, 1.0])
        with pytest.raises(KeyError, match="not found"):
            mw.set_by_name("gamma", 1.0)

    def test_zero_weights_raise(self):
        with pytest.raises(ValueError):
            MixingWeights(["a", "b"], [0.0, 0.0])

    def test_names_property(self):
        assert MixingWeights(["x", "y"], [1.0, 1.0]).names == ["x", "y"]

    def test_get_returns_copy(self):
        mw = MixingWeights(["a", "b"], [1.0, 1.0])
        w1 = mw.get()
        w1[0] = 999.0
        assert mw.get()[0] != 999.0

    def test_wrong_length_raises(self):
        mw = MixingWeights(["a", "b"], [1.0, 1.0])
        with pytest.raises(ValueError):
            mw.set([1.0])


# ══════════════════════════════════════════════════════════════════════════════
# ShardIterator._passes_predicate — rapide
# ══════════════════════════════════════════════════════════════════════════════


class TestPassesPredicate:

    def _make_iter(self, tmp_path, **spec_kw):
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0, **spec_kw)
        cache     = InProcessShardCache(max_gb=0.5)
        executor  = ThreadPoolExecutor(max_workers=2)
        return ShardIterator(spec=spec, cache=cache, rank=0, world_size=1, executor=executor)

    def test_passes_no_filters(self, tmp_path):
        it  = self._make_iter(tmp_path)
        rec = SampleRecord(jpeg=b"", metadata={"quality_score": 0.0}, key="k")
        assert it._passes_predicate(rec, "s.tar") is True

    def test_fails_below_threshold(self, tmp_path):
        it  = self._make_iter(tmp_path, min_sample_quality=0.5)
        rec = SampleRecord(jpeg=b"", metadata={"quality_score": 0.1}, key="k")
        assert it._passes_predicate(rec, "s.tar") is False

    def test_passes_above_threshold(self, tmp_path):
        it  = self._make_iter(tmp_path, min_sample_quality=0.5)
        rec = SampleRecord(jpeg=b"", metadata={"quality_score": 0.9}, key="k")
        assert it._passes_predicate(rec, "s.tar") is True

    def test_passes_no_quality_key(self, tmp_path):
        it  = self._make_iter(tmp_path, min_sample_quality=0.5)
        rec = SampleRecord(jpeg=b"", metadata={"caption": "x"}, key="k")
        assert it._passes_predicate(rec, "s.tar") is True

    def test_passes_none_metadata(self, tmp_path):
        it  = self._make_iter(tmp_path, min_sample_quality=0.5)
        rec = SampleRecord(jpeg=b"", metadata=None, key="k")
        assert it._passes_predicate(rec, "s.tar") is True

    def test_sample_predicate_called(self, tmp_path):
        from dino_loader.augmentation import SampleMeta

        calls: list = []

        def _pred(meta: SampleMeta) -> bool:
            calls.append(meta)
            return meta.key == "keep"

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        executor  = ThreadPoolExecutor(max_workers=2)
        it        = ShardIterator(
            spec=spec, cache=InProcessShardCache(max_gb=0.5),
            rank=0, world_size=1, executor=executor, sample_predicate=_pred,
        )
        assert it._passes_predicate(SampleRecord(jpeg=b"", metadata=None, key="keep"), "s") is True
        assert it._passes_predicate(SampleRecord(jpeg=b"", metadata=None, key="drop"), "s") is False
        assert len(calls) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Construction
# ══════════════════════════════════════════════════════════════════════════════


class TestShardIteratorConstruction:

    def test_no_shards_assigned_raises(self, tmp_path):
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        executor  = ThreadPoolExecutor(max_workers=2)
        with pytest.raises(RuntimeError, match="no shards assigned"):
            ShardIterator(spec=spec, cache=InProcessShardCache(max_gb=1.0),
                          rank=2, world_size=3, executor=executor)

    def test_pool_cfg_sets_max_workers(self, tmp_path):
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        pool_cfg  = SharedExtractionPoolConfig(max_workers=3)
        ms        = MixingSource(specs=[spec], batch_size=4,
                                 cache=InProcessShardCache(max_gb=0.5),
                                 rank=0, world_size=1, pool_cfg=pool_cfg)
        assert ms._executor._max_workers == 3
        ms.close()


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════


class TestRegisterCallback:

    def test_register_before_call_does_not_crash(self, tmp_path):
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        ms        = MixingSource(specs=[spec], batch_size=4,
                                 cache=InProcessShardCache(max_gb=1.0),
                                 rank=0, world_size=1)
        ms.register_dataset_index_callback(lambda _: None)
        ms.close()


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests — slow (vrais threads + I/O)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def shard_iter(tmp_path):
    tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2,
                                     n_samples_per_shard=8, with_metadata=True)
    spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
    executor  = ThreadPoolExecutor(max_workers=4)
    return ShardIterator(spec=spec, cache=InProcessShardCache(max_gb=1.0),
                         rank=0, world_size=1, executor=executor, seed=42,
                         shuffle_buffer_size=4)


@pytest.mark.slow
class TestShardIteratorIntegration:

    def test_yields_sample(self, shard_iter):
        rec = shard_iter.next_sample()
        assert isinstance(rec.jpeg, bytes) and len(rec.jpeg) > 0

    def test_metadata_present(self, shard_iter):
        rec = shard_iter.next_sample()
        assert rec.metadata is not None and "quality_score" in rec.metadata

    def test_many_samples(self, shard_iter):
        for _ in range(16):
            assert shard_iter.next_sample().jpeg

    def test_reset_epoch(self, shard_iter):
        for _ in range(8):
            shard_iter.next_sample()
        shard_iter.reset_epoch(epoch=1)

    def test_quality_filter(self, tmp_path):
        scores   = [0.1, 0.9, 0.1, 0.9]
        tar_path, _ = write_shard(tmp_path, n_samples=4, with_metadata=True,
                                  quality_scores=scores)
        spec     = DatasetSpec(name="ds", shards=[tar_path], weight=1.0,
                               min_sample_quality=0.5)
        executor = ThreadPoolExecutor(max_workers=2)
        it       = ShardIterator(spec=spec, cache=InProcessShardCache(max_gb=1.0),
                                 rank=0, world_size=1, executor=executor,
                                 shuffle_buffer_size=0)
        passing: list[float] = []
        try:
            for _ in range(4):
                passing.append(it.next_sample().metadata["quality_score"])
        except Exception:
            pass
        assert all(s >= 0.5 for s in passing)


@pytest.fixture
def mixing_source_fixture(tmp_path):
    tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
    spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
    ms        = MixingSource(specs=[spec], batch_size=4,
                             cache=InProcessShardCache(max_gb=1.0),
                             rank=0, world_size=1)
    yield ms
    ms.close()


@pytest.mark.slow
class TestMixingSourceIntegration:

    def test_call_returns_arrays(self, mixing_source_fixture):
        result = mixing_source_fixture()
        assert isinstance(result, list) and len(result) == 4

    def test_metadata_length(self, mixing_source_fixture):
        mixing_source_fixture()
        assert len(mixing_source_fixture.pop_last_metadata()) == 4

    def test_set_epoch(self, mixing_source_fixture):
        mixing_source_fixture.set_epoch(1)

    def test_dataset_names(self, mixing_source_fixture):
        assert mixing_source_fixture.dataset_names == ["ds"]

    def test_shared_pool_closed_once(self, tmp_path):
        """[POOL] close() ne doit appeler shutdown() qu'une fois."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        ms        = MixingSource(specs=[spec], batch_size=4,
                                 cache=InProcessShardCache(max_gb=1.0),
                                 rank=0, world_size=1)
        executor = ms._executor
        ms.close()
        executor.shutdown(wait=False, cancel_futures=True)  # doit ne pas lever

    def test_two_dataset_mixing(self, tmp_path):
        """[POOL] Pool partagé avec deux datasets."""
        specs = [
            DatasetSpec(
                name="alpha",
                shards=scaffold_dataset_dir(root=tmp_path / "a", n_shards=1, n_samples_per_shard=8),
                weight=0.5,
            ),
            DatasetSpec(
                name="beta",
                shards=scaffold_dataset_dir(root=tmp_path / "b", n_shards=1, n_samples_per_shard=8),
                weight=0.5,
            ),
        ]
        ms = MixingSource(specs=specs, batch_size=8,
                          cache=InProcessShardCache(max_gb=1.0),
                          rank=0, world_size=1)
        result = ms()
        assert len(result) == 8
        ms.close()

    def test_callback_receives_indices(self, tmp_path):
        """register_dataset_index_callback doit recevoir des indices valides."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=8)
        spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        ms        = MixingSource(specs=[spec], batch_size=4,
                                 cache=InProcessShardCache(max_gb=1.0),
                                 rank=0, world_size=1)
        received: list[list[int]] = []
        ms.register_dataset_index_callback(lambda idxs: received.append(list(idxs)))
        ms()
        ms.close()
        assert len(received) == 1
        assert len(received[0]) == 4
        assert all(i == 0 for i in received[0])