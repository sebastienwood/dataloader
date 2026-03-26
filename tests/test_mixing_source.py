"""tests/test_mixing_source.py.

Unit tests for the mixing_source.py layer, independent of DALI.

Tests are split into two categories:
- Fast unit tests (run by default): cover pure-Python classes like
  ``ResolutionSource``, ``MixingWeights``, and lightweight construction paths.
- Slow integration tests (``@pytest.mark.slow``): spin up real ``ShardIterator``
  threads and perform actual shard I/O; skipped in fast CI runs.

Run fast tests only::

    pytest -m "not slow"

Run the full suite::

    pytest

"""

import sys
import threading
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_datasets import DatasetSpec

from dino_loader.backends.cpu import InProcessShardCache
from dino_loader.mixing_source import (
    MixingSource,
    MixingWeights,
    ResolutionSource,
    SampleRecord,
    ShardIterator,
)
from tests.fixtures import scaffold_dataset_dir, write_shard

# ══════════════════════════════════════════════════════════════════════════════
# ResolutionSource  — pure, always fast
# ══════════════════════════════════════════════════════════════════════════════


class TestResolutionSource:

    def test_default_values(self) -> None:
        rs = ResolutionSource(224, 96)
        g, l = rs()
        assert int(g) == 224
        assert int(l) == 96

    def test_set_updates_values(self) -> None:
        rs = ResolutionSource(224, 96)
        rs.set(448, 192)
        g, l = rs()
        assert int(g) == 448
        assert int(l) == 192

    def test_thread_safety(self) -> None:
        """Two threads writing and reading concurrently must not produce torn reads."""
        rs     = ResolutionSource(32, 16)
        errors: list[tuple[int, int]] = []

        def writer() -> None:
            for _ in range(500):
                rs.set(64, 32)
                rs.set(32, 16)

        def reader() -> None:
            for _ in range(500):
                g_int, l_int = int(rs()[0]), int(rs()[1])
                if g_int not in (32, 64) or l_int not in (16, 32):
                    errors.append((g_int, l_int))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == [], f"Torn reads: {errors[:5]}"

    def test_returns_numpy_arrays(self) -> None:
        rs = ResolutionSource(224, 96)
        g, _l = rs()
        assert hasattr(g, "dtype")


# ══════════════════════════════════════════════════════════════════════════════
# MixingWeights  — pure, always fast
# ══════════════════════════════════════════════════════════════════════════════


class TestMixingWeights:
    """MixingWeights(names, raw_weights) — constructor takes two separate args."""

    def test_normalises_on_init(self) -> None:
        mw = MixingWeights(["a", "b"], [3.0, 1.0])
        w  = mw.get()
        assert abs(w[0] - 0.75) < 1e-6
        assert abs(w[1] - 0.25) < 1e-6

    def test_sum_to_one(self) -> None:
        mw = MixingWeights(["a", "b", "c"], [1.0, 2.0, 3.0])
        assert abs(sum(mw.get()) - 1.0) < 1e-6

    def test_set_normalises(self) -> None:
        mw = MixingWeights(["a", "b"], [1.0, 1.0])
        mw.set([7.0, 3.0])
        w = mw.get()
        assert abs(w[0] - 0.70) < 1e-6

    def test_set_by_name(self) -> None:
        mw = MixingWeights(["alpha", "beta"], [1.0, 1.0])
        mw.set_by_name("alpha", 9.0)
        w = mw.get()
        assert w[0] > w[1]

    def test_set_by_name_unknown_raises(self) -> None:
        mw = MixingWeights(["alpha", "beta"], [1.0, 1.0])
        with pytest.raises(KeyError, match="not found"):
            mw.set_by_name("gamma", 1.0)

    def test_zero_weights_raise(self) -> None:
        with pytest.raises(ValueError):
            MixingWeights(["a", "b"], [0.0, 0.0])

    def test_names_property(self) -> None:
        mw = MixingWeights(["x", "y"], [1.0, 1.0])
        assert mw.names == ["x", "y"]

    def test_get_returns_copy(self) -> None:
        """Mutating the list returned by get() must not affect the internal state."""
        mw = MixingWeights(["a", "b"], [1.0, 1.0])
        w1 = mw.get()
        w1[0] = 999.0
        w2 = mw.get()
        assert w2[0] != 999.0, "get() returned a direct reference to internal state"

    def test_wrong_length_raises(self) -> None:
        mw = MixingWeights(["a", "b"], [1.0, 1.0])
        with pytest.raises(ValueError):
            mw.set([1.0])  # wrong length


# ══════════════════════════════════════════════════════════════════════════════
# ShardIterator._passes_predicate — fast (no I/O threads)
# ══════════════════════════════════════════════════════════════════════════════


class TestPassesPredicate:
    """Unit tests for ShardIterator._passes_predicate (called from extraction workers).

    We test it directly without spinning up I/O threads by constructing a
    ShardIterator and calling the private method on SampleRecord stubs.
    """

    def _make_iter(self, tmp_path: Path, **spec_kwargs) -> ShardIterator:
        """Build a ShardIterator with minimal config (no threads started yet)."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0, **spec_kwargs)
        cache = InProcessShardCache(max_gb=0.5)
        return ShardIterator(spec=spec, cache=cache, rank=0, world_size=1)

    def test_passes_when_no_filters(self, tmp_path: Path) -> None:
        it = self._make_iter(tmp_path)
        rec = SampleRecord(jpeg=b"", metadata={"quality_score": 0.0}, key="k")
        assert it._passes_predicate(rec, "shard.tar") is True

    def test_fails_min_quality_below_threshold(self, tmp_path: Path) -> None:
        it = self._make_iter(tmp_path, min_sample_quality=0.5)
        rec = SampleRecord(jpeg=b"", metadata={"quality_score": 0.1}, key="k")
        assert it._passes_predicate(rec, "shard.tar") is False

    def test_passes_min_quality_above_threshold(self, tmp_path: Path) -> None:
        it = self._make_iter(tmp_path, min_sample_quality=0.5)
        rec = SampleRecord(jpeg=b"", metadata={"quality_score": 0.9}, key="k")
        assert it._passes_predicate(rec, "shard.tar") is True

    def test_passes_when_quality_score_missing_from_metadata(self, tmp_path: Path) -> None:
        """When quality_score key is absent, the filter must pass the sample."""
        it = self._make_iter(tmp_path, min_sample_quality=0.5)
        rec = SampleRecord(jpeg=b"", metadata={"caption": "no score here"}, key="k")
        assert it._passes_predicate(rec, "shard.tar") is True

    def test_passes_when_metadata_is_none(self, tmp_path: Path) -> None:
        """None metadata (no sidecar) must pass min_sample_quality filter."""
        it = self._make_iter(tmp_path, min_sample_quality=0.5)
        rec = SampleRecord(jpeg=b"", metadata=None, key="k")
        assert it._passes_predicate(rec, "shard.tar") is True

    def test_sample_predicate_called(self, tmp_path: Path) -> None:
        """SamplePredicate must be invoked and its return value respected."""
        from dino_loader.augmentation import SampleMeta

        calls: list[SampleMeta] = []

        def _pred(meta: SampleMeta) -> bool:
            calls.append(meta)
            return meta.key == "keep"

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        cache = InProcessShardCache(max_gb=0.5)
        it    = ShardIterator(
            spec=spec, cache=cache, rank=0, world_size=1,
            sample_predicate=_pred,
        )

        rec_keep   = SampleRecord(jpeg=b"", metadata=None, key="keep")
        rec_discard = SampleRecord(jpeg=b"", metadata=None, key="discard")

        assert it._passes_predicate(rec_keep,    "shard.tar") is True
        assert it._passes_predicate(rec_discard, "shard.tar") is False
        assert len(calls) == 2
        assert calls[0].key == "keep"
        assert calls[1].key == "discard"

    def test_sample_predicate_receives_shard_path(self, tmp_path: Path) -> None:
        """The SampleMeta passed to the predicate must contain the correct shard_path."""
        from dino_loader.augmentation import SampleMeta

        received: list[str] = []

        def _pred(meta: SampleMeta) -> bool:
            received.append(meta.shard_path)
            return True

        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        cache = InProcessShardCache(max_gb=0.5)
        it    = ShardIterator(
            spec=spec, cache=cache, rank=0, world_size=1,
            sample_predicate=_pred,
        )
        rec = SampleRecord(jpeg=b"", metadata=None, key="k")
        it._passes_predicate(rec, "/lustre/data/shard-000001.tar")

        assert received == ["/lustre/data/shard-000001.tar"]


# ══════════════════════════════════════════════════════════════════════════════
# ShardIterator construction — raises without touching I/O threads
# ══════════════════════════════════════════════════════════════════════════════


class TestShardIteratorConstruction:

    def test_no_shards_assigned_raises(self, tmp_path: Path) -> None:
        """Rank > number of shards → no shards assigned → RuntimeError."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        cache = InProcessShardCache(max_gb=1.0)
        with pytest.raises(RuntimeError, match="no shards assigned"):
            ShardIterator(
                spec       = spec,
                cache      = cache,
                rank       = 2,   # rank 2 with 1 shard → empty assignment
                world_size = 3,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MixingSource.register_dataset_index_callback — fast
# ══════════════════════════════════════════════════════════════════════════════


class TestRegisterDatasetIndexCallback:
    """register_dataset_index_callback must fire on every __call__."""

    @pytest.mark.slow
    def test_callback_receives_indices_per_call(self, tmp_path: Path) -> None:
        """The registered callback receives a list of per-sample dataset indices."""
        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=2, n_samples_per_shard=8,
        )
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        cache = InProcessShardCache(max_gb=1.0)
        ms    = MixingSource(
            specs=[spec], batch_size=4, cache=cache,
            rank=0, world_size=1,
        )

        received: list[list[int]] = []
        ms.register_dataset_index_callback(lambda idxs: received.append(list(idxs)))

        ms()  # one batch
        assert len(received) == 1
        assert len(received[0]) == 4           # one index per sample
        assert all(i == 0 for i in received[0])  # only one dataset

    @pytest.mark.slow
    def test_multiple_callbacks_all_fired(self, tmp_path: Path) -> None:
        """Multiple callbacks registered must each be called."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        cache = InProcessShardCache(max_gb=1.0)
        ms    = MixingSource(
            specs=[spec], batch_size=4, cache=cache,
            rank=0, world_size=1,
        )

        calls_a: list[list[int]] = []
        calls_b: list[list[int]] = []
        ms.register_dataset_index_callback(lambda idxs: calls_a.append(list(idxs)))
        ms.register_dataset_index_callback(lambda idxs: calls_b.append(list(idxs)))

        ms()
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_register_before_first_call_does_not_crash(self, tmp_path: Path) -> None:
        """Registering a callback before any call must not raise."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        cache = InProcessShardCache(max_gb=1.0)
        ms    = MixingSource(
            specs=[spec], batch_size=4, cache=cache,
            rank=0, world_size=1,
        )
        ms.register_dataset_index_callback(lambda _: None)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# ShardIterator integration  — slow (real threads + I/O)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def shard_iter(tmp_path: Path) -> ShardIterator:
    """A ShardIterator backed by two small synthetic shards."""
    tar_paths = scaffold_dataset_dir(
        root=tmp_path,
        n_shards=2,
        n_samples_per_shard=8,
        with_metadata=True,
    )
    spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
    cache = InProcessShardCache(max_gb=1.0)
    return ShardIterator(
        spec                = spec,
        cache               = cache,
        rank                = 0,
        world_size          = 1,
        prefetch_ahead      = 2,
        num_workers         = 1,
        seed                = 42,
        shuffle_buffer_size = 4,
    )


@pytest.mark.slow
class TestShardIteratorIntegration:

    def test_next_sample_returns_record(self, shard_iter: ShardIterator) -> None:
        rec = shard_iter.next_sample()
        assert isinstance(rec, SampleRecord)
        assert isinstance(rec.jpeg, bytes)
        assert len(rec.jpeg) > 0

    def test_next_sample_metadata_present(self, shard_iter: ShardIterator) -> None:
        rec = shard_iter.next_sample()
        assert rec.metadata is not None
        assert "quality_score" in rec.metadata

    def test_many_samples_no_error(self, shard_iter: ShardIterator) -> None:
        """Draw 16 samples (spanning both shards) without error."""
        for _ in range(16):
            rec = shard_iter.next_sample()
            assert rec.jpeg

    def test_reset_epoch_does_not_raise(self, shard_iter: ShardIterator) -> None:
        for _ in range(8):
            shard_iter.next_sample()
        shard_iter.reset_epoch(epoch=1)  # must not raise

    def test_without_metadata_key(self, tmp_path: Path) -> None:
        """metadata_key=None → SampleRecord.metadata is None."""
        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=1, n_samples_per_shard=4, with_metadata=True,
        )
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0, metadata_key=None)
        cache = InProcessShardCache(max_gb=1.0)
        it    = ShardIterator(
            spec=spec, cache=cache, rank=0, world_size=1, shuffle_buffer_size=0,
        )
        rec = it.next_sample()
        assert rec.metadata is None

    def test_quality_filter_drops_low_scores(self, tmp_path: Path) -> None:
        """Samples with quality_score < 0.5 are filtered when min_sample_quality=0.5."""
        scores = [0.1, 0.9, 0.1, 0.9]
        tar_path, _ = write_shard(
            tmp_path, n_samples=4, with_metadata=True, quality_scores=scores,
        )
        spec  = DatasetSpec(
            name="ds", shards=[tar_path], weight=1.0,
            min_sample_quality=0.5, metadata_key="json",
        )
        cache = InProcessShardCache(max_gb=1.0)
        it    = ShardIterator(
            spec=spec, cache=cache, rank=0, world_size=1, shuffle_buffer_size=0,
        )
        passing: list[float] = []
        try:
            for _ in range(4):
                rec = it.next_sample()
                passing.append(rec.metadata["quality_score"])
        except Exception:
            pass
        for score in passing:
            assert score >= 0.5, f"Score {score} should have been filtered"

    def test_weighted_shard_sampling(self, tmp_path: Path) -> None:
        """shard_quality_scores biases shard selection without raising."""
        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=3, n_samples_per_shard=4,
        )
        spec  = DatasetSpec(
            name="ds",
            shards=tar_paths,
            weight=1.0,
            shard_quality_scores=[0.01, 0.01, 100.0],
        )
        cache = InProcessShardCache(max_gb=1.0)
        it    = ShardIterator(
            spec=spec, cache=cache, rank=0, world_size=1, shuffle_buffer_size=0,
        )
        rec = it.next_sample()
        assert rec.jpeg


# ══════════════════════════════════════════════════════════════════════════════
# MixingSource integration  — slow (real threads + I/O)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mixing_source(tmp_path: Path) -> MixingSource:
    """A MixingSource backed by 2 shards of 8 samples each."""
    tar_paths = scaffold_dataset_dir(
        root=tmp_path, n_shards=2, n_samples_per_shard=8,
    )
    spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
    cache = InProcessShardCache(max_gb=1.0)
    return MixingSource(
        specs               = [spec],
        batch_size          = 4,
        cache               = cache,
        rank                = 0,
        world_size          = 1,
        num_workers         = 1,
        shuffle_buffer_size = 4,
    )


@pytest.mark.slow
class TestMixingSourceIntegration:

    def test_call_returns_list_of_arrays(self, mixing_source: MixingSource) -> None:
        import numpy as np
        result = mixing_source()
        assert isinstance(result, list)
        assert len(result) == 4
        for arr in result:
            assert hasattr(arr, "dtype")

    def test_pop_last_metadata_length(self, mixing_source: MixingSource) -> None:
        mixing_source()
        meta = mixing_source.pop_last_metadata()
        assert len(meta) == 4

    def test_set_epoch_does_not_raise(self, mixing_source: MixingSource) -> None:
        mixing_source.set_epoch(1)

    def test_set_weights_normalises(self, tmp_path: Path) -> None:
        specs = [
            DatasetSpec(
                name="a",
                shards=scaffold_dataset_dir(root=tmp_path / "a"),
                weight=1.0,
            ),
            DatasetSpec(
                name="b",
                shards=scaffold_dataset_dir(root=tmp_path / "b"),
                weight=1.0,
            ),
        ]
        cache = InProcessShardCache(max_gb=1.0)
        ms    = MixingSource(specs=specs, batch_size=4, cache=cache, rank=0, world_size=1)
        ms.set_weights([3.0, 1.0])
        w = ms.current_weights
        assert abs(w[0] - 0.75) < 1e-5

    def test_dataset_names(self, mixing_source: MixingSource) -> None:
        assert mixing_source.dataset_names == ["ds"]

    def test_two_dataset_mixing(self, tmp_path: Path) -> None:
        specs = [
            DatasetSpec(
                name="alpha",
                shards=scaffold_dataset_dir(
                    root=tmp_path / "alpha", n_shards=1, n_samples_per_shard=8,
                ),
                weight=0.5,
            ),
            DatasetSpec(
                name="beta",
                shards=scaffold_dataset_dir(
                    root=tmp_path / "beta", n_shards=1, n_samples_per_shard=8,
                ),
                weight=0.5,
            ),
        ]
        cache = InProcessShardCache(max_gb=1.0)
        ms    = MixingSource(
            specs=specs, batch_size=8, cache=cache, rank=0, world_size=1,
        )
        result = ms()
        assert len(result) == 8
