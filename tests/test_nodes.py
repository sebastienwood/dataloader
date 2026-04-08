"""tests/test_nodes.py
======================
Unit tests for the shard reader and pipeline graph nodes.

Imports from the canonical modules:
- :mod:`dino_loader.shard_reader`  (ShardReaderNode, build_reader_graph)
- :mod:`dino_loader.pipeline_graph` (MetadataNode, MaskMapNode)

The ``dino_loader.nodes`` shim is not used here — tests import directly from
the source modules as per CONVENTIONS.md.

Tests that spin up real ``MixingSource`` threads and perform shard I/O are
decorated with ``@pytest.mark.slow`` and excluded from the default fast run::

    pytest -m "not slow"   # CI default
    pytest                 # full suite

No GPU, DALI, or SLURM required — all shard I/O uses InProcessShardCache.

Coverage
--------
ShardReaderNode (fast, state-only)
- dataset_names available before reset()
- set_epoch before reset() does not crash

ShardReaderNode (slow, real I/O)
- next() returns (jpeg_list, metadata_list)
- get_state() contains epoch, mixing_weights, dataset_names
- set_epoch updates state
- set_weights normalises correctly
- reset with saved state restores epoch
- multiple batches without error

MetadataNode (slow)
- passes through jpegs and metadata
- pop_last_metadata clears after first call
- get_state delegates to source

build_reader_graph (slow)
- returns (loader, reader_node)
- loader is iterable
- loader state_dict round-trip

MaskMapNode
- as_transform applies mask to batch
- mask has correct shape and dtype
- exact count guarantee preserved after transform
- global_crops unchanged after masking
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
from dino_datasets import DatasetSpec

from dino_loader.backends.cpu import InProcessShardCache
from dino_loader.pipeline_graph import MaskMapNode, MetadataNode
from dino_loader.shard_reader import ShardReaderNode, build_reader_graph
from tests.fixtures import scaffold_dataset_dir


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_spec(tar_paths: list[str], name: str = "ds") -> DatasetSpec:
    return DatasetSpec(name=name, shards=tuple(tar_paths), weight=1.0)


def _cache() -> InProcessShardCache:
    return InProcessShardCache(max_gb=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# ShardReaderNode — state (fast, no I/O)
# ══════════════════════════════════════════════════════════════════════════════


class TestShardReaderNodeState:

    def test_dataset_names_before_reset(self, tmp_path: Path) -> None:
        """dataset_names is available before reset() (from spec list)."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths, "myds")], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        assert node.dataset_names == ["myds"]

    def test_set_epoch_before_reset(self, tmp_path: Path) -> None:
        """set_epoch before reset() should not crash (no source yet)."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node.set_epoch(5)  # must not raise

    def test_current_weights_before_reset_empty(self, tmp_path: Path) -> None:
        """current_weights returns [] before reset() (source not yet created)."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        assert node.current_weights == []

    def test_get_state_before_reset(self, tmp_path: Path) -> None:
        """get_state() works before reset() with empty weights."""
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=1)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        state = node.get_state()
        assert "epoch" in state
        assert "mixing_weights" in state
        assert "dataset_names" in state


# ══════════════════════════════════════════════════════════════════════════════
# ShardReaderNode — integration (slow)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestShardReaderNodeOutput:

    def test_yields_jpeg_and_metadata(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=2, n_samples_per_shard=8,
        )
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        jpegs, meta = node.next()
        assert len(jpegs) == 4
        assert len(meta) == 4
        for j in jpegs:
            assert isinstance(j, np.ndarray)

    def test_multiple_batches_no_error(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=4, n_samples_per_shard=16,
        )
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=8,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        for _ in range(5):
            jpegs, meta = node.next()
            assert len(jpegs) == 8
            assert len(meta) == 8

    def test_get_state_contains_epoch_weights_names(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        state = node.get_state()
        assert "epoch" in state
        assert "mixing_weights" in state
        assert "dataset_names" in state
        assert state["epoch"] == 0

    def test_set_epoch_updates_state(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        node.set_epoch(5)
        assert node.get_state()["epoch"] == 5

    def test_reset_with_state_restores_epoch(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        node = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        node.set_epoch(3)
        saved = node.get_state()

        node2 = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node2.reset(initial_state=saved)
        assert node2._epoch == 3

    def test_set_weights_normalised(self, tmp_path: Path) -> None:
        paths_a = scaffold_dataset_dir(root=tmp_path / "a", n_shards=1)
        paths_b = scaffold_dataset_dir(root=tmp_path / "b", n_shards=1)
        node = ShardReaderNode(
            specs=[_make_spec(paths_a, "a"), _make_spec(paths_b, "b")],
            batch_size=4, cache=_cache(), rank=0, world_size=1,
        )
        node.reset()
        node.set_weights([3.0, 1.0])
        w = node.current_weights
        assert abs(w[0] - 0.75) < 1e-5
        assert abs(w[1] - 0.25) < 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# MetadataNode — integration (slow)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestMetadataNode:

    def test_passes_through_jpegs_and_meta(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()
        jpegs, meta = node.next()
        assert len(jpegs) == 4
        assert len(meta) == 4

    def test_pop_last_metadata_clears_buffer(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()
        node.next()
        meta1 = node.pop_last_metadata()
        meta2 = node.pop_last_metadata()
        assert len(meta1) == 4
        assert meta2 == []

    def test_get_state_delegates_to_reader(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        reader = ShardReaderNode(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        node = MetadataNode(reader)
        node.reset()
        assert "epoch" in node.get_state()


# ══════════════════════════════════════════════════════════════════════════════
# build_reader_graph — integration (slow)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestBuildReaderGraph:

    def test_returns_loader_and_reader(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        loader, reader = build_reader_graph(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        assert loader is not None
        assert reader is not None
        assert isinstance(reader, ShardReaderNode)

    def test_loader_is_iterable(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        loader, reader = build_reader_graph(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        reader.set_epoch(0)
        jpegs, meta = next(iter(loader))
        assert len(jpegs) == 4
        assert len(meta) == 4

    def test_loader_state_dict_roundtrip(self, tmp_path: Path) -> None:
        tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2)
        loader, reader = build_reader_graph(
            specs=[_make_spec(tar_paths)], batch_size=4,
            cache=_cache(), rank=0, world_size=1,
        )
        reader.set_epoch(2)
        sd = loader.state_dict()
        assert isinstance(sd, dict)
        loader.load_state_dict(sd)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# MaskMapNode — fast (uses a stub Batch, no I/O)
# ══════════════════════════════════════════════════════════════════════════════


class TestMaskMapNodeTransform:
    """Tests for MaskMapNode.as_transform() — no shard I/O required."""

    def test_as_transform_adds_masks(self) -> None:
        import torch
        from dino_loader.masking import MaskingGenerator
        from dino_loader.memory import Batch

        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        fn = MaskMapNode.as_transform(gen)
        batch = Batch(
            global_crops=[torch.zeros(2, 3, 224, 224)],
            local_crops=[],
            metadata=[None, None],
        )
        out = fn(batch)
        assert out.masks is not None

    def test_masks_shape_matches_batch(self) -> None:
        import torch
        from dino_loader.masking import MaskingGenerator
        from dino_loader.memory import Batch

        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        fn = MaskMapNode.as_transform(gen)
        batch_b = 4
        batch = Batch(
            global_crops=[torch.zeros(batch_b, 3, 224, 224)],
            local_crops=[],
            metadata=[None] * batch_b,
        )
        out = fn(batch)
        # masks should be (B, H*W)
        assert out.masks.shape == (batch_b, 14 * 14)

    def test_masks_dtype_is_bool(self) -> None:
        import torch
        from dino_loader.masking import MaskingGenerator
        from dino_loader.memory import Batch

        gen = MaskingGenerator(input_size=(8, 8), num_masking_patches=20)
        fn = MaskMapNode.as_transform(gen)
        batch = Batch(
            global_crops=[torch.zeros(2, 3, 64, 64)],
            local_crops=[],
            metadata=[None, None],
        )
        out = fn(batch)
        assert out.masks.dtype == torch.bool

    def test_masks_exact_count(self) -> None:
        """MaskingGenerator guarantee is preserved end-to-end."""
        import torch
        from dino_loader.masking import MaskingGenerator
        from dino_loader.memory import Batch

        n_mask = 30
        gen = MaskingGenerator(input_size=(8, 8), num_masking_patches=n_mask)
        fn = MaskMapNode.as_transform(gen, num_masking_patches=n_mask)
        batch = Batch(
            global_crops=[torch.zeros(1, 3, 64, 64)],
            local_crops=[],
            metadata=[None],
        )
        for _ in range(10):
            out = fn(batch)
            assert int(out.masks[0].sum().item()) == n_mask

    def test_global_crops_unchanged(self) -> None:
        """Applying the mask transform must not mutate crop tensors."""
        import torch
        from dino_loader.masking import MaskingGenerator
        from dino_loader.memory import Batch

        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        fn = MaskMapNode.as_transform(gen)
        t = torch.ones(2, 3, 224, 224)
        batch = Batch(global_crops=[t], local_crops=[], metadata=[None, None])
        out = fn(batch)
        assert torch.equal(out.global_crops[0], t)

    def test_as_transform_raises_on_empty_global_crops(self) -> None:
        """as_transform must raise ValueError when global_crops is empty."""
        from dino_loader.masking import MaskingGenerator
        from dino_loader.memory import Batch

        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        fn = MaskMapNode.as_transform(gen)
        batch = Batch(global_crops=[], local_crops=[], metadata=[])
        with pytest.raises(ValueError, match="global_crops"):
            fn(batch)