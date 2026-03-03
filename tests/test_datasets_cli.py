"""
tests/test_datasets_cli.py
==========================
Tests for the datasets CLI (preview, count, add, stubs) and the Dataset
discovery/resolution logic.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests.fixtures import scaffold_dataset_dir, write_shard
from dino_loader.datasets.cli import (
    add_dataset,
    count_elements,
    preview_datasets,
)
from dino_loader.datasets.dataset import Dataset, GlobalDatasetFilter
from dino_loader.datasets.stub_gen import generate_stubs


# ══════════════════════════════════════════════════════════════════════════════
# Dataset.resolve()
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetResolve:

    def test_resolve_finds_shards(self, tmp_path):
        scaffold_dataset_dir(
            root=tmp_path, conf="public", modality="rgb",
            name="imagenet", split="train", n_shards=3,
        )
        ds     = Dataset("imagenet", root_path=str(tmp_path))
        shards = ds.resolve()
        assert len(shards) == 3

    def test_resolve_respects_allowed_splits(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", split="train",   n_shards=2)
        scaffold_dataset_dir(root=tmp_path, name="imagenet", split="val",     n_shards=1)
        ds     = Dataset("imagenet", root_path=str(tmp_path))
        shards = ds.resolve(global_filter=GlobalDatasetFilter(allowed_splits=["train"]))
        assert len(shards) == 2

    def test_resolve_unknown_dataset_returns_empty(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=2)
        ds = Dataset("nonexistent", root_path=str(tmp_path))
        assert ds.resolve() == []

    def test_resolve_nonexistent_root_returns_empty(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ds     = Dataset("ds", root_path=str(tmp_path / "no_such_dir"))
            shards = ds.resolve()
        assert shards == []

    def test_to_spec_returns_dataset_spec(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=2)
        ds   = Dataset("imagenet", root_path=str(tmp_path))
        spec = ds.to_spec()
        assert spec is not None
        assert spec.name == "imagenet"
        assert len(spec.shards) == 2

    def test_to_spec_returns_none_if_no_shards(self, tmp_path):
        ds   = Dataset("ghost", root_path=str(tmp_path))
        spec = ds.to_spec()
        assert spec is None

    def test_to_spec_uses_config_weight(self, tmp_path):
        from dino_loader.datasets.dataset import DatasetConfig
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=1)
        ds   = Dataset("imagenet", root_path=str(tmp_path))
        spec = ds.to_spec(config=DatasetConfig(weight=0.42))
        assert abs(spec.weight - 0.42) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# count_elements
# ══════════════════════════════════════════════════════════════════════════════

class TestCountElements:

    def test_counts_from_idx(self, tmp_path, capsys):
        """count_elements reads binary .idx file and reports item count."""
        n_samples = 12
        scaffold_dataset_dir(
            root=tmp_path, name="myds",
            n_shards=1, n_samples_per_shard=n_samples,
        )
        count_elements("myds", root_path=str(tmp_path))
        captured = capsys.readouterr()
        # The idx file has n_samples * 8 bytes → n_samples entries
        assert str(n_samples) in captured.out

    def test_no_valid_shards(self, tmp_path, capsys):
        count_elements("ghost_dataset", root_path=str(tmp_path))
        captured = capsys.readouterr()
        assert "No valid shards" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# add_dataset
# ══════════════════════════════════════════════════════════════════════════════

class TestAddDataset:

    def test_creates_directory(self, tmp_path):
        add_dataset("private", "rgb", "my_new_ds", "train", root_path=str(tmp_path))
        expected = tmp_path / "private" / "rgb" / "my_new_ds" / "train"
        assert expected.is_dir()

    def test_idempotent(self, tmp_path):
        """Calling add twice does not raise."""
        add_dataset("public", "rgb", "ds", "train", root_path=str(tmp_path))
        add_dataset("public", "rgb", "ds", "train", root_path=str(tmp_path))


# ══════════════════════════════════════════════════════════════════════════════
# preview_datasets
# ══════════════════════════════════════════════════════════════════════════════

class TestPreviewDatasets:

    def test_preview_no_crash(self, tmp_path, capsys):
        scaffold_dataset_dir(root=tmp_path, name="imagenet", n_shards=1)
        preview_datasets(root_path=str(tmp_path))
        captured = capsys.readouterr()
        assert "imagenet" in captured.out

    def test_preview_nonexistent_root(self, tmp_path, capsys):
        preview_datasets(root_path=str(tmp_path / "no_such"))
        captured = capsys.readouterr()
        assert "Error" in captured.out or "does not exist" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# generate_stubs
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateStubs:

    def test_generates_hub_py(self, tmp_path):
        scaffold_dataset_dir(
            root=tmp_path, conf="public", modality="rgb",
            name="imagenet", split="train", n_shards=1,
        )
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        assert Path(out_file).exists()
        content = Path(out_file).read_text()
        assert "imagenet" in content
        assert "Dataset" in content

    def test_stub_has_do_not_edit_comment(self, tmp_path):
        scaffold_dataset_dir(root=tmp_path, name="ds", n_shards=1)
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        content = Path(out_file).read_text()
        assert "do not edit" in content.lower()

    def test_empty_root_produces_valid_file(self, tmp_path):
        (tmp_path / "datasets_root").mkdir()
        out_file = str(tmp_path / "hub.py")
        generate_stubs(
            root_path=str(tmp_path / "datasets_root"),
            output_file=out_file,
        )
        assert Path(out_file).exists()

    def test_stubs_for_multiple_datasets(self, tmp_path):
        for ds in ("laion", "imagenet", "custom"):
            scaffold_dataset_dir(root=tmp_path, name=ds, n_shards=1)
        out_file = str(tmp_path / "hub.py")
        generate_stubs(root_path=str(tmp_path), output_file=out_file)
        content = Path(out_file).read_text()
        for ds in ("laion", "imagenet", "custom"):
            assert ds in content
