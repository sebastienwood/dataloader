"""
tests/test_config.py
====================
Unit tests for dino_loader.config dataclasses.

These have no external dependencies beyond the stdlib.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.config import (
    CheckpointState,
    DatasetSpec,
    DINOAugConfig,
    LoaderConfig,
)


# ══════════════════════════════════════════════════════════════════════════════
# DatasetSpec
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetSpec:

    def test_basic_construction(self):
        spec = DatasetSpec(name="laion", shards=["/path/a.tar", "/path/b.tar"], weight=0.5)
        assert spec.name == "laion"
        assert len(spec.shards) == 2
        assert spec.weight == 0.5

    def test_empty_shards_raises(self):
        with pytest.raises(ValueError, match="no shards"):
            DatasetSpec(name="empty", shards=[], weight=1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight"):
            DatasetSpec(name="ds", shards=["x.tar"], weight=-1.0)

    def test_shard_quality_scores_length_mismatch(self):
        with pytest.raises(ValueError, match="shard_quality_scores"):
            DatasetSpec(
                name                 = "ds",
                shards               = ["a.tar", "b.tar"],
                shard_quality_scores = [0.5],    # wrong length
            )

    def test_negative_quality_score_raises(self):
        with pytest.raises(ValueError, match="shard_quality_scores"):
            DatasetSpec(
                name                 = "ds",
                shards               = ["a.tar"],
                shard_quality_scores = [-0.1],
            )

    def test_min_sample_quality_out_of_range(self):
        with pytest.raises(ValueError, match="min_sample_quality"):
            DatasetSpec(name="ds", shards=["x.tar"], min_sample_quality=1.5)

    def test_min_sample_quality_zero_is_valid(self):
        spec = DatasetSpec(name="ds", shards=["x.tar"], min_sample_quality=0.0)
        assert spec.min_sample_quality == 0.0

    def test_default_metadata_key_is_json(self):
        spec = DatasetSpec(name="ds", shards=["x.tar"])
        assert spec.metadata_key == "json"

    def test_mean_std_per_dataset(self):
        spec = DatasetSpec(
            name  = "ds",
            shards= ["x.tar"],
            mean  = (0.4, 0.3, 0.2),
            std   = (0.1, 0.1, 0.1),
        )
        assert spec.mean == (0.4, 0.3, 0.2)

    def test_shard_quality_scores_valid(self):
        spec = DatasetSpec(
            name                 = "ds",
            shards               = ["a.tar", "b.tar"],
            shard_quality_scores = [0.8, 0.3],
        )
        assert spec.shard_quality_scores[0] == 0.8


# ══════════════════════════════════════════════════════════════════════════════
# DINOAugConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestDINOAugConfig:

    def test_defaults(self):
        cfg = DINOAugConfig()
        assert cfg.global_crop_size == 224
        assert cfg.local_crop_size  == 96
        assert cfg.n_global_crops   == 2
        assert cfg.n_local_crops    == 8

    def test_n_views(self):
        cfg = DINOAugConfig(n_global_crops=2, n_local_crops=6)
        assert cfg.n_views == 8

    def test_max_crop_sizes_default_to_crop_sizes(self):
        cfg = DINOAugConfig(global_crop_size=224, local_crop_size=96)
        assert cfg.max_global_crop_size == 224
        assert cfg.max_local_crop_size  == 96

    def test_max_crop_sizes_explicit(self):
        cfg = DINOAugConfig(global_crop_size=224, max_global_crop_size=518)
        assert cfg.max_global_crop_size == 518

    def test_resolution_schedule_sorted(self):
        cfg = DINOAugConfig(
            resolution_schedule = [(30, 518), (0, 224), (10, 448)]
        )
        epochs = [e for e, _ in cfg.resolution_schedule]
        assert epochs == sorted(epochs)

    def test_crop_size_at_epoch_no_schedule(self):
        cfg = DINOAugConfig(global_crop_size=224)
        assert cfg.crop_size_at_epoch(0)  == 224
        assert cfg.crop_size_at_epoch(50) == 224

    def test_crop_size_at_epoch_with_schedule(self):
        cfg = DINOAugConfig(
            global_crop_size    = 224,
            max_global_crop_size= 518,
            resolution_schedule = [(0, 224), (10, 448), (30, 518)],
        )
        assert cfg.crop_size_at_epoch(0)  == 224
        assert cfg.crop_size_at_epoch(9)  == 224
        assert cfg.crop_size_at_epoch(10) == 448
        assert cfg.crop_size_at_epoch(29) == 448
        assert cfg.crop_size_at_epoch(30) == 518
        assert cfg.crop_size_at_epoch(99) == 518

    def test_invalid_resolution_schedule_entry(self):
        with pytest.raises(ValueError):
            DINOAugConfig(resolution_schedule=[(-1, 224)])


# ══════════════════════════════════════════════════════════════════════════════
# LoaderConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderConfig:

    def test_defaults(self):
        cfg = LoaderConfig()
        assert cfg.node_shm_gb           == 128.0
        assert cfg.shuffle_buffer_size   == 512
        assert cfg.stateful_dataloader   is True
        assert cfg.output_dtype          == "bf16"

    def test_invalid_output_dtype(self):
        with pytest.raises(ValueError, match="output_dtype"):
            LoaderConfig(output_dtype="int8")

    def test_valid_dtypes(self):
        LoaderConfig(output_dtype="bf16")
        LoaderConfig(output_dtype="fp32")

    def test_shm_warn_threshold_range(self):
        with pytest.raises(ValueError):
            LoaderConfig(shm_warn_threshold=1.5)
        with pytest.raises(ValueError):
            LoaderConfig(shm_warn_threshold=-0.1)

    def test_hw_decoder_load_range(self):
        with pytest.raises(ValueError):
            LoaderConfig(hw_decoder_load=1.5)

    def test_seed_field(self):
        cfg = LoaderConfig(seed=1234)
        assert cfg.seed == 1234


# ══════════════════════════════════════════════════════════════════════════════
# CheckpointState
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointState:

    def _make_state(self, step=100, epoch=2):
        return CheckpointState(
            step            = step,
            epoch           = epoch,
            dataset_names   = ["laion", "imagenet"],
            mixing_weights  = [0.7, 0.3],
            global_crop_size= 224,
            local_crop_size = 96,
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        state  = self._make_state()
        path   = tmp_path / "dl_state_000000100.json"
        state.save(path)

        loaded = CheckpointState.load(path)
        assert loaded.step             == state.step
        assert loaded.epoch            == state.epoch
        assert loaded.dataset_names    == state.dataset_names
        assert loaded.mixing_weights   == state.mixing_weights
        assert loaded.global_crop_size == 224
        assert loaded.local_crop_size  == 96

    def test_saved_file_is_valid_json(self, tmp_path):
        state = self._make_state()
        path  = tmp_path / "dl_state.json"
        state.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["step"] == 100

    def test_atomic_write_via_tmp(self, tmp_path):
        """The .tmp file should not exist after a successful save."""
        state = self._make_state()
        path  = tmp_path / "dl_state.json"
        state.save(path)
        assert not (tmp_path / "dl_state.tmp").exists()
        assert path.exists()

    def test_backward_compat_missing_crop_sizes(self, tmp_path):
        """
        Checkpoint files from older versions may lack global_crop_size /
        local_crop_size.  They should default gracefully.
        """
        path = tmp_path / "old_state.json"
        old_data = {
            "step":           50,
            "epoch":          1,
            "dataset_names":  ["laion"],
            "mixing_weights": [1.0],
            # global_crop_size / local_crop_size intentionally absent
        }
        path.write_text(json.dumps(old_data))

        loaded = CheckpointState.load(path)
        assert loaded.global_crop_size == 224   # dataclass default
        assert loaded.local_crop_size  == 96    # dataclass default
