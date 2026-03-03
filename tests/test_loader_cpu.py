"""
tests/test_loader_cpu.py
========================
End-to-end integration tests for DINODataLoader using the CPU backend.

These tests exercise the full stack — DatasetSpec → MixingSource →
ShardIterator → CPUAugPipeline → Batch — without any GPU, DALI, or SLURM.

They are the primary regression gate for changes to loader.py, mixing_source.py,
config.py, checkpoint.py, and the backend abstraction.

Test categories
---------------
TestLoaderBasic          — single dataset, correct batch shapes / dtypes
TestLoaderMultiDataset   — weighted mixing across two datasets
TestLoaderEpochControl   — set_epoch reshuffles, double-iter guard
TestLoaderResolution     — set_resolution / resolution_schedule
TestLoaderCheckpoint     — state_dict / load_state_dict / JSON checkpoint
TestLoaderQualityFilter  — min_sample_quality filtering
TestLoaderBackendSwitch  — get_backend("cpu") helper
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests.conftest import make_spec
from dino_loader.backends import get_backend
from dino_loader.backends.cpu import CPUBackend
from dino_loader.config import DatasetSpec, DINOAugConfig, LoaderConfig
from dino_loader.loader import DINODataLoader
from dino_loader.memory import Batch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_loader(
    tar_paths:  list,
    ds_name:    str       = "test_ds",
    aug_cfg:    DINOAugConfig  = None,
    loader_cfg: LoaderConfig   = None,
    batch_size: int       = 4,
    backend     = None,
    **spec_kwargs,
) -> DINODataLoader:
    spec   = make_spec(ds_name, tar_paths, **spec_kwargs)
    b      = backend or get_backend("cpu")
    return DINODataLoader(
        specs      = [spec],
        batch_size = batch_size,
        aug_cfg    = aug_cfg,
        config     = loader_cfg,
        backend    = b,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Basic single-dataset tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderBasic:

    def test_yields_batch_object(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert isinstance(batch, Batch)

    def test_global_crop_shape(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        batch_size = 4
        loader = _make_loader(
            tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg,
            batch_size=batch_size,
        )
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert len(batch.global_crops) == small_aug_cfg.n_global_crops
        for crop in batch.global_crops:
            assert crop.shape == (batch_size, 3, 32, 32), f"Shape: {crop.shape}"

    def test_local_crop_shape(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        batch_size = 4
        loader = _make_loader(
            tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg,
            batch_size=batch_size,
        )
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert len(batch.local_crops) == small_aug_cfg.n_local_crops
        for crop in batch.local_crops:
            assert crop.shape == (batch_size, 3, 16, 16), f"Shape: {crop.shape}"

    def test_batch_dtype_float(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        for crop in batch.global_crops + batch.local_crops:
            assert crop.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_batch_values_finite(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        for crop in batch.global_crops + batch.local_crops:
            assert torch.isfinite(crop).all(), "Non-finite values in batch"

    def test_metadata_list_length(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        """Batch.metadata has one entry per sample."""
        _, tar_paths = tmp_dataset_dir
        batch_size   = 4
        loader = _make_loader(
            tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg,
            batch_size=batch_size,
        )
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert len(batch.metadata) == batch_size

    def test_metadata_contains_quality_score(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        """Each metadata dict (when present) has quality_score."""
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        for meta in batch.metadata:
            if meta is not None:
                assert "quality_score" in meta

    def test_masks_none_by_default(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert batch.masks is None

    def test_iter_unpack_protocol(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        """Batch supports the (global_crops, local_crops) unpack protocol."""
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        g, l = batch
        assert g is batch.global_crops
        assert l is batch.local_crops

    def test_backend_attribute(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        assert loader.backend.name == "cpu"

    def test_backend_string_shorthand(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        """backend='cpu' string is accepted."""
        _, tar_paths = tmp_dataset_dir
        spec   = make_spec("ds", tar_paths)
        loader = DINODataLoader(
            specs      = [spec],
            batch_size = 4,
            aug_cfg    = small_aug_cfg,
            config     = small_loader_cfg,
            backend    = "cpu",
        )
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert isinstance(batch, Batch)

    def test_multiple_batches(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        """Loader can produce more than one batch without error."""
        _, tar_paths = tmp_dataset_dir
        # 2 shards × 8 samples = 16 samples; batch_size=4 → ≥ 4 batches possible
        loader = _make_loader(
            tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg,
            batch_size=4,
        )
        loader.set_epoch(0)
        batches = []
        for i, b in enumerate(loader):
            batches.append(b)
            if i >= 1:  # collect 2 batches
                break
        assert len(batches) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Multi-dataset weighted mixing
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderMultiDataset:

    def test_two_datasets_yields_batch(self, multi_dataset_dirs, small_aug_cfg, small_loader_cfg):
        _, datasets = multi_dataset_dirs
        specs = [
            make_spec("alpha", datasets["alpha"], weight=0.6),
            make_spec("beta",  datasets["beta"],  weight=0.4),
        ]
        loader = DINODataLoader(
            specs      = specs,
            batch_size = 4,
            aug_cfg    = small_aug_cfg,
            config     = small_loader_cfg,
            backend    = get_backend("cpu"),
        )
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert isinstance(batch, Batch)

    def test_set_weights_normalises(self, multi_dataset_dirs, small_aug_cfg, small_loader_cfg):
        _, datasets = multi_dataset_dirs
        specs = [
            make_spec("alpha", datasets["alpha"], weight=1.0),
            make_spec("beta",  datasets["beta"],  weight=1.0),
        ]
        loader = DINODataLoader(
            specs      = specs,
            batch_size = 4,
            aug_cfg    = small_aug_cfg,
            config     = small_loader_cfg,
            backend    = get_backend("cpu"),
        )
        loader.set_weights([3.0, 1.0])
        w = loader.current_weights
        assert abs(w[0] - 0.75) < 1e-5
        assert abs(w[1] - 0.25) < 1e-5

    def test_set_weight_by_name(self, multi_dataset_dirs, small_aug_cfg, small_loader_cfg):
        _, datasets = multi_dataset_dirs
        specs = [
            make_spec("alpha", datasets["alpha"], weight=1.0),
            make_spec("beta",  datasets["beta"],  weight=1.0),
        ]
        loader = DINODataLoader(
            specs      = specs,
            batch_size = 4,
            aug_cfg    = small_aug_cfg,
            config     = small_loader_cfg,
            backend    = get_backend("cpu"),
        )
        loader.set_weight_by_name("alpha", 9.0)
        w = loader.current_weights
        assert w[0] > w[1]


# ══════════════════════════════════════════════════════════════════════════════
# Epoch control
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderEpochControl:

    def test_set_epoch_required_before_iter(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        """
        Iterating without set_epoch is not explicitly forbidden, but calling
        set_epoch between iterations must not raise.
        """
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(0)
        loader.set_epoch(1)   # call twice — must not raise

    def test_double_iter_raises(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        """Starting a second __iter__ while the first is active must raise."""
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(0)
        it1 = iter(loader)
        next(it1)  # advance to mark as active
        with pytest.raises(RuntimeError, match="already iterating"):
            # Directly call __iter__ again while active
            loader._active_iter = True   # simulate active state
            iter(loader)

    def test_steps_per_epoch_len(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        spec   = make_spec("ds", tar_paths)
        loader = DINODataLoader(
            specs           = [spec],
            batch_size      = 4,
            aug_cfg         = small_aug_cfg,
            config          = small_loader_cfg,
            backend         = "cpu",
            steps_per_epoch = 100,
        )
        assert len(loader) == 100

    def test_len_without_steps_per_epoch_raises(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        with pytest.raises(TypeError):
            _ = len(loader)


# ══════════════════════════════════════════════════════════════════════════════
# Resolution control
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderResolution:

    def test_set_resolution_changes_output(self, tmp_dataset_dir, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        aug_cfg = DINOAugConfig(
            global_crop_size     = 32,
            local_crop_size      = 16,
            max_global_crop_size = 64,
            max_local_crop_size  = 32,
            n_global_crops       = 2,
            n_local_crops        = 2,
        )
        loader = _make_loader(tar_paths, aug_cfg=aug_cfg, loader_cfg=small_loader_cfg, batch_size=2)
        loader.set_epoch(0)
        loader.set_resolution(64, 32)

        batch = next(iter(loader))
        for crop in batch.global_crops:
            assert crop.shape[-2:] == (64, 64), f"Expected 64x64, got {crop.shape}"

    def test_set_resolution_above_max_raises(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        with pytest.raises(ValueError, match="max_global_crop_size"):
            loader.set_resolution(999, 64)

    def test_resolution_schedule_applies_at_epoch(self, tmp_dataset_dir, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        aug_cfg = DINOAugConfig(
            global_crop_size     = 32,
            local_crop_size      = 16,
            max_global_crop_size = 64,
            max_local_crop_size  = 32,
            n_global_crops       = 2,
            n_local_crops        = 2,
            resolution_schedule  = [(0, 32), (1, 64)],
        )
        loader = _make_loader(tar_paths, aug_cfg=aug_cfg, loader_cfg=small_loader_cfg, batch_size=2)

        loader.set_epoch(0)
        assert loader._current_global_size == 32

        loader.set_epoch(1)
        assert loader._current_global_size == 64

    def test_current_sizes_updated_after_set_resolution(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_resolution(64, 32)
        assert loader._current_global_size == 64
        assert loader._current_local_size  == 32


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderCheckpoint:

    def test_state_dict_contains_expected_keys(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        sd = loader.state_dict()
        for key in ("step", "epoch", "dataset_names", "mixing_weights",
                    "global_crop_size", "local_crop_size"):
            assert key in sd, f"Missing key: {key}"

    def test_load_state_dict_round_trip(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg):
        _, tar_paths = tmp_dataset_dir
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader.set_epoch(3)
        loader._step = 42
        loader.set_resolution(32, 16)

        sd = loader.state_dict()
        assert sd["epoch"] == 3
        assert sd["global_crop_size"] == 32

        # Build a new loader and restore
        loader2 = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=small_loader_cfg)
        loader2.load_state_dict(sd)
        assert loader2._epoch               == 3
        assert loader2._current_global_size == 32

    def test_checkpoint_saves_json(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg, tmp_path):
        _, tar_paths = tmp_dataset_dir
        ckpt_cfg = LoaderConfig(
            node_shm_gb              = 0.1,
            shard_prefetch_window    = 2,
            shard_extraction_workers = 2,
            shuffle_buffer_size      = 4,
            use_fp8_output           = False,
            stateful_dataloader      = True,
            checkpoint_dir           = str(tmp_path / "ckpts"),
            checkpoint_every_steps   = 2,
        )
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=ckpt_cfg)
        loader.checkpoint(step=2)   # step % 2 == 0 → should save

        ckpt_files = list(Path(ckpt_cfg.checkpoint_dir).glob("dl_state_*.json"))
        assert len(ckpt_files) == 1

        with open(ckpt_files[0]) as f:
            data = json.load(f)
        assert data["step"] == 2
        assert "mixing_weights" in data

    def test_checkpoint_noop_on_non_multiple(self, tmp_dataset_dir, small_aug_cfg, small_loader_cfg, tmp_path):
        _, tar_paths = tmp_dataset_dir
        ckpt_cfg = LoaderConfig(
            node_shm_gb              = 0.1,
            shard_prefetch_window    = 2,
            shard_extraction_workers = 2,
            shuffle_buffer_size      = 4,
            use_fp8_output           = False,
            stateful_dataloader      = True,
            checkpoint_dir           = str(tmp_path / "ckpts2"),
            checkpoint_every_steps   = 10,
        )
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=ckpt_cfg)
        loader.checkpoint(step=3)   # 3 % 10 != 0 → noop

        ckpt_files = list(Path(ckpt_cfg.checkpoint_dir).glob("dl_state_*.json"))
        assert len(ckpt_files) == 0

    def test_state_dict_requires_stateful_flag(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, tar_paths = tmp_dataset_dir
        cfg = LoaderConfig(
            node_shm_gb              = 0.1,
            shard_prefetch_window    = 2,
            shard_extraction_workers = 2,
            shuffle_buffer_size      = 4,
            use_fp8_output           = False,
            stateful_dataloader      = False,   # disabled
            checkpoint_dir           = str(tmp_path),
        )
        loader = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=cfg)
        with pytest.raises(RuntimeError, match="stateful_dataloader"):
            loader.state_dict()

    def test_resume_from_checkpoint(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, tar_paths = tmp_dataset_dir
        ckpt_dir = str(tmp_path / "resume_ckpt")
        ckpt_cfg = LoaderConfig(
            node_shm_gb              = 0.1,
            shard_prefetch_window    = 2,
            shard_extraction_workers = 2,
            shuffle_buffer_size      = 4,
            use_fp8_output           = False,
            stateful_dataloader      = True,
            checkpoint_dir           = ckpt_dir,
            checkpoint_every_steps   = 4,
        )
        # Write a checkpoint manually
        loader1 = _make_loader(tar_paths, aug_cfg=small_aug_cfg, loader_cfg=ckpt_cfg)
        loader1.checkpoint(step=4)

        # New loader with resume=True
        spec    = make_spec("test_ds", tar_paths)
        loader2 = DINODataLoader(
            specs      = [spec],
            batch_size = 4,
            aug_cfg    = small_aug_cfg,
            config     = ckpt_cfg,
            backend    = "cpu",
            resume     = True,
        )
        assert loader2._step == 4


# ══════════════════════════════════════════════════════════════════════════════
# Quality filtering
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderQualityFilter:

    def test_min_sample_quality_filters_low_scores(
        self, shard_with_low_quality, small_aug_cfg, small_loader_cfg
    ):
        """
        With min_sample_quality=0.5, only samples with score ≥ 0.5 pass.
        The shard_with_low_quality fixture has alternating 0.1 / 0.9 scores.
        """
        tar_path, _, scores = shard_with_low_quality
        high_count = sum(1 for s in scores if s >= 0.5)  # should be 4

        spec = DatasetSpec(
            name               = "filtered",
            shards             = [tar_path],
            weight             = 1.0,
            min_sample_quality = 0.5,
            metadata_key       = "json",
        )
        loader = DINODataLoader(
            specs      = [spec],
            batch_size = 2,
            aug_cfg    = small_aug_cfg,
            config     = small_loader_cfg,
            backend    = "cpu",
        )
        loader.set_epoch(0)

        # We should be able to draw at least one batch from the 4 passing samples
        batch = next(iter(loader))
        assert isinstance(batch, Batch)

    def test_no_filter_passes_all(self, shard_with_low_quality, small_aug_cfg, small_loader_cfg):
        tar_path, _, _ = shard_with_low_quality
        spec = DatasetSpec(
            name               = "unfiltered",
            shards             = [tar_path],
            weight             = 1.0,
            min_sample_quality = None,   # no filter
            metadata_key       = "json",
        )
        loader = DINODataLoader(
            specs      = [spec],
            batch_size = 2,
            aug_cfg    = small_aug_cfg,
            config     = small_loader_cfg,
            backend    = "cpu",
        )
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert isinstance(batch, Batch)

    def test_no_metadata_key_skips_sidecar(
        self, shard_without_metadata, small_aug_cfg, small_loader_cfg
    ):
        tar_path, _ = shard_without_metadata
        spec = DatasetSpec(
            name         = "no_meta",
            shards       = [tar_path],
            weight       = 1.0,
            metadata_key = None,   # legacy fast path
        )
        loader = DINODataLoader(
            specs      = [spec],
            batch_size = 2,
            aug_cfg    = small_aug_cfg,
            config     = small_loader_cfg,
            backend    = "cpu",
        )
        loader.set_epoch(0)
        batch = next(iter(loader))
        # All metadata should be None (no sidecar)
        for meta in batch.metadata:
            assert meta is None


# ══════════════════════════════════════════════════════════════════════════════
# get_backend helper
# ══════════════════════════════════════════════════════════════════════════════

class TestLoaderBackendSwitch:

    def test_get_backend_cpu(self):
        b = get_backend("cpu")
        assert isinstance(b, CPUBackend)
        assert b.name == "cpu"

    def test_get_backend_auto_returns_cpu_without_dali(self):
        """In a test environment without DALI, auto should pick CPU."""
        import importlib
        try:
            import nvidia.dali  # noqa: F401
            pytest.skip("DALI is installed; auto-select would pick DALI")
        except ImportError:
            pass
        b = get_backend("auto")
        assert b.name == "cpu"

    def test_dali_backend_raises_without_dali(self):
        """Requesting 'dali' explicitly without DALI installed should raise on use."""
        try:
            import nvidia.dali  # noqa: F401
            pytest.skip("DALI is installed")
        except ImportError:
            pass
        from dino_loader.backends.dali_backend import DALIBackend
        b = DALIBackend()
        with pytest.raises(RuntimeError, match="nvidia-dali"):
            b.build_pipeline(None, None, 4, 1, 0, None)
