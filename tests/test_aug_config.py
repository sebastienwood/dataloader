"""tests/test_aug_config.py
===========================
Unit tests for :class:`dino_loader.config.DINOAugConfig`.

Coverage
--------
- Default field values
- n_views property (n_global_crops + n_local_crops)
- max_crop_sizes: default to crop_sizes when unset, explicit override
- resolution_schedule: sorted on construction, valid/invalid epoch values
- crop_size_at_epoch: no schedule, with schedule, boundary conditions
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.config import DINOAugConfig


class TestDINOAugConfigDefaults:

    def test_global_crop_size(self):
        assert DINOAugConfig().global_crop_size == 224

    def test_local_crop_size(self):
        assert DINOAugConfig().local_crop_size == 96

    def test_n_global_crops(self):
        assert DINOAugConfig().n_global_crops == 2

    def test_n_local_crops(self):
        assert DINOAugConfig().n_local_crops == 8

    def test_mean_imagenet(self):
        assert DINOAugConfig().mean == (0.485, 0.456, 0.406)

    def test_std_imagenet(self):
        assert DINOAugConfig().std == (0.229, 0.224, 0.225)


class TestNViews:

    def test_n_views_equals_global_plus_local(self):
        cfg = DINOAugConfig(n_global_crops=2, n_local_crops=6)
        assert cfg.n_views == 8

    def test_n_views_production_settings(self):
        cfg = DINOAugConfig(n_global_crops=2, n_local_crops=8)
        assert cfg.n_views == 10

    def test_n_views_minimal(self):
        cfg = DINOAugConfig(n_global_crops=1, n_local_crops=0)
        assert cfg.n_views == 1


class TestMaxCropSizes:

    def test_max_global_defaults_to_global_crop_size(self):
        cfg = DINOAugConfig(global_crop_size=224)
        assert cfg.max_global_crop_size == 224

    def test_max_local_defaults_to_local_crop_size(self):
        cfg = DINOAugConfig(local_crop_size=96)
        assert cfg.max_local_crop_size == 96

    def test_max_global_explicit_override(self):
        cfg = DINOAugConfig(global_crop_size=224, max_global_crop_size=518)
        assert cfg.max_global_crop_size == 518

    def test_max_local_explicit_override(self):
        cfg = DINOAugConfig(local_crop_size=96, max_local_crop_size=224)
        assert cfg.max_local_crop_size == 224


class TestResolutionSchedule:

    def test_schedule_is_sorted_by_epoch(self):
        cfg = DINOAugConfig(resolution_schedule=[(30, 518), (0, 224), (10, 448)])
        epochs = [e for e, _ in cfg.resolution_schedule]
        assert epochs == sorted(epochs)

    def test_empty_schedule_is_valid(self):
        cfg = DINOAugConfig(resolution_schedule=[])
        assert cfg.resolution_schedule == []

    def test_negative_epoch_raises(self):
        with pytest.raises(ValueError):
            DINOAugConfig(resolution_schedule=[(-1, 224)])

    def test_single_entry_valid(self):
        cfg = DINOAugConfig(resolution_schedule=[(0, 224)])
        assert len(cfg.resolution_schedule) == 1


class TestCropSizeAtEpoch:

    def test_no_schedule_returns_global_crop_size(self):
        cfg = DINOAugConfig(global_crop_size=224)
        assert cfg.crop_size_at_epoch(0) == 224
        assert cfg.crop_size_at_epoch(50) == 224

    def test_schedule_boundary_at_epoch_0(self):
        cfg = DINOAugConfig(
            global_crop_size=224,
            max_global_crop_size=518,
            resolution_schedule=[(0, 224), (10, 448), (30, 518)],
        )
        assert cfg.crop_size_at_epoch(0) == 224

    def test_schedule_before_first_threshold(self):
        cfg = DINOAugConfig(
            global_crop_size=224,
            max_global_crop_size=518,
            resolution_schedule=[(0, 224), (10, 448), (30, 518)],
        )
        assert cfg.crop_size_at_epoch(9) == 224

    def test_schedule_at_second_threshold(self):
        cfg = DINOAugConfig(
            global_crop_size=224,
            max_global_crop_size=518,
            resolution_schedule=[(0, 224), (10, 448), (30, 518)],
        )
        assert cfg.crop_size_at_epoch(10) == 448

    def test_schedule_between_thresholds(self):
        cfg = DINOAugConfig(
            global_crop_size=224,
            max_global_crop_size=518,
            resolution_schedule=[(0, 224), (10, 448), (30, 518)],
        )
        assert cfg.crop_size_at_epoch(29) == 448

    def test_schedule_at_last_threshold(self):
        cfg = DINOAugConfig(
            global_crop_size=224,
            max_global_crop_size=518,
            resolution_schedule=[(0, 224), (10, 448), (30, 518)],
        )
        assert cfg.crop_size_at_epoch(30) == 518

    def test_schedule_beyond_last_threshold(self):
        cfg = DINOAugConfig(
            global_crop_size=224,
            max_global_crop_size=518,
            resolution_schedule=[(0, 224), (10, 448), (30, 518)],
        )
        assert cfg.crop_size_at_epoch(99) == 518
