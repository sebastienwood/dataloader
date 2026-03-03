"""
tests/conftest.py
=================
Shared pytest fixtures available to all test modules.

The fixtures are scoped as narrowly as possible:
- ``tmp_dataset_dir`` — function-scoped tmp directory with synthetic shards.
- ``cpu_backend``     — module-scoped (constructing it is free).
- ``small_aug_cfg``   — session-scoped (stateless dataclass).
- ``small_loader_cfg``— session-scoped.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure the src layout is importable without a full install
_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tests.fixtures import scaffold_dataset_dir, write_shard
from dino_loader.backends.cpu import CPUBackend
from dino_loader.config import DatasetSpec, DINOAugConfig, LoaderConfig


# ══════════════════════════════════════════════════════════════════════════════
# Directory fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def tmp_dataset_dir(tmp_path: Path):
    """
    A temporary directory populated with two synthetic shards for one dataset.

    Layout::

        tmp_path/
          public/rgb/test_ds/train/
            shard-000000.tar  (8 samples, with .json)
            shard-000000.idx
            shard-000001.tar  (8 samples, with .json)
            shard-000001.idx
    """
    tar_paths = scaffold_dataset_dir(
        root     = tmp_path,
        conf     = "public",
        modality = "rgb",
        name     = "test_ds",
        split    = "train",
        n_shards = 2,
        n_samples_per_shard = 8,
        with_metadata = True,
    )
    return tmp_path, tar_paths


@pytest.fixture(scope="function")
def multi_dataset_dirs(tmp_path: Path):
    """
    Two datasets in a shared root, used for mixing tests.
    Returns (root, {name: [tar_path, ...]}).
    """
    datasets = {}
    for ds_name, n_samples in [("alpha", 6), ("beta", 10)]:
        tar_paths = scaffold_dataset_dir(
            root     = tmp_path,
            conf     = "public",
            modality = "rgb",
            name     = ds_name,
            split    = "train",
            n_shards = 2,
            n_samples_per_shard = n_samples,
            with_metadata = True,
        )
        datasets[ds_name] = tar_paths
    return tmp_path, datasets


@pytest.fixture(scope="function")
def shard_with_low_quality(tmp_path: Path):
    """
    A single shard where half the samples have quality_score=0.1 (below common
    min_sample_quality thresholds) and half have 0.9.
    """
    scores = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    tar_path, idx_path = write_shard(
        directory      = tmp_path,
        shard_idx      = 0,
        n_samples      = 8,
        with_metadata  = True,
        quality_scores = scores,
    )
    return tar_path, idx_path, scores


@pytest.fixture(scope="function")
def shard_without_metadata(tmp_path: Path):
    """A shard with no .json sidecars."""
    tar_path, idx_path = write_shard(
        directory      = tmp_path,
        shard_idx      = 0,
        n_samples      = 8,
        with_metadata  = False,
    )
    return tar_path, idx_path


# ══════════════════════════════════════════════════════════════════════════════
# Backend fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def cpu_backend():
    """The CPU backend instance."""
    return CPUBackend()


# ══════════════════════════════════════════════════════════════════════════════
# Config fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def small_aug_cfg():
    """
    Minimal DINOAugConfig for fast CPU tests.
    - 2 global crops at 32px, 2 local crops at 16px (instead of 224/96).
    - Heavy augmentations kept at realistic probabilities.
    """
    return DINOAugConfig(
        global_crop_size     = 32,
        local_crop_size      = 16,
        n_global_crops       = 2,
        n_local_crops        = 2,
        max_global_crop_size = 64,
        max_local_crop_size  = 32,
        global_crops_scale   = (0.32, 1.0),
        local_crops_scale    = (0.05, 0.32),
        blur_prob_global1    = 1.0,
        blur_prob_global2    = 0.1,
        blur_prob_local      = 0.5,
        solarize_prob        = 0.2,
    )


@pytest.fixture(scope="session")
def small_loader_cfg(tmp_path_factory):
    """
    LoaderConfig suited for CPU / no-SLURM testing.
    Checkpoints go into a temporary directory.
    """
    ckpt_dir = str(tmp_path_factory.mktemp("checkpoints"))
    return LoaderConfig(
        node_shm_gb              = 0.1,   # 100 MB — plenty for a test shard
        shard_prefetch_window    = 2,
        shard_extraction_workers = 2,
        shuffle_buffer_size      = 4,
        use_fp8_output           = False,  # no TE in CI
        stateful_dataloader      = True,
        checkpoint_dir           = ckpt_dir,
        checkpoint_every_steps   = 2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dataset spec helper
# ══════════════════════════════════════════════════════════════════════════════

def make_spec(name: str, tar_paths: list, weight: float = 1.0, **kwargs) -> DatasetSpec:
    """Convenience factory used across multiple test modules."""
    return DatasetSpec(name=name, shards=tar_paths, weight=weight, **kwargs)
