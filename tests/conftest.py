"""tests/conftest.py
====================
Shared pytest fixtures for the dino_loader test suite.

Design principles
-----------------
- Only pytest fixtures live here. Pure helper functions belong in
  ``tests/fixtures/__init__.py``.
- Fixtures are scoped as narrowly as possible to avoid hidden cross-test state.

This conftest covers dino_loader tests only. dino_datasets and dino_env
each have their own independent test suite and conftest.

Fixture overview
----------------
============= ========================== ==========================================
Scope         Fixture                    Rationale
============= ========================== ==========================================
session       small_aug_cfg              Stateless dataclass — safe to share.
session       small_loader_cfg           Same; checkpoint dir is session-scoped.
module        cpu_backend                Constructing CPUBackend is cheap.
function      tmp_dataset_dir            Writes files to disk — must be isolated.
function      multi_dataset_dirs         Same.
function      shard_with_low_quality     Same.
function      shard_without_metadata     Same.
============= ========================== ==========================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure src is importable without a full package install.
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_datasets import DatasetSpec

from dino_loader.backends.cpu import CPUBackend
from dino_loader.config import DINOAugConfig, LoaderConfig
from tests.fixtures import scaffold_dataset_dir, write_shard

# ── Public helper (not a fixture) ─────────────────────────────────────────────

def make_spec(name: str, tar_paths: list, weight: float = 1.0, **kwargs) -> DatasetSpec:
    """Convenience factory for DatasetSpec used in loader tests."""
    return DatasetSpec(name=name, shards=tuple(tar_paths), weight=weight, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Config fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def small_aug_cfg() -> DINOAugConfig:
    """Minimal DINOAugConfig for fast CPU tests.

    Uses 32 px global crops and 16 px local crops rather than production
    224 / 96 values to keep test runtime low.
    """
    return DINOAugConfig(
        global_crop_size=32,
        local_crop_size=16,
        n_global_crops=2,
        n_local_crops=2,
        max_global_crop_size=64,
        max_local_crop_size=32,
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        blur_prob_global1=1.0,
        blur_prob_global2=0.1,
        blur_prob_local=0.5,
        solarize_prob=0.2,
    )


@pytest.fixture(scope="session")
def small_loader_cfg(tmp_path_factory) -> LoaderConfig:
    """LoaderConfig suited for CPU / no-SLURM testing.

    - SHM budget: 100 MB.
    - Stateful dataloader enabled for checkpoint round-trip tests.
    - Checkpoint directory is session-scoped.
    """
    ckpt_dir = str(tmp_path_factory.mktemp("checkpoints"))
    return LoaderConfig(
        node_shm_gb=0.1,
        shard_prefetch_window=2,
        shard_extraction_workers=2,
        shuffle_buffer_size=4,
        use_fp8_output=False,
        stateful_dataloader=True,
        checkpoint_dir=ckpt_dir,
        checkpoint_every_steps=2,
        stall_timeout_s=0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Backend fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def cpu_backend() -> CPUBackend:
    """The CPU backend singleton. Module-scoped because construction is cheap."""
    return CPUBackend()


# ══════════════════════════════════════════════════════════════════════════════
# Directory / shard fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path):
    """Return a temporary directory pre-populated with two synthetic shards.

    Layout::

        tmp_path/
          <conf>/rgb/test_ds/outputs/default/train/
            shard-000000.tar  (8 samples, with .json)
            shard-000000.idx
            shard-000001.tar  (8 samples, with .json)
            shard-000001.idx

    Returns:
        (root, tar_paths) : (Path, list[str])

    """
    tar_paths = scaffold_dataset_dir(
        root=tmp_path,
        modality="rgb",
        name="test_ds",
        split="train",
        n_shards=2,
        n_samples_per_shard=8,
        with_metadata=True,
    )
    return tmp_path, tar_paths


@pytest.fixture
def multi_dataset_dirs(tmp_path: Path):
    """Return two datasets in a shared root, used for mixing tests.

    Returns:
        (root, datasets) : (Path, dict[str, list[str]])
            *datasets* maps dataset name → list of ``.tar`` paths.

    """
    datasets: dict[str, list[str]] = {}
    for ds_name, n_samples in [("alpha", 6), ("beta", 10)]:
        tar_paths = scaffold_dataset_dir(
            root=tmp_path,
            modality="rgb",
            name=ds_name,
            split="train",
            n_shards=2,
            n_samples_per_shard=n_samples,
            with_metadata=True,
        )
        datasets[ds_name] = tar_paths
    return tmp_path, datasets


@pytest.fixture
def shard_with_low_quality(tmp_path: Path):
    """Return a single shard with alternating quality_score 0.1 / 0.9.

    With min_sample_quality=0.5, exactly half the samples pass.

    Returns:
        (tar_path, idx_path, quality_scores) : (str, str, list[float])

    """
    scores = [0.1, 0.9] * 4  # 8 samples: 4 low, 4 high
    tar_path, idx_path = write_shard(
        directory=tmp_path,
        shard_idx=0,
        n_samples=8,
        with_metadata=True,
        quality_scores=scores,
    )
    return tar_path, idx_path, scores


@pytest.fixture
def shard_without_metadata(tmp_path: Path):
    """Return a single shard with no .json sidecars.

    Returns:
        (tar_path, idx_path) : (str, str)

    """
    tar_path, idx_path = write_shard(
        directory=tmp_path,
        shard_idx=0,
        n_samples=8,
        with_metadata=False,
    )
    return tar_path, idx_path
