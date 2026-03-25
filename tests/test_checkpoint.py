"""tests/test_checkpoint.py
===========================
Unit tests for :class:`dino_loader.checkpoint.DataLoaderCheckpointer`.

Coverage
--------
- save: creates file, encodes step in filename, respects every_n_steps,
  rank-0-only gate
- load: returns latest, None when empty, warns on corrupt file
- LATEST pointer: created on first save, updated to most-recent,
  survives pruning [CK-3]
- Pruning: retains at most 3 most-recent checkpoints
- Crash safety: .tmp file cleaned up when LATEST write fails
- Fallback: glob-sort when LATEST pointer is missing or stale
- Directory auto-creation
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.checkpoint import _KEEP_LAST, _LATEST_FILE, DataLoaderCheckpointer
from dino_loader.config import CheckpointState


def _state(step: int = 100, epoch: int = 1) -> CheckpointState:
    return CheckpointState(
        step=step,
        epoch=epoch,
        dataset_names=["laion", "imagenet"],
        mixing_weights=[0.7, 0.3],
        global_crop_size=224,
        local_crop_size=96,
    )


# ══════════════════════════════════════════════════════════════════════════════
# save
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointerSave:

    def test_creates_file(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=10))
        files = list(tmp_path.glob("dl_state_*.json"))
        assert len(files) == 1

    def test_filename_encodes_step(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=42))
        files = list(tmp_path.glob("dl_state_*.json"))
        assert "000000000042" in files[0].name

    def test_noop_on_non_multiple(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=10, rank=0)
        ckpt.save(_state(step=7))
        assert len(list(tmp_path.glob("dl_state_*.json"))) == 0

    def test_noop_for_non_rank0(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=1)
        ckpt.save(_state(step=10))
        assert len(list(tmp_path.glob("dl_state_*.json"))) == 0

    def test_directory_created_automatically(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        ckpt = DataLoaderCheckpointer(nested, every_n_steps=1, rank=0)
        ckpt.save(_state(step=1))
        assert Path(nested).is_dir()


# ══════════════════════════════════════════════════════════════════════════════
# load
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointerLoad:

    def test_returns_latest_by_step(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=10))
        ckpt.save(_state(step=20))
        ckpt.save(_state(step=30))
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 30

    def test_returns_none_when_no_files(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        assert ckpt.load() is None

    def test_warns_on_corrupt_file(self, tmp_path, caplog):
        import logging
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        (tmp_path / "dl_state_000000000001.json").write_text("NOT JSON {{{")
        with caplog.at_level(logging.WARNING):
            result = ckpt.load()
        assert result is None
        assert any(
            "corrupt" in r.message.lower() or "could not" in r.message.lower()
            for r in caplog.records
        )


# ══════════════════════════════════════════════════════════════════════════════
# LATEST pointer [CK-3]
# ══════════════════════════════════════════════════════════════════════════════


class TestLatestPointer:

    def test_latest_file_created_on_first_save(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=10))
        assert (tmp_path / _LATEST_FILE).exists()

    def test_latest_points_to_most_recent_save(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in (10, 20, 30):
            ckpt.save(_state(step=step))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert "000000000030" in latest_name

    def test_load_uses_latest_pointer(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in (10, 20, 30):
            ckpt.save(_state(step=step))
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 30

    def test_latest_survives_pruning(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        n = _KEEP_LAST + 2
        for step in range(1, n + 1):
            ckpt.save(_state(step=step))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert f"{n:012d}" in latest_name

    def test_fallback_to_glob_when_no_latest(self, tmp_path):
        state = _state(step=50)
        state.save(tmp_path / "dl_state_000000000050.json")
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 50

    def test_latest_tmp_cleaned_up_on_write_failure(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)

        def bad_rename(self_path, target):
            raise OSError("simulated disk full")

        with patch.object(Path, "rename", bad_rename):
            ckpt.save(_state(step=5))

        assert not (tmp_path / f"{_LATEST_FILE}.tmp").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Pruning
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointerPruning:

    def test_prune_keeps_only_three(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in [10, 20, 30, 40, 50]:
            ckpt.save(_state(step=step))
        files = sorted(tmp_path.glob("dl_state_*.json"))
        assert len(files) == 3

    def test_prune_retains_most_recent_three(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in [10, 20, 30, 40, 50]:
            ckpt.save(_state(step=step))
        files = sorted(tmp_path.glob("dl_state_*.json"))
        steps = [int(f.stem.split("_")[-1]) for f in files]
        assert max(steps) == 50
        assert min(steps) == 30
