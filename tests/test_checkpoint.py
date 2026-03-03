"""
tests/test_checkpoint.py
========================
Unit tests for dino_loader.checkpoint.DataLoaderCheckpointer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.checkpoint import DataLoaderCheckpointer
from dino_loader.config import CheckpointState


def _state(step: int = 100, epoch: int = 1) -> CheckpointState:
    return CheckpointState(
        step            = step,
        epoch           = epoch,
        dataset_names   = ["laion", "imagenet"],
        mixing_weights  = [0.7, 0.3],
        global_crop_size= 224,
        local_crop_size = 96,
    )


class TestDataLoaderCheckpointer:

    def test_save_creates_file(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=10))
        files = list(tmp_path.glob("dl_state_*.json"))
        assert len(files) == 1

    def test_save_filename_encodes_step(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=42))
        files = list(tmp_path.glob("dl_state_*.json"))
        assert "000000000042" in files[0].name

    def test_save_noop_on_non_multiple(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=10, rank=0)
        ckpt.save(_state(step=7))   # 7 % 10 != 0
        assert len(list(tmp_path.glob("dl_state_*.json"))) == 0

    def test_save_noop_non_rank0(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=1)
        ckpt.save(_state(step=10))
        assert len(list(tmp_path.glob("dl_state_*.json"))) == 0

    def test_load_returns_latest(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=10))
        ckpt.save(_state(step=20))
        ckpt.save(_state(step=30))
        loaded = ckpt.load()
        assert loaded is not None
        assert loaded.step == 30

    def test_load_returns_none_if_no_files(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        assert ckpt.load() is None

    def test_load_warns_on_corrupt_file(self, tmp_path, caplog):
        import logging
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        # Write a corrupt JSON file
        (tmp_path / "dl_state_000000000001.json").write_text("NOT JSON {{{")
        with caplog.at_level(logging.WARNING):
            result = ckpt.load()
        assert result is None
        assert any("corrupt" in r.message.lower() or "could not" in r.message.lower()
                   for r in caplog.records)

    def test_prune_keeps_only_3(self, tmp_path):
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in [10, 20, 30, 40, 50]:
            ckpt.save(_state(step=step))
        files = sorted(tmp_path.glob("dl_state_*.json"))
        assert len(files) == 3, f"Expected 3 retained, got {len(files)}"
        # The 3 most recent should be retained
        steps = [int(f.stem.split("_")[-1]) for f in files]
        assert max(steps) == 50
        assert min(steps) == 30

    def test_directory_created_automatically(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        ckpt   = DataLoaderCheckpointer(nested, every_n_steps=1, rank=0)
        ckpt.save(_state(step=1))
        assert Path(nested).is_dir()
