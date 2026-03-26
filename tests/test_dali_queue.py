"""tests/test_dali_queue.py
===========================
Tests verifying the removal of AsyncPrefetchIterator and correct DALI queue
sizing after that removal.

Background
----------
AsyncPrefetchIterator was removed because DALI's internal prefetch queues
(controlled by ``dali_cpu_queue`` / ``dali_gpu_queue``) already provide
equivalent double-buffering natively.  ``dali_cpu_queue`` was raised to
≥ 16 to compensate.

Coverage
--------
- AsyncPrefetchIterator is absent from memory.py and loader.py
- dali_cpu_queue default is ≥ 16 in LoaderConfig
- _raw_iter iterates directly over self._dali_iter (no Future/executor layer)
- End-to-end: multiple epochs produce valid Batch objects (CPU backend)
- Batch content: global_crops shape, dtype, values finite
- set_epoch between epochs does not raise or corrupt state
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class TestAsyncPrefetchIteratorRemoved:

    def test_not_in_memory_module(self):
        import dino_loader.memory as m
        assert not hasattr(m, "AsyncPrefetchIterator"), (
            "AsyncPrefetchIterator still exists in memory.py — it should be removed."
        )

    def test_not_referenced_in_loader(self):
        loader_path = Path(_SRC) / "dino_loader" / "loader.py"
        assert "AsyncPrefetchIterator" not in loader_path.read_text(), (
            "loader.py still references AsyncPrefetchIterator."
        )


class TestDALICpuQueueSize:

    def test_default_dali_cpu_queue_at_least_16(self):
        from dino_loader.config import LoaderConfig
        cfg = LoaderConfig()
        assert cfg.dali_cpu_queue >= 16, (
            f"dali_cpu_queue={cfg.dali_cpu_queue} is insufficient; "
            "set to ≥ 16 to replace AsyncPrefetchIterator buffering."
        )

    def test_custom_cpu_queue_accepted(self):
        from dino_loader.config import LoaderConfig
        cfg = LoaderConfig(dali_cpu_queue=32)
        assert cfg.dali_cpu_queue == 32


class TestRawIterIsSimpleLoop:

    def test_no_future_or_executor_in_raw_iter(self):
        from dino_loader.loader import DINODataLoader
        src = inspect.getsource(DINODataLoader._raw_iter)
        assert "Future" not in src
        assert "ThreadPoolExecutor" not in src
        assert "executor" not in src.lower()

    def test_raw_iter_drives_dali_iter_directly(self):
        from dino_loader.loader import DINODataLoader
        src = inspect.getsource(DINODataLoader._raw_iter)
        assert "self._dali_iter" in src


class TestEndToEndMultipleEpochs:

    def _make_loader(self, tmp_path: Path):
        from dino_datasets import DatasetSpec

        from dino_loader.config import DINOAugConfig, LoaderConfig
        from dino_loader.loader import DINODataLoader
        from tests.fixtures import scaffold_dataset_dir

        tar_paths = scaffold_dataset_dir(
            root=tmp_path, n_shards=2, n_samples_per_shard=8,
        )
        return DINODataLoader(
            specs=[DatasetSpec(name="ds", shards=tar_paths, weight=1.0)],
            batch_size=4,
            aug_cfg=DINOAugConfig(
                global_crop_size=32,
                local_crop_size=16,
                n_global_crops=2,
                n_local_crops=2,
            ),
            config=LoaderConfig(
                node_shm_gb=0.1,
                stall_timeout_s=0,
                stateful_dataloader=False,
                checkpoint_dir=str(tmp_path / "ckpt"),
            ),
            backend="cpu",
        )

    def test_multiple_epochs_produce_valid_batches(self, tmp_path):
        from dino_loader.memory import Batch

        loader = self._make_loader(tmp_path)
        for epoch in range(2):
            loader.set_epoch(epoch)
            batch = next(iter(loader))
            assert isinstance(batch, Batch)
            assert len(batch.global_crops) == 2
            assert len(batch.local_crops)  == 2
            assert batch.global_crops[0].shape == (4, 3, 32, 32)
            assert batch.local_crops[0].shape  == (4, 3, 16, 16)

    def test_global_crop_dtype_is_float(self, tmp_path):
        loader = self._make_loader(tmp_path)
        loader.set_epoch(0)
        batch = next(iter(loader))
        for crop in batch.global_crops:
            assert crop.dtype in (torch.float32, torch.float16, torch.bfloat16), (
                f"Unexpected dtype {crop.dtype} — expected a float type."
            )

    def test_global_crop_values_are_finite(self, tmp_path):
        loader = self._make_loader(tmp_path)
        loader.set_epoch(0)
        batch = next(iter(loader))
        for crop in batch.global_crops + batch.local_crops:
            assert torch.isfinite(crop).all(), (
                f"Non-finite values found in crop of shape {crop.shape}."
            )

    def test_set_epoch_between_epochs_does_not_raise(self, tmp_path):
        """set_epoch() called before each epoch must not raise or corrupt the loader."""
        loader = self._make_loader(tmp_path)
        for epoch in range(3):
            loader.set_epoch(epoch)
            # Consuming at least one batch exercises the full pipeline.
            batch = next(iter(loader))
            assert batch.global_crops[0].shape[0] == 4  # batch dimension intact

    def test_metadata_length_matches_batch_size(self, tmp_path):
        """batch.metadata must have one entry per sample."""
        loader = self._make_loader(tmp_path)
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert len(batch.metadata) == 4

    def test_masks_none_without_mask_generator(self, tmp_path):
        """Without a MaskingGenerator, batch.masks must be None."""
        loader = self._make_loader(tmp_path)
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert batch.masks is None
