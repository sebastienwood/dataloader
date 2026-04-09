"""tests/test_loader_concurrency.py
====================================
Tests for concurrency safety and initialization preconditions in
:class:`dino_loader.loader.DINODataLoader`.

Coverage
--------
set_epoch thread safety [B3]
- Concurrent calls to set_epoch do not raise or corrupt state

__iter__ double-call guard [FIX-ACTIVE-ITER]
- Starting a second __iter__ while already iterating raises RuntimeError

dino_env.init() precondition [FIX-ENV]
- DALI backend requires dino_env.init() before construction
- CPU backend has no such requirement
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_datasets import DatasetSpec

from dino_loader.config import DINOAugConfig, LoaderConfig
from dino_loader.loader import DINODataLoader
from tests.fixtures import scaffold_dataset_dir


def _small_loader(tmp_path: Path) -> DINODataLoader:
    tar_paths = scaffold_dataset_dir(root=tmp_path, n_shards=2, n_samples_per_shard=4)
    return DINODataLoader(
        specs=[DatasetSpec(name="ds", shards=tar_paths, weight=1.0)],
        batch_size=2,
        aug_cfg=DINOAugConfig(
            global_crop_size=32, local_crop_size=16,
            n_global_crops=2, n_local_crops=2,
        ),
        config=LoaderConfig(
            node_shm_gb=0.1,
            stall_timeout_s=0,
            stateful_dataloader=False,
            checkpoint_dir="",
        ),
        backend="cpu",
    )


class TestSetEpochThreadSafety:
    """Concurrent set_epoch calls must not raise or corrupt state [B3]."""

    def test_concurrent_set_epoch_no_errors(self, tmp_path):
        loader = _small_loader(tmp_path)
        errors: list[Exception] = []

        def _call(epoch: int) -> None:
            try:
                loader.set_epoch(epoch)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_call, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent set_epoch: {errors}"


class TestDoubleIterGuard:
    """Starting __iter__ while already iterating must raise [FIX-ACTIVE-ITER]."""

    def test_double_iter_raises_runtime_error(self, tmp_path):
        loader = _small_loader(tmp_path)
        loader.set_epoch(0)

        # Manually set the guard flag — this is the exact internal mechanism.
        with loader._active_iter_lock:
            loader._active_iter = True

        try:
            with pytest.raises(RuntimeError, match="déjà en cours d'itération"):
                iter(loader)
        finally:
            # Clean up so the loader doesn't stay locked.
            with loader._active_iter_lock:
                loader._active_iter = False

    def test_iter_resets_guard_after_completion(self, tmp_path):
        """After iteration ends, the guard must be cleared for re-use."""
        loader = _small_loader(tmp_path)
        loader.set_epoch(0)

        # Consume one batch only — then stop early.
        it = loader.__iter__()
        try:
            next(it)
        except StopIteration:
            pass

        # Force exit from the generator to trigger the finally block.
        it.close()

        with loader._active_iter_lock:
            assert not loader._active_iter, "Guard was not cleared after iteration ended"


class TestDALIBackendPrecondition:
    """DALI backend requires dino_env.init() before construction [FIX-ENV]."""

    def test_dali_backend_without_env_init_raises(self):
        import dino_env
        with patch.object(dino_env, "get_env", side_effect=RuntimeError("not initialised")):
            with pytest.raises(RuntimeError, match="dino_env.init"):
                DINODataLoader(
                    specs=[DatasetSpec(name="ds", shards=["x.tar"], weight=1.0)],
                    batch_size=4,
                    backend="dali",
                )

    def test_cpu_backend_does_not_require_env_init(self, tmp_path):
        """CPU backend never calls dino_env.get_env()."""
        loader = _small_loader(tmp_path)
        assert loader._backend.name == "cpu"