"""tests/test_loader_cpu.py
========================
End-to-end integration tests for DINODataLoader using the CPU backend.

DINODataLoader.__iter__ delegates to an internal NodePipeline.
DINODataLoader._reader is a ShardReaderNode.
DINODataLoader.as_pipeline() returns the NodePipeline for composition.
DINODataLoader.map/select/with_epoch are composition shortcuts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_datasets import DatasetSpec

from dino_loader.backends import get_backend
from dino_loader.backends.cpu import CPUBackend
from dino_loader.config import DINOAugConfig, LoaderConfig
from dino_loader.loader import DINODataLoader
from dino_loader.memory import Batch
from dino_loader.pipeline_graph import NodePipeline
from dino_loader.shard_reader import ShardReaderNode
from tests.conftest import make_spec


def _cfg(tmp_path, stateful=False):
    return LoaderConfig(
        node_shm_gb=0.1, shard_prefetch_window=2, shard_extraction_workers=2,
        shuffle_buffer_size=4, use_fp8_output=False,
        stateful_dataloader=stateful,
        checkpoint_dir=str(tmp_path / "ckpt") if stateful else "",
        checkpoint_every_steps=2, stall_timeout_s=0,
    )


def _loader(tar_paths, tmp_path, aug_cfg=None, loader_cfg=None, batch_size=4, **spec_kw):
    spec = make_spec("test_ds", tar_paths, **spec_kw)
    return DINODataLoader(
        specs=[spec], batch_size=batch_size, aug_cfg=aug_cfg,
        config=loader_cfg or _cfg(tmp_path), backend="cpu",
    )


# ═══════════════════ Basic ═══════════════════


class TestLoaderBasic:

    def test_yields_batch(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert isinstance(batch, Batch)

    def test_global_crop_shape(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, batch_size=4)
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert len(batch.global_crops) == small_aug_cfg.n_global_crops
        for c in batch.global_crops:
            assert c.shape == (4, 3, 32, 32)

    def test_local_crop_shape(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, batch_size=4)
        loader.set_epoch(0)
        batch = next(iter(loader))
        for c in batch.local_crops:
            assert c.shape == (4, 3, 16, 16)

    def test_batch_dtype_float(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        for c in batch.global_crops + batch.local_crops:
            assert c.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_batch_values_finite(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        for c in batch.global_crops + batch.local_crops:
            assert torch.isfinite(c).all()

    def test_metadata_length(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, batch_size=4)
        loader.set_epoch(0)
        assert len(next(iter(loader)).metadata) == 4

    def test_masks_none_by_default(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        loader.set_epoch(0)
        assert next(iter(loader)).masks is None

    def test_iter_unpack(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        loader.set_epoch(0)
        batch = next(iter(loader))
        g, l = batch
        assert g is batch.global_crops and l is batch.local_crops

    def test_backend_name(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        assert _loader(paths, tmp_path, aug_cfg=small_aug_cfg).backend.name == "cpu"


# ═══════════════════ Internal architecture ═══════════════════


class TestLoaderInternalArchitecture:
    """Verify the internal structure introduced by Phase 1/3 migration."""

    def test_reader_is_shard_reader_node(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        assert isinstance(loader._reader, ShardReaderNode)

    def test_pipeline_is_node_pipeline(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        assert isinstance(loader._pipeline, NodePipeline)

    def test_as_pipeline_returns_node_pipeline(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        assert isinstance(_loader(paths, tmp_path, aug_cfg=small_aug_cfg).as_pipeline(), NodePipeline)

    def test_set_epoch_updates_reader_epoch(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        loader.set_epoch(5)
        assert loader._reader._epoch == 5

    def test_dali_node_exists(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        """_dali_node is required for wrap_loader and NodePipeline composition."""
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        assert hasattr(loader, "_dali_node")


# ═══════════════════ Composition API ═══════════════════


class TestLoaderComposition:

    def test_map_returns_node_pipeline(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        assert isinstance(_loader(paths, tmp_path, aug_cfg=small_aug_cfg).map(lambda b: b), NodePipeline)

    def test_select_returns_node_pipeline(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        assert isinstance(_loader(paths, tmp_path, aug_cfg=small_aug_cfg).select(lambda b: True), NodePipeline)

    def test_with_epoch_returns_node_pipeline(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        assert isinstance(_loader(paths, tmp_path, aug_cfg=small_aug_cfg).with_epoch(10), NodePipeline)

    def test_composition_does_not_mutate_loader(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        pipeline = loader.map(lambda b: b)

        loader.set_epoch(0)
        assert isinstance(next(iter(loader)), Batch)

        pipeline.set_epoch(0)
        assert isinstance(next(iter(pipeline)), Batch)

    def test_map_applied_to_every_batch(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        counter = [0]

        def _count(b):
            counter[0] += 1
            return b

        pipeline = (
            _loader(paths, tmp_path, aug_cfg=small_aug_cfg, batch_size=4)
            .map(_count)
            .with_epoch(2)
        )
        pipeline.set_epoch(0)
        assert len(list(pipeline)) == 2
        assert counter[0] == 2

    def test_pipeline_exposes_loader_properties(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        pipeline = loader.as_pipeline()
        assert pipeline.current_resolution == loader.current_resolution
        assert pipeline.current_weights == loader.current_weights
        assert pipeline.backend.name == "cpu"
        assert pipeline.aug_spec is loader.aug_spec

    def test_select_filters_batches(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        """select() must pass through only matching batches."""
        _, paths = tmp_dataset_dir
        accepted: list[Batch] = []
        rejected_count = [0]

        def _track(b: Batch) -> Batch:
            accepted.append(b)
            return b

        toggle = [0]

        def _predicate(b: Batch) -> bool:
            result = toggle[0] % 2 == 0
            toggle[0] += 1
            return result

        pipeline = (
            _loader(paths, tmp_path, aug_cfg=small_aug_cfg, batch_size=4)
            .select(_predicate)
            .map(_track)
            .with_epoch(2)
        )
        pipeline.set_epoch(0)
        list(pipeline)
        assert len(accepted) == 2


# ═══════════════════ Multi-dataset ═══════════════════


class TestLoaderMultiDataset:

    def test_two_datasets_yields_batch(self, multi_dataset_dirs, small_aug_cfg, tmp_path):
        _, datasets = multi_dataset_dirs
        specs = [
            make_spec("alpha", datasets["alpha"], weight=0.6),
            make_spec("beta",  datasets["beta"],  weight=0.4),
        ]
        loader = DINODataLoader(specs=specs, batch_size=4, aug_cfg=small_aug_cfg,
                                config=_cfg(tmp_path), backend="cpu")
        loader.set_epoch(0)
        assert isinstance(next(iter(loader)), Batch)

    def test_set_weights_normalises(self, multi_dataset_dirs, small_aug_cfg, tmp_path):
        _, datasets = multi_dataset_dirs
        specs = [make_spec("a", datasets["alpha"]), make_spec("b", datasets["beta"])]
        loader = DINODataLoader(specs=specs, batch_size=4, aug_cfg=small_aug_cfg,
                                config=_cfg(tmp_path), backend="cpu")
        loader.set_weights([3.0, 1.0])
        w = loader.current_weights
        assert abs(w[0] - 0.75) < 1e-5 and abs(w[1] - 0.25) < 1e-5

    def test_set_weight_by_name(self, multi_dataset_dirs, small_aug_cfg, tmp_path):
        _, datasets = multi_dataset_dirs
        specs = [make_spec("alpha", datasets["alpha"]), make_spec("beta", datasets["beta"])]
        loader = DINODataLoader(specs=specs, batch_size=4, aug_cfg=small_aug_cfg,
                                config=_cfg(tmp_path), backend="cpu")
        loader.set_weight_by_name("alpha", 9.0)
        assert loader.current_weights[0] > loader.current_weights[1]


# ═══════════════════ Epoch control ═══════════════════


class TestLoaderEpochControl:

    def test_set_epoch_does_not_raise(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        loader.set_epoch(0)
        loader.set_epoch(1)

    def test_steps_per_epoch_len(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        spec = make_spec("ds", paths)
        loader = DINODataLoader(specs=[spec], batch_size=4, aug_cfg=small_aug_cfg,
                                config=_cfg(tmp_path), backend="cpu", steps_per_epoch=100)
        assert len(loader) == 100

    def test_len_without_steps_raises(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        with pytest.raises(TypeError):
            _ = len(_loader(paths, tmp_path, aug_cfg=small_aug_cfg))


# ═══════════════════ Resolution ═══════════════════


class TestLoaderResolution:

    def test_set_resolution_changes_output(self, tmp_dataset_dir, tmp_path):
        _, paths = tmp_dataset_dir
        aug_cfg = DINOAugConfig(global_crop_size=32, local_crop_size=16,
                                max_global_crop_size=64, max_local_crop_size=32,
                                n_global_crops=2, n_local_crops=2)
        loader = _loader(paths, tmp_path, aug_cfg=aug_cfg, batch_size=2)
        loader.set_epoch(0)
        loader.set_resolution(64, 32)
        batch = next(iter(loader))
        for c in batch.global_crops:
            assert c.shape[-2:] == (64, 64)

    def test_set_resolution_above_max_raises(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        with pytest.raises(ValueError, match="max_global_crop_size"):
            _loader(paths, tmp_path, aug_cfg=small_aug_cfg).set_resolution(999, 64)

    def test_resolution_schedule(self, tmp_dataset_dir, tmp_path):
        _, paths = tmp_dataset_dir
        aug_cfg = DINOAugConfig(global_crop_size=32, local_crop_size=16,
                                max_global_crop_size=64, max_local_crop_size=32,
                                n_global_crops=2, n_local_crops=2,
                                resolution_schedule=[(0, 32), (1, 64)])
        loader = _loader(paths, tmp_path, aug_cfg=aug_cfg, batch_size=2)
        loader.set_epoch(0)
        assert loader._current_global_size == 32
        loader.set_epoch(1)
        assert loader._current_global_size == 64

    def test_current_resolution_property(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg)
        g, l = loader.current_resolution
        assert g == small_aug_cfg.global_crop_size
        assert l == small_aug_cfg.local_crop_size


# ═══════════════════ Checkpointing ═══════════════════


class TestLoaderCheckpoint:

    def test_state_dict_keys(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, loader_cfg=_cfg(tmp_path, stateful=True))
        sd = loader.state_dict()
        for k in ("step", "epoch", "dataset_names", "mixing_weights", "global_crop_size", "local_crop_size"):
            assert k in sd

    def test_round_trip(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        cfg = _cfg(tmp_path, stateful=True)
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, loader_cfg=cfg)
        loader.set_epoch(3)
        loader._step = 42
        loader.set_resolution(32, 16)
        sd = loader.state_dict()

        loader2 = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, loader_cfg=cfg)
        loader2.load_state_dict(sd)
        assert loader2._epoch == 3 and loader2._current_global_size == 32

    def test_checkpoint_saves_json(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        ckpt_dir = str(tmp_path / "ckpts")
        cfg = LoaderConfig(node_shm_gb=0.1, shard_prefetch_window=2, shard_extraction_workers=2,
                           shuffle_buffer_size=4, use_fp8_output=False, stateful_dataloader=True,
                           checkpoint_dir=ckpt_dir, checkpoint_every_steps=2)
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, loader_cfg=cfg)
        loader.checkpoint(step=2)
        files = list(Path(ckpt_dir).glob("dl_state_*.json"))
        assert len(files) == 1
        raw = json.load(open(files[0]))
        # Supports both envelope format and legacy flat format.
        payload = raw.get("payload", raw)
        assert payload["step"] == 2

    def test_state_dict_no_disk_io(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, loader_cfg=_cfg(tmp_path, stateful=True))
        assert loader.state_dict()["step"] == 0
        loader.checkpoint(step=2)
        assert loader.state_dict()["step"] == 2

    def test_state_dict_requires_stateful(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, loader_cfg=_cfg(tmp_path, stateful=False))
        with pytest.raises(RuntimeError, match="stateful_dataloader"):
            loader.state_dict()

    def test_resume(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        ckpt_dir = str(tmp_path / "resume")
        cfg = LoaderConfig(node_shm_gb=0.1, shard_prefetch_window=2, shard_extraction_workers=2,
                           shuffle_buffer_size=4, use_fp8_output=False, stateful_dataloader=True,
                           checkpoint_dir=ckpt_dir, checkpoint_every_steps=4)
        _loader(paths, tmp_path, aug_cfg=small_aug_cfg, loader_cfg=cfg).checkpoint(step=4)
        loader2 = DINODataLoader(specs=[make_spec("test_ds", paths)], batch_size=4,
                                 aug_cfg=small_aug_cfg, config=cfg, backend="cpu", resume=True)
        assert loader2._step == 4

    def test_pipeline_state_dict(self, tmp_dataset_dir, small_aug_cfg, tmp_path):
        _, paths = tmp_dataset_dir
        loader = _loader(paths, tmp_path, aug_cfg=small_aug_cfg, loader_cfg=_cfg(tmp_path, stateful=True))
        pipeline = loader.as_pipeline()
        pipeline.set_epoch(0)
        next(iter(pipeline))
        sd = pipeline.state_dict()
        assert "loader" in sd and "tn_graph" in sd


# ═══════════════════ Quality filtering ═══════════════════


class TestLoaderQualityFilter:

    def test_min_quality_filters(self, shard_with_low_quality, small_aug_cfg, tmp_path):
        tar_path, _, _ = shard_with_low_quality
        spec = DatasetSpec(name="f", shards=[tar_path], weight=1.0, min_sample_quality=0.5, metadata_key="json")
        loader = DINODataLoader(specs=[spec], batch_size=2, aug_cfg=small_aug_cfg,
                                config=_cfg(tmp_path), backend="cpu")
        loader.set_epoch(0)
        assert isinstance(next(iter(loader)), Batch)

    def test_no_metadata_key(self, shard_without_metadata, small_aug_cfg, tmp_path):
        tar_path, _ = shard_without_metadata
        spec = DatasetSpec(name="nm", shards=[tar_path], weight=1.0, metadata_key=None)
        loader = DINODataLoader(specs=[spec], batch_size=2, aug_cfg=small_aug_cfg,
                                config=_cfg(tmp_path), backend="cpu")
        loader.set_epoch(0)
        batch = next(iter(loader))
        assert all(m is None for m in batch.metadata)


# ═══════════════════ Backend switch ═══════════════════


class TestLoaderBackendSwitch:

    def test_get_backend_cpu(self):
        b = get_backend("cpu")
        assert isinstance(b, CPUBackend) and b.name == "cpu"

    def test_get_backend_auto_cpu(self):
        try:
            import nvidia.dali  # noqa: F401
            pytest.skip("DALI installed")
        except ImportError:
            pass
        assert get_backend("auto").name == "cpu"

    def test_dali_raises_without_dali(self):
        try:
            import nvidia.dali  # noqa: F401
            pytest.skip("DALI installed")
        except ImportError:
            pass
        from dino_loader.backends.dali_backend import DALIBackend
        from dino_loader.config import PipelineConfig
        with pytest.raises(RuntimeError, match="nvidia-dali"):
            DALIBackend().build_pipeline(None, None, PipelineConfig())