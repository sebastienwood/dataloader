"""tests/test_loader_config.py
==============================
Unit tests for config dataclasses and checkpoint serialization.

Coverage
--------
LoaderConfig
- Default field values
- Validation: output_dtype, hw_decoder_load, shm_warn_threshold,
  stall_timeout_s, heartbeat_stale_s, prometheus_port
- checkpoint_dir obligatoire si stateful_dataloader=True [CFG-CKPT]
- FP8 requires transformer-engine at construction time [CFG-B4]
- dali_cpu_queue default ≥ 16
- adaptive_prefetch validation [ARCH2]
- shard_extraction_workers field removed

SharedExtractionPoolConfig
- Valeurs par défaut
- Validation max_workers > 0

PipelineConfig
- from_loader_config() produit les bons champs
- Seed offset par rank

CheckpointState — pure dataclass (no I/O)
- to_dict / from_dict round-trip
- Missing fields default gracefully

save_checkpoint / load_checkpoint (from checkpoint.py)
- SHA-256 envelope round-trip [M3]
- Tampered file raises ValueError
- Backward-compatible flat format
- Atomic write via .tmp → rename
- save_checkpoint(path, state) — path is always first argument
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.checkpoint import load_checkpoint, save_checkpoint
from dino_loader.config import (
    CheckpointState,
    LoaderConfig,
    PipelineConfig,
    SharedExtractionPoolConfig,
)


# ══════════════════════════════════════════════════════════════════════════════
# LoaderConfig — defaults
# ══════════════════════════════════════════════════════════════════════════════


class TestLoaderConfigDefaults:

    def test_node_shm_gb(self):
        assert LoaderConfig().node_shm_gb == 128.0

    def test_shuffle_buffer_size(self):
        assert LoaderConfig().shuffle_buffer_size == 512

    def test_stateful_dataloader_disabled_by_default(self):
        """[FIX-STATEFUL] Default is False so LoaderConfig() works without checkpoint_dir."""
        cfg = LoaderConfig()
        assert cfg.stateful_dataloader is False

    def test_output_dtype_bf16(self):
        assert LoaderConfig().output_dtype == "bf16"

    def test_seed_default(self):
        assert LoaderConfig().seed == 0

    def test_dali_cpu_queue_at_least_16(self):
        assert LoaderConfig().dali_cpu_queue >= 16

    def test_heartbeat_stale_s_default(self):
        assert LoaderConfig().heartbeat_stale_s == 300.0

    def test_prometheus_port_disabled_by_default(self):
        assert LoaderConfig().prometheus_port is None

    def test_no_shard_extraction_workers_field(self):
        """shard_extraction_workers has been removed — use extraction_pool instead."""
        cfg = LoaderConfig()
        assert not hasattr(cfg, "shard_extraction_workers"), (
            "shard_extraction_workers must be removed; use extraction_pool.max_workers"
        )


# ══════════════════════════════════════════════════════════════════════════════
# LoaderConfig — validation
# ══════════════════════════════════════════════════════════════════════════════


class TestLoaderConfigValidation:

    def test_invalid_output_dtype_raises(self):
        with pytest.raises(ValueError, match="output_dtype"):
            LoaderConfig(output_dtype="int8")

    def test_valid_dtype_bf16(self):
        LoaderConfig(output_dtype="bf16")

    def test_valid_dtype_fp32(self):
        LoaderConfig(output_dtype="fp32")

    def test_hw_decoder_load_above_one_raises(self):
        with pytest.raises(ValueError):
            LoaderConfig(hw_decoder_load=1.5)

    def test_hw_decoder_load_at_boundaries_valid(self):
        LoaderConfig(hw_decoder_load=0.0)
        LoaderConfig(hw_decoder_load=1.0)

    def test_shm_warn_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            LoaderConfig(shm_warn_threshold=1.5)

    def test_shm_warn_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            LoaderConfig(shm_warn_threshold=-0.1)

    def test_stall_timeout_negative_raises(self):
        with pytest.raises(ValueError, match="stall_timeout_s"):
            LoaderConfig(stall_timeout_s=-1.0)

    def test_stall_timeout_zero_is_valid(self):
        LoaderConfig(stall_timeout_s=0.0)

    def test_heartbeat_stale_zero_raises(self):
        with pytest.raises(ValueError, match="heartbeat_stale_s"):
            LoaderConfig(heartbeat_stale_s=0.0)

    def test_dali_fp8_without_use_fp8_raises(self):
        with pytest.raises(ValueError):
            LoaderConfig(dali_fp8_output=True, use_fp8_output=False)

    def test_stateful_without_checkpoint_dir_raises(self):
        with pytest.raises(ValueError, match="checkpoint_dir"):
            LoaderConfig(stateful_dataloader=True, checkpoint_dir="")

    def test_stateful_with_checkpoint_dir_valid(self, tmp_path):
        cfg = LoaderConfig(stateful_dataloader=True, checkpoint_dir=str(tmp_path / "ckpt"))
        assert cfg.stateful_dataloader is True

    def test_non_stateful_without_checkpoint_dir_valid(self):
        # [FIX-STATEFUL] This is now the default — must not raise.
        cfg = LoaderConfig(stateful_dataloader=False, checkpoint_dir="")
        assert not cfg.stateful_dataloader

    def test_default_loader_config_does_not_raise(self):
        """LoaderConfig() with no arguments must work (stateful_dataloader=False by default)."""
        cfg = LoaderConfig()
        assert not cfg.stateful_dataloader
        assert cfg.checkpoint_dir == ""


# ══════════════════════════════════════════════════════════════════════════════
# LoaderConfig — FP8 requires TE
# ══════════════════════════════════════════════════════════════════════════════


class TestFP8RequiresTE:

    def test_fp8_without_te_raises_at_construction(self):
        with patch.dict("sys.modules", {"transformer_engine": None,
                                        "transformer_engine.pytorch": None}):
            with pytest.raises(ValueError, match="transformer-engine"):
                LoaderConfig(use_fp8_output=True)

    def test_fp8_false_does_not_require_te(self):
        with patch.dict("sys.modules", {"transformer_engine": None,
                                        "transformer_engine.pytorch": None}):
            cfg = LoaderConfig(use_fp8_output=False)
            assert cfg.use_fp8_output is False


# ══════════════════════════════════════════════════════════════════════════════
# SharedExtractionPoolConfig
# ══════════════════════════════════════════════════════════════════════════════


class TestSharedExtractionPoolConfig:

    def test_default_max_workers(self):
        assert SharedExtractionPoolConfig().max_workers == 16

    def test_default_queue_depth(self):
        assert SharedExtractionPoolConfig().queue_depth_per_shard == 256

    def test_max_workers_zero_raises(self):
        with pytest.raises(ValueError, match="max_workers"):
            SharedExtractionPoolConfig(max_workers=0)

    def test_queue_depth_zero_raises(self):
        with pytest.raises(ValueError, match="queue_depth_per_shard"):
            SharedExtractionPoolConfig(queue_depth_per_shard=0)

    def test_custom_values_valid(self):
        cfg = SharedExtractionPoolConfig(max_workers=8, queue_depth_per_shard=128)
        assert cfg.max_workers == 8

    def test_loader_config_has_extraction_pool(self):
        cfg = LoaderConfig()
        assert isinstance(cfg.extraction_pool, SharedExtractionPoolConfig)


# ══════════════════════════════════════════════════════════════════════════════
# PipelineConfig
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineConfig:

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.num_threads     == 8
        assert cfg.device_id       == 0
        assert cfg.hw_decoder_load == 0.90
        assert cfg.cpu_queue       == 16
        assert cfg.gpu_queue       == 6
        assert cfg.seed            == 0
        assert cfg.fuse_normalization is True
        assert cfg.dali_fp8_output    is False

    def test_from_loader_config(self, tmp_path):
        loader_cfg = LoaderConfig(
            dali_num_threads   = 4,
            hw_decoder_load    = 0.75,
            dali_cpu_queue     = 20,
            dali_gpu_queue     = 8,
            seed               = 42,
            fuse_normalization = False,
            stateful_dataloader = True,
            checkpoint_dir      = str(tmp_path / "ckpt"),
        )
        cfg = PipelineConfig.from_loader_config(cfg=loader_cfg, device_id=2, rank=3)
        assert cfg.num_threads     == 4
        assert cfg.device_id       == 2
        assert cfg.hw_decoder_load == 0.75
        assert cfg.cpu_queue       == 20
        assert cfg.gpu_queue       == 8
        assert cfg.seed            == 42 + 3
        assert cfg.fuse_normalization is False

    def test_seed_offset_by_rank(self, tmp_path):
        loader_cfg = LoaderConfig(
            seed=100, stateful_dataloader=True,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cfg0 = PipelineConfig.from_loader_config(cfg=loader_cfg, device_id=0, rank=0)
        cfg1 = PipelineConfig.from_loader_config(cfg=loader_cfg, device_id=1, rank=1)
        assert cfg0.seed == 100
        assert cfg1.seed == 101

    def test_is_frozen(self):
        cfg = PipelineConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.seed = 999  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# LoaderConfig — adaptive prefetch
# ══════════════════════════════════════════════════════════════════════════════


class TestAdaptivePrefetch:

    def test_default_disabled(self):
        assert LoaderConfig().adaptive_prefetch is False

    def test_enable_with_valid_target(self):
        cfg = LoaderConfig(
            adaptive_prefetch=True,
            adaptive_prefetch_target_util=0.80,
        )
        assert cfg.adaptive_prefetch is True

    def test_target_zero_raises(self):
        with pytest.raises(ValueError, match="adaptive_prefetch_target_util"):
            LoaderConfig(
                adaptive_prefetch=True,
                adaptive_prefetch_target_util=0.0,
            )

    def test_target_above_one_raises(self):
        with pytest.raises(ValueError, match="adaptive_prefetch_target_util"):
            LoaderConfig(
                adaptive_prefetch=True,
                adaptive_prefetch_target_util=1.1,
            )


# ══════════════════════════════════════════════════════════════════════════════
# CheckpointState — pur dataclass (sans méthodes I/O)
# ══════════════════════════════════════════════════════════════════════════════


def _make_state(step: int = 100, epoch: int = 2) -> CheckpointState:
    return CheckpointState(
        step=step, epoch=epoch,
        dataset_names=["laion", "imagenet"],
        mixing_weights=[0.7, 0.3],
        global_crop_size=224, local_crop_size=96,
    )


class TestCheckpointStateDataclass:

    def test_to_dict_contains_all_fields(self):
        d = _make_state().to_dict()
        for k in ("step", "epoch", "dataset_names", "mixing_weights",
                  "global_crop_size", "local_crop_size"):
            assert k in d

    def test_from_dict_round_trip(self):
        state = _make_state(step=7, epoch=3)
        d     = state.to_dict()
        restored = CheckpointState.from_dict(d)
        assert restored.step  == 7
        assert restored.epoch == 3

    def test_from_dict_ignores_unknown_keys(self):
        d = _make_state().to_dict()
        d["unknown_key"] = "ignored"
        restored = CheckpointState.from_dict(d)
        assert restored.step == 100


# ══════════════════════════════════════════════════════════════════════════════
# save_checkpoint / load_checkpoint — path est toujours le premier argument
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointIO:

    def test_save_and_load_step(self, tmp_path):
        path = tmp_path / "dl_state.json"
        save_checkpoint(path, _make_state(step=42))
        assert load_checkpoint(path).step == 42

    def test_save_and_load_epoch(self, tmp_path):
        path = tmp_path / "dl_state.json"
        save_checkpoint(path, _make_state(epoch=5))
        assert load_checkpoint(path).epoch == 5

    def test_save_and_load_crop_sizes(self, tmp_path):
        path = tmp_path / "dl_state.json"
        save_checkpoint(path, _make_state())
        loaded = load_checkpoint(path)
        assert loaded.global_crop_size == 224
        assert loaded.local_crop_size  == 96

    def test_saved_file_is_valid_json(self, tmp_path):
        path = tmp_path / "dl_state.json"
        save_checkpoint(path, _make_state())
        data = json.loads(path.read_text())
        assert "payload" in data or "step" in data

    def test_envelope_sha256(self, tmp_path):
        path = tmp_path / "state.json"
        save_checkpoint(path, _make_state())
        raw      = json.loads(path.read_text())
        computed = hashlib.sha256(json.dumps(raw["payload"], indent=2, sort_keys=True).encode()).hexdigest()
        assert raw["sha256"] == computed

    def test_tampered_raises(self, tmp_path):
        path = tmp_path / "state.json"
        save_checkpoint(path, _make_state())
        raw = json.loads(path.read_text())
        raw["payload"]["step"] = 9999
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="integrity check"):
            load_checkpoint(path)

    def test_missing_crop_sizes_default_to_224_96(self, tmp_path):
        path = tmp_path / "old_state.json"
        path.write_text(json.dumps({
            "step": 50, "epoch": 1,
            "dataset_names": ["laion"],
            "mixing_weights": [1.0],
        }))
        loaded = load_checkpoint(path)
        assert loaded.global_crop_size == 224
        assert loaded.local_crop_size  == 96

    def test_path_is_first_argument(self, tmp_path):
        """Verify canonical call signature: save_checkpoint(path, state)."""
        path  = tmp_path / "sig_test.json"
        state = _make_state(step=7)
        # Positional call — path first, state second.
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)
        assert loaded.step == 7