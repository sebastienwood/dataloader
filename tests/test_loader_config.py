"""tests/test_loader_config.py
==============================
Unit tests for :class:`dino_loader.config.LoaderConfig` and
:class:`dino_loader.config.CheckpointState`.

Coverage
--------
LoaderConfig
- Default field values
- Validation: output_dtype, hw_decoder_load, shm_warn_threshold,
  stall_timeout_s, heartbeat_stale_s, prometheus_port
- FP8 requires transformer-engine at construction time [CFG-B4]
- dali_cpu_queue default ≥ 16 (post-AsyncPrefetchIterator removal)
- adaptive_prefetch validation [ARCH2]
- prometheus_port validation [ARCH3]

CheckpointState
- save / load round-trip with SHA-256 envelope [M3]
- Checksum verification rejects tampered files
- Backward-compatible flat format (no checksum)
- Atomic write via .tmp → rename
- Missing crop-size fields default gracefully
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

from dino_loader.config import CheckpointState, LoaderConfig

# ══════════════════════════════════════════════════════════════════════════════
# LoaderConfig
# ══════════════════════════════════════════════════════════════════════════════


class TestLoaderConfigDefaults:

    def test_node_shm_gb(self):
        assert LoaderConfig().node_shm_gb == 128.0

    def test_shuffle_buffer_size(self):
        assert LoaderConfig().shuffle_buffer_size == 512

    def test_stateful_dataloader_enabled_by_default(self):
        assert LoaderConfig().stateful_dataloader is True

    def test_output_dtype_bf16(self):
        assert LoaderConfig().output_dtype == "bf16"

    def test_seed_default(self):
        assert LoaderConfig().seed == 0

    def test_dali_cpu_queue_at_least_16(self):
        """dali_cpu_queue must be ≥ 16 after AsyncPrefetchIterator removal."""
        assert LoaderConfig().dali_cpu_queue >= 16

    def test_heartbeat_stale_s_default(self):
        assert LoaderConfig().heartbeat_stale_s == 300.0

    def test_prometheus_port_disabled_by_default(self):
        assert LoaderConfig().prometheus_port is None


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

    def test_heartbeat_stale_negative_raises(self):
        with pytest.raises(ValueError, match="heartbeat_stale_s"):
            LoaderConfig(heartbeat_stale_s=-1.0)

    def test_heartbeat_stale_custom_value(self):
        cfg = LoaderConfig(heartbeat_stale_s=600.0)
        assert cfg.heartbeat_stale_s == 600.0

    def test_dali_fp8_without_use_fp8_raises(self):
        with pytest.raises(ValueError):
            LoaderConfig(dali_fp8_output=True, use_fp8_output=False)


class TestFP8RequiresTE:
    """FP8 output requires transformer-engine at construction time [CFG-B4]."""

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


class TestAdaptivePrefetch:
    """Adaptive prefetch flag and PID controller config [ARCH2]."""

    def test_default_disabled(self):
        assert LoaderConfig().adaptive_prefetch is False

    def test_enable_with_valid_target(self):
        cfg = LoaderConfig(adaptive_prefetch=True, adaptive_prefetch_target_util=0.80)
        assert cfg.adaptive_prefetch is True
        assert cfg.adaptive_prefetch_target_util == 0.80

    def test_target_zero_raises(self):
        with pytest.raises(ValueError, match="adaptive_prefetch_target_util"):
            LoaderConfig(adaptive_prefetch=True, adaptive_prefetch_target_util=0.0)

    def test_target_above_one_raises(self):
        with pytest.raises(ValueError, match="adaptive_prefetch_target_util"):
            LoaderConfig(adaptive_prefetch=True, adaptive_prefetch_target_util=1.1)


class TestPrometheusPort:
    """Prometheus metrics endpoint validation [ARCH3]."""

    def test_invalid_port_zero(self):
        with patch.dict("sys.modules", {"prometheus_client": __import__("unittest.mock").mock.MagicMock()}):
            with pytest.raises(ValueError, match="prometheus_port"):
                LoaderConfig(prometheus_port=0)

    def test_invalid_port_too_large(self):
        with patch.dict("sys.modules", {"prometheus_client": __import__("unittest.mock").mock.MagicMock()}):
            with pytest.raises(ValueError, match="prometheus_port"):
                LoaderConfig(prometheus_port=99999)

    def test_missing_prometheus_client_raises(self):
        with patch.dict("sys.modules", {"prometheus_client": None}):
            with pytest.raises(ValueError, match="prometheus_client"):
                LoaderConfig(prometheus_port=9100)

    def test_valid_port_with_package_installed(self):
        from unittest.mock import MagicMock
        with patch.dict("sys.modules", {"prometheus_client": MagicMock()}):
            cfg = LoaderConfig(prometheus_port=9100)
            assert cfg.prometheus_port == 9100


# ══════════════════════════════════════════════════════════════════════════════
# CheckpointState
# ══════════════════════════════════════════════════════════════════════════════


def _make_state(step: int = 100, epoch: int = 2) -> CheckpointState:
    return CheckpointState(
        step=step,
        epoch=epoch,
        dataset_names=["laion", "imagenet"],
        mixing_weights=[0.7, 0.3],
        global_crop_size=224,
        local_crop_size=96,
    )


class TestCheckpointStateRoundTrip:

    def test_save_and_load_step(self, tmp_path):
        state = _make_state(step=42)
        path = tmp_path / "dl_state.json"
        state.save(path)
        loaded = CheckpointState.load(path)
        assert loaded.step == 42

    def test_save_and_load_epoch(self, tmp_path):
        state = _make_state(epoch=5)
        path = tmp_path / "dl_state.json"
        state.save(path)
        assert CheckpointState.load(path).epoch == 5

    def test_save_and_load_dataset_names(self, tmp_path):
        path = tmp_path / "dl_state.json"
        _make_state().save(path)
        assert CheckpointState.load(path).dataset_names == ["laion", "imagenet"]

    def test_save_and_load_mixing_weights(self, tmp_path):
        path = tmp_path / "dl_state.json"
        _make_state().save(path)
        loaded = CheckpointState.load(path)
        assert loaded.mixing_weights == [0.7, 0.3]

    def test_save_and_load_crop_sizes(self, tmp_path):
        path = tmp_path / "dl_state.json"
        _make_state().save(path)
        loaded = CheckpointState.load(path)
        assert loaded.global_crop_size == 224
        assert loaded.local_crop_size == 96

    def test_saved_file_is_valid_json(self, tmp_path):
        path = tmp_path / "dl_state.json"
        _make_state(step=100).save(path)
        data = json.loads(path.read_text())
        assert "payload" in data or "step" in data  # envelope or legacy

    def test_atomic_write_no_tmp_after_success(self, tmp_path):
        path = tmp_path / "dl_state.json"
        _make_state().save(path)
        assert not (tmp_path / "dl_state.tmp").exists()
        assert path.exists()


class TestCheckpointStateIntegrity:
    """SHA-256 envelope prevents silent corruption [M3]."""

    def test_save_creates_envelope_with_sha256(self, tmp_path):
        path = tmp_path / "state.json"
        _make_state().save(path)
        raw = json.loads(path.read_text())
        assert "payload" in raw
        assert "sha256" in raw
        assert len(raw["sha256"]) == 64

    def test_checksum_is_correct(self, tmp_path):
        path = tmp_path / "state.json"
        _make_state().save(path)
        raw = json.loads(path.read_text())
        computed = hashlib.sha256(json.dumps(raw["payload"], indent=2).encode()).hexdigest()
        assert raw["sha256"] == computed

    def test_tampered_payload_raises_on_load(self, tmp_path):
        path = tmp_path / "state.json"
        _make_state().save(path)
        raw = json.loads(path.read_text())
        raw["payload"]["step"] = 9999
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="integrity check"):
            CheckpointState.load(path)


class TestCheckpointStateBackwardCompat:

    def test_legacy_flat_format_loads(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({
            "step": 10, "epoch": 0,
            "dataset_names": ["laion"],
            "mixing_weights": [1.0],
        }))
        state = CheckpointState.load(path)
        assert state.step == 10

    def test_missing_crop_sizes_default_to_224_96(self, tmp_path):
        path = tmp_path / "old_state.json"
        path.write_text(json.dumps({
            "step": 50,
            "epoch": 1,
            "dataset_names": ["laion"],
            "mixing_weights": [1.0],
            # global_crop_size / local_crop_size intentionally absent
        }))
        loaded = CheckpointState.load(path)
        assert loaded.global_crop_size == 224
        assert loaded.local_crop_size == 96
