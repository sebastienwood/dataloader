"""tests/test_cpu_backend.py
=========================
Unit tests for CPU backend components.

Tests here are deliberately fine-grained and fast — they target individual
classes rather than the full loader stack.  Integration tests live in
test_loader_cpu.py.

Coverage
--------
- InProcessShardCache  : get, get_view, LRU eviction, utilisation
- CPUAugPipeline       : output shapes, dtype, value range, close()
- CPUPipelineIterator  : DALI-compatible iteration protocol
- NullH2DStream        : passthrough context manager
- NullFP8Formatter     : identity quantise
- StubDistribEnv       : attribute access
- CPUBackend           : factory methods, BackendProtocol compliance
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.backends.cpu import (
    CPUAugPipeline,
    CPUPipelineIterator,
    InProcessShardCache,
    NullFP8Formatter,
    NullH2DStream,
    StubClusterTopology,
    StubDistribEnv,
    _ViewAugParams,
    _augment_one,
)
from dino_loader.config import DINOAugConfig, PipelineConfig
from dino_loader.sources.resolution import ResolutionSource
from tests.fixtures import write_shard

# ══════════════════════════════════════════════════════════════════════════════
# InProcessShardCache
# ══════════════════════════════════════════════════════════════════════════════


class TestInProcessShardCache:

    def test_get_reads_file(self, tmp_path):
        tar_path, _ = write_shard(tmp_path, n_samples=4)
        cache = InProcessShardCache(max_gb=1.0)
        data = cache.get(tar_path)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_get_twice_hits_cache(self, tmp_path):
        tar_path, _ = write_shard(tmp_path, n_samples=4)
        cache = InProcessShardCache(max_gb=1.0)
        data1 = cache.get(tar_path)
        data2 = cache.get(tar_path)
        assert data1 == data2

    def test_get_view_yields_memoryview(self, tmp_path):
        tar_path, _ = write_shard(tmp_path, n_samples=4)
        cache = InProcessShardCache(max_gb=1.0)
        with cache.get_view(tar_path) as mv:
            assert isinstance(mv, memoryview)
            assert len(mv) > 0

    def test_utilisation_zero_initially(self):
        cache = InProcessShardCache(max_gb=1.0)
        assert cache.utilisation == 0.0

    def test_utilisation_nonzero_after_get(self, tmp_path):
        tar_path, _ = write_shard(tmp_path, n_samples=8)
        cache = InProcessShardCache(max_gb=1.0)
        cache.get(tar_path)
        assert 0.0 < cache.utilisation < 1.0

    def test_lru_eviction_respects_budget(self, tmp_path):
        tar_path1, _ = write_shard(tmp_path, shard_idx=0, n_samples=8)
        tar_path2, _ = write_shard(tmp_path, shard_idx=1, n_samples=8)

        cache = InProcessShardCache(max_gb=1e-4)  # ~100 KB
        cache.get(tar_path1)
        assert len(cache._lru) == 1

        cache.get(tar_path2)
        assert tar_path1 not in cache._lru or cache._total <= cache._max_bytes

    def test_prefetch_is_noop(self, tmp_path):
        tar_path, _ = write_shard(tmp_path, n_samples=4)
        cache = InProcessShardCache(max_gb=1.0)
        cache.prefetch(tar_path)   # must not raise

    def test_max_bytes_zero_utilisation(self):
        cache = InProcessShardCache(max_gb=0.0)
        assert cache.utilisation == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# _augment_one — utilise _ViewAugParams
# ══════════════════════════════════════════════════════════════════════════════


class TestAugmentOne:

    @pytest.fixture(autouse=True)
    def _aug_cfg(self, small_aug_cfg):
        self.cfg = small_aug_cfg

    def _sample_jpeg(self) -> bytes:
        from tests.fixtures import make_jpeg_bytes
        return make_jpeg_bytes(64, 64)

    def _global_params(self, crop_size: int = 32, sol_prob: float = 0.0) -> _ViewAugParams:
        return _ViewAugParams(
            crop_size = crop_size,
            scale     = (0.5, 1.0),
            blur_prob = 0.0,
            sol_prob  = sol_prob,
        )

    def _local_params(self, crop_size: int = 16) -> _ViewAugParams:
        return _ViewAugParams(
            crop_size = crop_size,
            scale     = (0.05, 0.32),
            blur_prob = 0.5,
            sol_prob  = 0.0,
        )

    def test_output_shape(self):
        t = _augment_one(self._sample_jpeg(), self.cfg, self._global_params(32))
        assert t.shape == (3, 32, 32), f"Unexpected shape: {t.shape}"

    def test_output_dtype_float(self):
        t = _augment_one(self._sample_jpeg(), self.cfg, self._global_params(32))
        assert t.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_corrupt_jpeg_returns_zeros(self):
        params = self._global_params(32)
        t = _augment_one(b"not a valid jpeg", self.cfg, params)
        assert t.shape == (3, 32, 32)
        assert torch.all(t == 0)

    def test_solarize_enabled(self):
        params = _ViewAugParams(crop_size=32, scale=(0.5, 1.0), blur_prob=0.0, sol_prob=1.0)
        t = _augment_one(self._sample_jpeg(), self.cfg, params)
        assert t.shape == (3, 32, 32)

    def test_local_crop_size(self):
        t = _augment_one(self._sample_jpeg(), self.cfg, self._local_params(16))
        assert t.shape == (3, 16, 16)


# ══════════════════════════════════════════════════════════════════════════════
# CPUAugPipeline
# ══════════════════════════════════════════════════════════════════════════════


class TestCPUAugPipeline:

    def _make_source(self, batch_size: int):
        from tests.fixtures import make_jpeg_bytes

        def _source(info=None):
            return [
                np.frombuffer(make_jpeg_bytes(64, 64), dtype=np.uint8)
                for _ in range(batch_size)
            ]
        return _source

    def test_run_one_batch_shapes(self, small_aug_cfg):
        batch_size = 4
        res_src    = ResolutionSource(32, 16)
        source     = self._make_source(batch_size)
        pipe       = CPUAugPipeline(source, small_aug_cfg, batch_size, res_src, seed=0)

        result = pipe.run_one_batch()

        n_views = small_aug_cfg.n_views
        assert len(result) == n_views

        for i in range(small_aug_cfg.n_global_crops):
            t = result[f"view_{i}"]
            assert t.shape == (batch_size, 3, 32, 32), f"view_{i}: {t.shape}"

        for i in range(small_aug_cfg.n_global_crops,
                       small_aug_cfg.n_global_crops + small_aug_cfg.n_local_crops):
            t = result[f"view_{i}"]
            assert t.shape == (batch_size, 3, 16, 16), f"view_{i}: {t.shape}"

        pipe.close()

    def test_dynamic_resolution(self, small_aug_cfg):
        batch_size = 2
        res_src    = ResolutionSource(32, 16)
        source     = self._make_source(batch_size)
        pipe       = CPUAugPipeline(source, small_aug_cfg, batch_size, res_src, seed=0)

        res_src.set(64, 32)
        result = pipe.run_one_batch()

        t = result["view_0"]
        assert t.shape == (batch_size, 3, 64, 64), f"Expected 64x64, got {t.shape}"
        pipe.close()

    def test_values_are_finite(self, small_aug_cfg):
        batch_size = 2
        res_src    = ResolutionSource(32, 16)
        source     = self._make_source(batch_size)
        pipe       = CPUAugPipeline(source, small_aug_cfg, batch_size, res_src, seed=42)

        result = pipe.run_one_batch()
        for k, v in result.items():
            assert torch.isfinite(v).all(), f"Non-finite values in {k}"
        pipe.close()

    def test_all_views_present(self, small_aug_cfg):
        batch_size = 2
        res_src    = ResolutionSource(32, 16)
        source     = self._make_source(batch_size)
        pipe       = CPUAugPipeline(source, small_aug_cfg, batch_size, res_src, seed=0)
        result     = pipe.run_one_batch()

        expected_keys = {f"view_{i}" for i in range(small_aug_cfg.n_views)}
        assert set(result.keys()) == expected_keys
        pipe.close()

    def test_close_is_idempotent(self, small_aug_cfg):
        """Calling close() multiple times must not raise."""
        res_src = ResolutionSource(32, 16)
        pipe    = CPUAugPipeline(self._make_source(2), small_aug_cfg, 2, res_src, seed=0)
        pipe.close()
        pipe.close()  # second call must be safe

    def test_run_after_close_raises(self, small_aug_cfg):
        res_src = ResolutionSource(32, 16)
        pipe    = CPUAugPipeline(self._make_source(2), small_aug_cfg, 2, res_src, seed=0)
        pipe.close()
        with pytest.raises(RuntimeError, match="close"):
            pipe.run_one_batch()


# ══════════════════════════════════════════════════════════════════════════════
# CPUPipelineIterator
# ══════════════════════════════════════════════════════════════════════════════


class TestCPUPipelineIterator:

    def _make_iterator(self, small_aug_cfg, batch_size=2, n_batches=3):
        from tests.fixtures import make_jpeg_bytes
        res_src = ResolutionSource(32, 16)

        call_count = [0]

        def _source(info=None):
            if call_count[0] >= n_batches:
                raise StopIteration
            call_count[0] += 1
            return [
                np.frombuffer(make_jpeg_bytes(64, 64), dtype=np.uint8)
                for _ in range(batch_size)
            ]

        pipe = CPUAugPipeline(_source, small_aug_cfg, batch_size, res_src, seed=0)
        output_map = [f"view_{i}" for i in range(small_aug_cfg.n_views)]
        return CPUPipelineIterator(pipe, output_map, batch_size)

    def test_yields_list_of_dict(self, small_aug_cfg):
        it = self._make_iterator(small_aug_cfg)
        out = next(it)
        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], dict)

    def test_dict_has_view_keys(self, small_aug_cfg):
        it  = self._make_iterator(small_aug_cfg)
        out = next(it)[0]
        for i in range(small_aug_cfg.n_views):
            assert f"view_{i}" in out

    def test_reset_allows_reiteration(self, small_aug_cfg):
        it  = self._make_iterator(small_aug_cfg, n_batches=1)
        next(it)
        it.reset()
        assert not it._exhausted


# ══════════════════════════════════════════════════════════════════════════════
# NullH2DStream
# ══════════════════════════════════════════════════════════════════════════════


class TestNullH2DStream:

    def test_transfer_returns_same_tensors(self):
        device = torch.device("cpu")
        topo   = StubClusterTopology()
        h2d    = NullH2DStream(device=device, topo=topo)

        t1 = torch.randn(2, 3, 32, 32)
        t2 = torch.randn(2, 3, 16, 16)
        batch = {"global": [t1], "local": [t2]}

        with h2d.transfer(batch) as out:
            assert out["global"][0] is t1
            assert out["local"][0]  is t2

    def test_send_returns_same_batch(self):
        device = torch.device("cpu")
        topo   = StubClusterTopology()
        h2d    = NullH2DStream(device=device, topo=topo)

        batch = {"global": [torch.zeros(2, 3, 32, 32)]}
        out   = h2d.send(batch)
        assert out is batch

    def test_wait_is_noop(self):
        h2d = NullH2DStream(device=torch.device("cpu"), topo=StubClusterTopology())
        h2d.wait()


# ══════════════════════════════════════════════════════════════════════════════
# NullFP8Formatter
# ══════════════════════════════════════════════════════════════════════════════


class TestNullFP8Formatter:

    def test_quantise_returns_same_tensor(self):
        fmt = NullFP8Formatter()
        t   = torch.randn(4, 3, 32, 32)
        out = fmt.quantise(t)
        assert out is t

    def test_quantise_does_not_modify_values(self):
        fmt = NullFP8Formatter()
        t   = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        out = fmt.quantise(t)
        assert torch.equal(out, t)


# ══════════════════════════════════════════════════════════════════════════════
# StubDistribEnv / StubClusterTopology
# ══════════════════════════════════════════════════════════════════════════════


class TestStubDistribEnv:

    def test_default_values(self):
        env = StubDistribEnv()
        assert env.rank             == 0
        assert env.world_size       == 1
        assert env.local_rank       == 0
        assert env.local_world_size == 1
        assert env.topology is not None

    def test_topology_label(self):
        env = StubDistribEnv()
        assert "CPU" in env.topology.label or "stub" in env.topology.label.lower()

    def test_topology_no_nvl72(self):
        topo = StubClusterTopology()
        assert not topo.is_nvl72
        assert not topo.is_grace_blackwell
        assert not topo.has_infiniband

    def test_custom_values(self):
        env = StubDistribEnv(rank=2, world_size=8, local_rank=2, local_world_size=4)
        assert env.rank        == 2
        assert env.world_size  == 8
        assert env.local_rank  == 2


# ══════════════════════════════════════════════════════════════════════════════
# CPUBackend factory — utilise PipelineConfig [CFG-PIPE]
# ══════════════════════════════════════════════════════════════════════════════


class TestCPUBackend:

    def test_name(self, cpu_backend):
        assert cpu_backend.name == "cpu"

    def test_supports_fp8_false(self, cpu_backend):
        assert cpu_backend.supports_fp8 is False

    def test_supports_gpu_false(self, cpu_backend):
        assert cpu_backend.supports_gpu is False

    def test_build_shard_cache(self, cpu_backend):
        cache = cpu_backend.build_shard_cache(
            job_id="test", node_master=True,
            max_gb=0.1, prefetch_window=2,
            timeout_s=10.0, warn_threshold=0.85,
        )
        assert isinstance(cache, InProcessShardCache)

    def test_build_pipeline_with_pipeline_cfg(self, cpu_backend, small_aug_cfg, tmp_path):
        from tests.fixtures import make_jpeg_bytes
        from dino_loader.augmentation import DinoV2AugSpec

        res_src    = ResolutionSource(32, 16)
        batch_size = 2

        def _source(info=None):
            return [
                np.frombuffer(make_jpeg_bytes(64, 64), dtype=np.uint8)
                for _ in range(batch_size)
            ]
        _source._resolution_src = res_src
        _source._batch_size     = batch_size

        aug_spec     = DinoV2AugSpec(aug_cfg=small_aug_cfg)
        pipeline_cfg = PipelineConfig(device_id=0, seed=0)

        pipe = cpu_backend.build_pipeline(
            source       = _source,
            aug_spec     = aug_spec,
            pipeline_cfg = pipeline_cfg,
        )
        assert isinstance(pipe, CPUAugPipeline)
        pipe.close()

    def test_build_pipeline_iterator(self, cpu_backend, small_aug_cfg):
        from tests.fixtures import make_jpeg_bytes
        from dino_loader.augmentation import DinoV2AugSpec

        res_src    = ResolutionSource(32, 16)
        batch_size = 2

        def _source(info=None):
            return [
                np.frombuffer(make_jpeg_bytes(64, 64), dtype=np.uint8)
                for _ in range(batch_size)
            ]
        _source._resolution_src = res_src
        _source._batch_size     = batch_size

        aug_spec     = DinoV2AugSpec(aug_cfg=small_aug_cfg)
        pipeline_cfg = PipelineConfig(device_id=0, seed=0)

        pipe = cpu_backend.build_pipeline(
            source=_source, aug_spec=aug_spec, pipeline_cfg=pipeline_cfg,
        )
        it = cpu_backend.build_pipeline_iterator(
            pipeline   = pipe,
            aug_spec   = aug_spec,
            output_map = [f"view_{i}" for i in range(small_aug_cfg.n_views)],
            batch_size = batch_size,
        )
        assert isinstance(it, CPUPipelineIterator)
        pipe.close()

    def test_build_h2d_stream(self, cpu_backend):
        h2d = cpu_backend.build_h2d_stream(
            device = torch.device("cpu"),
            topo   = StubClusterTopology(),
        )
        assert isinstance(h2d, NullH2DStream)

    def test_build_fp8_formatter(self, cpu_backend):
        fmt = cpu_backend.build_fp8_formatter()
        assert isinstance(fmt, NullFP8Formatter)

    def test_init_distributed(self, cpu_backend):
        env = cpu_backend.init_distributed(rank=0, world_size=1)
        assert isinstance(env, StubDistribEnv)
        assert env.rank == 0

    def test_protocol_compliance(self, cpu_backend):
        from dino_loader.backends.protocol import BackendProtocol
        assert isinstance(cpu_backend, BackendProtocol)