"""
dino_loader.backends.dali_backend
==================================
Production DALI backend — thin delegation layer over the existing
``pipeline.py``, ``memory.py``, ``shard_cache.py``, and ``distributed.py``
modules.

This class was always implicit in the original code.  Making it explicit as a
``BackendProtocol`` implementor lets ``DINODataLoader`` select the backend at
construction time, which in turn allows the CPU backend to be injected for
testing without any monkey-patching.

No production behaviour is changed.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

log = logging.getLogger(__name__)

# All production imports are deferred so that importing this module does not
# fail on machines without torch / DALI.  The actual import errors surface
# at method-call time with a clear message.

log = logging.getLogger(__name__)

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False


class DALIBackend:
    """
    Concrete backend: NVIDIA DALI + CUDA + SLURM production path.

    Delegates to the original module-level functions so that all the
    carefully tuned production code paths (async Lustre I/O, inotify,
    FP8 quantisation, nvjpeg, etc.) remain exactly as authored.
    """

    # ── BackendProtocol identity ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "dali"

    @property
    def supports_fp8(self) -> bool:
        try:
            import transformer_engine.pytorch  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def supports_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def build_shard_cache(
        self,
        job_id:          str,
        node_master:     bool,
        max_gb:          float,
        prefetch_window: int,
        timeout_s:       float,
        warn_threshold:  float,
    ):
        from dino_loader.shard_cache import NodeSharedShardCache
        return NodeSharedShardCache(
            job_id             = job_id,
            node_master        = node_master,
            max_shm_gb         = max_gb,
            prefetch_window    = prefetch_window,
            shard_timeout_s    = timeout_s,
            shm_warn_threshold = warn_threshold,
        )

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def build_pipeline(
        self,
        source,
        aug_cfg,
        batch_size:      int,
        num_threads:     int,
        device_id:       int,
        resolution_src,
        hw_decoder_load: float = 0.90,
        cpu_queue:       int   = 8,
        gpu_queue:       int   = 6,
        seed:            int   = 42,
    ):
        try:
            import nvidia.dali  # noqa: F401
            has_dali = True
        except ImportError:
            has_dali = False

        if not has_dali:
            raise RuntimeError(
                "nvidia-dali is required for the DALI backend but is not installed. "
                "Install it with: pip install nvidia-dali-cuda120\n"
                "Or use the CPU backend: get_backend('cpu')"
            )
        from dino_loader.pipeline import build_pipeline
        return build_pipeline(
            source          = source,
            aug_cfg         = aug_cfg,
            batch_size      = batch_size,
            num_threads     = num_threads,
            device_id       = device_id,
            resolution_src  = resolution_src,
            hw_decoder_load = hw_decoder_load,
            cpu_queue       = cpu_queue,
            gpu_queue       = gpu_queue,
            seed            = seed,
        )

    def build_pipeline_iterator(
        self,
        pipeline,
        output_map: List[str],
        batch_size: int,
    ):
        try:
            from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
        except ImportError:
            raise RuntimeError("nvidia-dali required for DALIGenericIterator.")
        return DALIGenericIterator(
            pipelines         = [pipeline],
            output_map        = output_map,
            last_batch_policy = LastBatchPolicy.DROP,
            auto_reset        = False,
        )

    # ── Stage 4 ───────────────────────────────────────────────────────────────

    def build_h2d_stream(self, device, topo):
        import torch
        from dino_loader.memory import H2DStream
        return H2DStream(device=device, topo=topo)

    # ── Stage 5 ───────────────────────────────────────────────────────────────

    def build_fp8_formatter(self):
        from dino_loader.memory import FP8Formatter
        return FP8Formatter()

    # ── Distributed ──────────────────────────────────────────────────────────

    def init_distributed(
        self,
        rank:             int = 0,
        world_size:       int = 1,
        local_rank:       int = 0,
        local_world_size: int = 1,
        force_topology:   Optional[str] = None,
    ):
        """
        For the DALI backend, distributed init is expected to have been
        performed already via ``slurm_init()``.  This method builds the
        DistribEnv from already-initialised state.
        """
        from dino_loader.distributed import detect_topology, DistribEnv
        topo = detect_topology(force=force_topology)
        return DistribEnv(
            rank             = rank,
            world_size       = world_size,
            local_rank       = local_rank,
            local_world_size = local_world_size,
            topology         = topo,
        )
