"""
dino_loader.backends.dali_backend
==================================
Concrete backend: NVIDIA DALI + CUDA production path.

[DALI-AUG-1] build_pipeline dispatches on AugmentationSpec subtype via
             isinstance chains (replacing structural pattern matching).
[DALI-AUG-2] UserAugSpec path: _UserAugIterator calls aug_fn on decoded GPU
             tensors after each DALI batch.
[DALI-AUG-3] EvalAugSpec and LeJEPAAugSpec are fully native DALI pipelines.
"""

import logging
from typing import Any

log = logging.getLogger(__name__)

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False


class _UserAugIterator:
    """Wraps a DALI decode-only iterator and applies UserAugSpec.aug_fn.

    DALI provides decoded, normalised Tensor[B, C, H, W] under the key
    "decoded". aug_fn transforms this into a dict mapping view names to
    tensors, returned in the same list-of-dict format as DALIGenericIterator.
    """

    def __init__(self, dali_iter: Any, aug_spec: Any) -> None:
        self._iter   = dali_iter
        self._aug_fn = aug_spec.aug_fn
        self._map    = aug_spec.output_map
        self._closed = False

    def __iter__(self) -> "_UserAugIterator":
        return self

    def __next__(self) -> list[dict[str, Any]]:
        raw             = next(self._iter)  # list[{"decoded": Tensor[B,C,H,W]}]
        decoded_batch   = raw[0]["decoded"]
        augmented       = self._aug_fn(decoded_batch)

        missing = [k for k in self._map if k not in augmented]
        if missing:
            msg = (
                f"UserAugSpec.aug_fn did not return expected view(s): {missing}. "
                f"Got keys: {list(augmented.keys())}"
            )
            raise ValueError(msg)

        return [augmented]

    def reset(self) -> None:
        """Delegate to the underlying DALI iterator."""
        self._iter.reset()


class DALIBackend:
    """Concrete backend: NVIDIA DALI + CUDA production path."""

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

    def build_shard_cache(
        self,
        job_id:          str,
        node_master:     bool,
        max_gb:          float,
        prefetch_window: int,
        timeout_s:       float,
        warn_threshold:  float,
        **kwargs: Any,
    ) -> Any:
        from dino_loader.shard_cache import NodeSharedShardCache
        return NodeSharedShardCache(
            node_master       = node_master,
            job_id            = job_id,
            max_shm_gb        = max_gb,
            prefetch_window   = prefetch_window,
            shard_timeout_s   = timeout_s,
            shm_warn_threshold = warn_threshold,
            heartbeat_stale_s = kwargs.get("heartbeat_stale_s", 300.0),
        )

    def build_pipeline(
        self,
        source:             Any,
        aug_spec:           Any,
        aug_cfg:            Any         = None,
        batch_size:         int         = 1,
        num_threads:        int         = 8,
        device_id:          int         = 0,
        resolution_src:     Any         = None,
        hw_decoder_load:    float       = 0.90,
        cpu_queue:          int         = 8,
        gpu_queue:          int         = 6,
        seed:               int         = 42,
        specs:              list | None = None,
        fuse_normalization: bool        = False,
        dali_fp8_output:    bool        = False,
    ) -> Any:
        """Build and return a compiled DALI pipeline, dispatching on aug_spec type."""
        try:
            import nvidia.dali  # noqa: F401
        except ImportError:
            msg = (
                "nvidia-dali is required for the DALI backend but is not installed.\n"
                "Install it with: pip install nvidia-dali-cuda120\n"
                "Or use the CPU backend: get_backend('cpu')"
            )
            raise RuntimeError(msg) from None

        from dino_loader.pipeline import NormSource, build_pipeline
        from dino_loader.augmentation import DinoV2AugSpec

        norm_source = None
        if (
            isinstance(aug_spec, DinoV2AugSpec)
            and fuse_normalization
            and specs is not None
        ):
            norm_source = NormSource(aug_cfg=aug_spec.aug_cfg, specs=specs)
            source.register_dataset_index_callback(norm_source.set_dataset_indices)
            log.debug(
                "DALIBackend: NormSource built for %d dataset(s), fused into DALI graph.",
                len(specs),
            )

        return build_pipeline(
            source             = source,
            aug_spec           = aug_spec,
            batch_size         = batch_size,
            num_threads        = num_threads,
            device_id          = device_id,
            resolution_src     = resolution_src,
            hw_decoder_load    = hw_decoder_load,
            cpu_queue          = cpu_queue,
            gpu_queue          = gpu_queue,
            seed               = seed,
            norm_source        = norm_source,
            fuse_normalization = fuse_normalization,
            dali_fp8_output    = dali_fp8_output,
        )

    def build_pipeline_iterator(
        self,
        pipeline:   Any,
        aug_spec:   Any,
        output_map: list[str],
        batch_size: int,
    ) -> Any:
        """Wrap the pipeline in the appropriate iterator."""
        if not HAS_DALI:
            raise RuntimeError("nvidia-dali required for DALIGenericIterator.")

        from dino_loader.augmentation import UserAugSpec

        dali_iter = DALIGenericIterator(
            pipelines         = [pipeline],
            output_map        = ["decoded"] if isinstance(aug_spec, UserAugSpec) else output_map,
            last_batch_policy = LastBatchPolicy.DROP,
            auto_reset        = False,
        )

        if isinstance(aug_spec, UserAugSpec):
            return _UserAugIterator(dali_iter, aug_spec)

        return dali_iter

    def build_h2d_stream(self, device: Any, topo: Any) -> Any:
        from dino_loader.memory import H2DStream
        return H2DStream(device=device, topo=topo)

    def build_fp8_formatter(self) -> Any:
        from dino_loader.memory import FP8Formatter
        return FP8Formatter()

    def init_distributed(
        self,
        rank:             int = 0,
        world_size:       int = 1,
        local_rank:       int = 0,
        local_world_size: int = 1,
        force_topology:   str | None = None,
    ) -> Any:
        from dino_env import detect_topology, ClusterTopology
        from dino_loader.distributed import DistribEnv
        topo = detect_topology(force=force_topology, gpu_index=local_rank)
        return DistribEnv(
            rank             = rank,
            world_size       = world_size,
            local_rank       = local_rank,
            local_world_size = local_world_size,
            topology         = topo,
        )
