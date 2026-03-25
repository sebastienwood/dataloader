"""dino_loader.loader
==================
DINODataLoader: the single public entry point for training code.

[LD-AUG-1] aug_spec parameter accepts any AugmentationSpec subclass.
           The legacy aug_cfg kwarg is accepted and silently wrapped in
           DinoV2AugSpec(aug_cfg=aug_cfg) for backward compatibility.
[LD-AUG-2] sample_predicate: early filtering before DALI decode.
[B3-FIX]   set_epoch() is protected by threading.Lock.
[M6-FIX]   PostProcessPipeline.select() increments batches_filtered metric.
[LD-13]    current_resolution public property.
[FIX-ENV]  DINODataLoader now asserts dino_env.init() has been called before
           construction when running in DALI mode, preventing silent NCCL
           misconfiguration.
[FIX-ITER] _active_iter is protected by a threading.Lock to prevent a TOCTOU
           race when two threads call __iter__() simultaneously.

AsyncPrefetchIterator removal
------------------------------
_raw_iter now iterates directly over self._dali_iter.  DALI's own prefetch
queues (controlled by LoaderConfig.dali_cpu_queue / dali_gpu_queue, defaulting
to 16 / 6) provide equivalent or better double-buffering natively — inside the
C++ runtime, without the GIL overhead that our Python-level Future threading
introduced.  The previous AsyncPrefetchIterator was also the source of the B1
race condition that required a dedicated fix; removing it eliminates the class
of bugs entirely.
"""

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
import torch.distributed as dist
from dino_datasets import DatasetSpec

from dino_loader.augmentation import (
    AugmentationSpec,
    DinoV2AugSpec,
    EvalAugSpec,
    LeJEPAAugSpec,
    SamplePredicate,
    UserAugSpec,
)
from dino_loader.backends import get_backend
from dino_loader.backends.protocol import BackendProtocol
from dino_loader.checkpoint import DataLoaderCheckpointer
from dino_loader.config import CheckpointState, DINOAugConfig, LoaderConfig
from dino_loader.memory import Batch
from dino_loader.mixing_source import MixingSource, ResolutionSource
from dino_loader.monitor.metrics import get_registry, init_registry

log = logging.getLogger(__name__)


class PostProcessPipeline:
    """Lazy, composable wrapper over a Batch iterator.

    Each method returns a new PostProcessPipeline; the original is not mutated.
    Transforms execute only as batches flow through — no buffering.

    Filtering guidance
    ------------------
    select() drops batches after DALI decode. For metadata-based filtering
    (quality score, class label, …), prefer DINODataLoader's sample_predicate
    parameter which rejects samples before DALI decode at near-zero cost.
    """

    def __init__(
        self,
        source:     Iterator[Batch],
        transforms: list[Callable],
        loader:     "DINODataLoader",
        max_steps:  int | None = None,
    ) -> None:
        self._source     = source
        self._transforms = transforms
        self._loader     = loader
        self._max_steps  = max_steps

    def map(self, fn: Callable[[Batch], Batch]) -> "PostProcessPipeline":
        """Apply fn to every batch."""
        return PostProcessPipeline(
            source     = self._source,
            transforms = self._transforms + [fn],
            loader     = self._loader,
            max_steps  = self._max_steps,
        )

    def select(self, predicate: Callable[[Batch], bool]) -> "PostProcessPipeline":
        """Drop batches for which predicate returns False.

        For metadata-based filtering, use DINODataLoader(sample_predicate=...)
        instead to avoid the decode cost.
        """
        metrics = get_registry()

        def _filter(b: Batch) -> Batch | None:
            if predicate(b):
                return b
            if metrics is not None:
                metrics.inc("batches_filtered", 1)
            return None

        return PostProcessPipeline(
            source     = self._source,
            transforms = self._transforms + [_filter],
            loader     = self._loader,
            max_steps  = self._max_steps,
        )

    def with_epoch(self, n_steps: int) -> "PostProcessPipeline":
        """Limit iteration to n_steps batches per epoch."""
        return PostProcessPipeline(
            source     = self._source,
            transforms = self._transforms,
            loader     = self._loader,
            max_steps  = n_steps,
        )

    def set_epoch(self, epoch: int) -> None:
        """Delegate to the underlying loader."""
        self._loader.set_epoch(epoch)

    def checkpoint(self, step: int) -> None:
        """Delegate to the underlying loader."""
        self._loader.checkpoint(step)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Delegate to the underlying loader."""
        self._loader.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Delegate to the underlying loader."""
        self._loader.set_weight_by_name(name, weight)

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Delegate to the underlying loader."""
        self._loader.set_resolution(global_size, local_size)

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Return the current (global_size, local_size)."""
        return self._loader.current_resolution

    def state_dict(self) -> dict:
        """Delegate to the underlying loader."""
        return self._loader.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        """Delegate to the underlying loader."""
        self._loader.load_state_dict(sd)

    def __iter__(self) -> Iterator[Batch]:
        step = 0
        for batch in self._source:
            if self._max_steps is not None and step >= self._max_steps:
                break
            result: Batch | None = batch
            for fn in self._transforms:
                if result is None:
                    break
                result = fn(result)
            if result is not None:
                yield result
                step += 1

    def __len__(self) -> int:
        """Return max_steps if set, else delegate to the underlying loader."""
        if self._max_steps is not None:
            return self._max_steps
        return len(self._loader)


class DINODataLoader:
    """HPC-grade data loader for DINO-style self-supervised training.

    Prerequisites
    -------------
    For the DALI backend (production), dino_env.init() must be called before
    constructing this loader. The loader asserts this to prevent silent NCCL
    misconfiguration.

    Pipeline stages
    ---------------
    Stage 1 — NodeSharedShardCache: one rank reads Lustre, others read /dev/shm.
    Stage 2 — MixingSource: weighted multi-dataset ExternalSource for DALI.
    Stage 3 — DALI pipeline: HW JPEG decode + augmentation on GPU.
              DALI's prefetch queues (dali_cpu_queue, dali_gpu_queue) overlap
              I/O and compute natively — no application-level threading needed.
    Stage 4 — H2DStream: dedicated CUDA stream for pinned→GPU transfer.
    Stage 5 — FP8Formatter: optional BF16→FP8 E4M3 quantisation.

    Augmentation strategies
    -----------------------
    Pass an AugmentationSpec to aug_spec:
    - DinoV2AugSpec(aug_cfg)   — DINOv2 multi-crop (default).
    - EvalAugSpec(crop_size)   — deterministic resize + centre-crop.
    - LeJEPAAugSpec(...)       — one context + N target crops.
    - UserAugSpec(aug_fn, ...) — custom function on GPU tensors.

    Args:
        specs: List of DatasetSpec objects.
        batch_size: Per-GPU batch size.
        aug_spec: Augmentation strategy. Defaults to DinoV2AugSpec(DINOAugConfig()).
        aug_cfg: Legacy parameter. Wrapped in DinoV2AugSpec if aug_spec is None.
        config: Loader runtime configuration.
        device_id: Local GPU index.
        rank: Global rank (inferred from environment if None).
        world_size: Total number of ranks (inferred if None).
        local_rank: Local rank (defaults to device_id).
        local_world_size: Ranks per node (inferred if None).
        resume: Load the latest checkpoint on construction.
        steps_per_epoch: Enables len(loader).
        mask_generator: Optional iBOT MaskingGenerator (CPU, post-DALI).
        sample_predicate: Optional early filter before DALI decode.
        backend: "auto" | "dali" | "cpu" | BackendProtocol instance.

    """

    def __init__(
        self,
        specs:            list[DatasetSpec],
        batch_size:       int,
        aug_spec:         AugmentationSpec | None  = None,
        aug_cfg:          DINOAugConfig | None     = None,
        config:           LoaderConfig | None      = None,
        device_id:        int                      = 0,
        rank:             int | None               = None,
        world_size:       int | None               = None,
        local_rank:       int | None               = None,
        local_world_size: int | None               = None,
        resume:           bool                     = False,
        steps_per_epoch:  int | None               = None,
        mask_generator:   Any                      = None,
        sample_predicate: SamplePredicate | None   = None,
        backend:          Any                      = "auto",
    ) -> None:
        # Backward-compat: wrap legacy aug_cfg in DinoV2AugSpec.
        if aug_spec is None:
            effective_aug_cfg: DINOAugConfig     = aug_cfg if aug_cfg is not None else DINOAugConfig()
            self._aug_spec: AugmentationSpec     = DinoV2AugSpec(aug_cfg=effective_aug_cfg)
        else:
            if aug_cfg is not None:
                log.warning(
                    "DINODataLoader: both aug_spec and aug_cfg provided; "
                    "aug_cfg is ignored (aug_spec takes precedence).",
                )
            self._aug_spec = aug_spec

        # Keep a reference to DINOAugConfig for resolution scheduling.
        self._aug_cfg: DINOAugConfig = (
            self._aug_spec.aug_cfg  # type: ignore[attr-defined]
            if isinstance(self._aug_spec, DinoV2AugSpec)
            else DINOAugConfig()
        )

        self._cfg              = config or LoaderConfig()
        self._mask_generator   = mask_generator
        self._sample_predicate = sample_predicate
        self._steps_per_epoch  = steps_per_epoch

        # [FIX-ITER] Lock prevents TOCTOU race when __iter__ called concurrently.
        self._active_iter = False
        self._iter_lock   = threading.Lock()
        self._epoch_lock  = threading.Lock()  # [B3-FIX]

        # Backend
        if isinstance(backend, str):
            self._backend: BackendProtocol = get_backend(backend)
        else:
            self._backend = backend

        # [FIX-ENV] For the DALI backend, require dino_env.init() before construction.
        if self._backend.name == "dali":
            try:
                import dino_env  # noqa: PLC0415
                dino_env.get_env()
            except RuntimeError:
                msg = (
                    "DINODataLoader with the DALI backend requires dino_env.init() "
                    "to be called before constructing the loader."
                )
                raise RuntimeError(msg) from None

        # Distributed
        env = self._backend.init_distributed(
            rank             = rank             if rank             is not None else self._infer_rank(),
            world_size       = world_size       if world_size       is not None else self._infer_world_size(),
            local_rank       = local_rank       if local_rank       is not None else device_id,
            local_world_size = local_world_size if local_world_size is not None else self._infer_local_world_size(),
            force_topology   = self._cfg.force_topology,
        )
        self._rank             = env.rank
        self._world_size       = env.world_size
        self._local_rank       = env.local_rank
        self._local_world_size = env.local_world_size
        self._topo             = env.topology

        # Metrics
        init_registry(rank=self._rank)

        if self._cfg.prometheus_port is not None and self._rank == 0:
            self._start_prometheus(self._cfg.prometheus_port)

        # Resolution tracking
        self._current_global_size = self._initial_global_size()
        self._current_local_size  = self._initial_local_size()
        self._resolution_src      = ResolutionSource(
            self._current_global_size,
            self._current_local_size,
        )

        # Stage 1–2: shard cache + mixing source
        node_master = (self._local_rank == 0)
        job_id      = os.environ.get("SLURM_JOB_ID", "dino_local")

        shard_cache = self._backend.build_shard_cache(
            job_id            = job_id,
            node_master       = node_master,
            max_gb            = self._cfg.node_shm_gb,
            prefetch_window   = self._cfg.shard_prefetch_window,
            timeout_s         = self._cfg.shard_timeout_s,
            warn_threshold    = self._cfg.shm_warn_threshold,
            heartbeat_stale_s = self._cfg.heartbeat_stale_s,
        )

        self._validate_shard_coverage(specs)

        self._source = MixingSource(
            specs               = specs,
            batch_size          = batch_size,
            cache               = shard_cache,
            rank                = self._rank,
            world_size          = self._world_size,
            num_workers         = self._cfg.shard_extraction_workers,
            seed                = self._cfg.seed,
            device_id           = device_id,
            shuffle_buffer_size = self._cfg.shuffle_buffer_size,
            debug_log_keys      = self._cfg.debug_log_keys,
            sample_predicate    = sample_predicate,
        )

        # Stage 3: augmentation pipeline
        pipeline = self._backend.build_pipeline(
            source             = self._source,
            aug_spec           = self._aug_spec,
            aug_cfg            = self._aug_cfg,
            batch_size         = batch_size,
            num_threads        = self._cfg.dali_num_threads,
            device_id          = device_id,
            resolution_src     = self._resolution_src,
            hw_decoder_load    = self._cfg.hw_decoder_load,
            cpu_queue          = self._cfg.dali_cpu_queue,
            gpu_queue          = self._cfg.dali_gpu_queue,
            seed               = self._cfg.seed + self._rank,
            specs              = specs,
            fuse_normalization = self._cfg.fuse_normalization and self._backend.supports_gpu,
            dali_fp8_output    = self._cfg.use_fp8_output and self._cfg.dali_fp8_output,
        )

        self._dali_iter = self._backend.build_pipeline_iterator(
            pipeline   = pipeline,
            aug_spec   = self._aug_spec,
            output_map = self._aug_spec.output_map,
            batch_size = batch_size,
        )

        # Stage 4 & 5: H2D + FP8
        device = (
            torch.device(f"cuda:{device_id}")
            if self._backend.supports_gpu
            else torch.device("cpu")
        )
        dali_fp8     = self._cfg.use_fp8_output and self._cfg.dali_fp8_output
        self._h2d    = self._backend.build_h2d_stream(device=device, topo=self._topo)
        self._fp8    = (
            self._backend.build_fp8_formatter()
            if self._cfg.use_fp8_output and not dali_fp8
            else None
        )

        # Checkpointing
        self._ckpt = DataLoaderCheckpointer(
            ckpt_dir      = self._cfg.checkpoint_dir,
            every_n_steps = self._cfg.checkpoint_every_steps,
            rank          = self._rank,
        )

        if resume:
            self._restore()

        log.info(
            "DINODataLoader ready: backend=%s rank=%d/%d batch=%d "
            "aug=%s resolution=%dx%d dali_queues=cpu%d/gpu%d sample_predicate=%s",
            self._backend.name,
            self._rank, self._world_size, batch_size,
            type(self._aug_spec).__name__,
            self._current_global_size, self._current_local_size,
            self._cfg.dali_cpu_queue, self._cfg.dali_gpu_queue,
            "yes" if sample_predicate is not None else "no",
        )

    # ── Augmentation spec helpers ─────────────────────────────────────────────

    def _initial_global_size(self) -> int:
        if isinstance(self._aug_spec, DinoV2AugSpec):
            return self._aug_spec.aug_cfg.global_crop_size
        if isinstance(self._aug_spec, EvalAugSpec):
            return self._aug_spec.crop_size
        if isinstance(self._aug_spec, LeJEPAAugSpec):
            return self._aug_spec.context_crop_size
        if isinstance(self._aug_spec, UserAugSpec):
            return self._aug_spec.decode_size
        return 224

    def _initial_local_size(self) -> int:
        if isinstance(self._aug_spec, DinoV2AugSpec):
            return self._aug_spec.aug_cfg.local_crop_size
        if isinstance(self._aug_spec, EvalAugSpec):
            return self._aug_spec.crop_size
        if isinstance(self._aug_spec, LeJEPAAugSpec):
            return self._aug_spec.target_crop_size
        if isinstance(self._aug_spec, UserAugSpec):
            return self._aug_spec.decode_size
        return 96

    # ── Fluid API ─────────────────────────────────────────────────────────────

    def map(self, fn: Callable[[Batch], Batch]) -> PostProcessPipeline:
        """Chain a transform on every Batch."""
        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [fn],
            loader     = self,
        )

    def select(self, predicate: Callable[[Batch], bool]) -> PostProcessPipeline:
        """Drop batches post-decode.

        For metadata-based filtering, prefer sample_predicate in the constructor.
        """
        metrics = get_registry()

        def _filter(b: Batch) -> Batch | None:
            if predicate(b):
                return b
            if metrics is not None:
                metrics.inc("batches_filtered", 1)
            return None

        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [_filter],
            loader     = self,
        )

    def with_epoch(self, n_steps: int) -> PostProcessPipeline:
        """Limit to n_steps batches per epoch."""
        return PostProcessPipeline(
            source     = iter(self._raw_iter()),
            transforms = [],
            loader     = self,
            max_steps  = n_steps,
        )

    # ── Epoch / weight / resolution control ───────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Prepare for a new epoch. [B3-FIX] Thread-safe."""
        with self._epoch_lock:
            if isinstance(self._aug_spec, DinoV2AugSpec):
                new_global = self._aug_spec.aug_cfg.crop_size_at_epoch(epoch)
                if new_global != self._current_global_size:
                    self.set_resolution(new_global, self._current_local_size)

            self._source.set_epoch(epoch)
            self._dali_iter.reset()

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Update crop resolution (only meaningful for DinoV2AugSpec)."""
        if isinstance(self._aug_spec, DinoV2AugSpec):
            cfg = self._aug_spec.aug_cfg
            if global_size > cfg.max_global_crop_size:
                msg = (
                    f"set_resolution: global_size={global_size} exceeds "
                    f"max_global_crop_size={cfg.max_global_crop_size}."
                )
                raise ValueError(msg)
            if local_size > cfg.max_local_crop_size:
                msg = (
                    f"set_resolution: local_size={local_size} exceeds "
                    f"max_local_crop_size={cfg.max_local_crop_size}."
                )
                raise ValueError(msg)
        else:
            log.warning(
                "set_resolution called on a %s — resolution changes have no effect "
                "for non-DinoV2 augmentation specs.",
                type(self._aug_spec).__name__,
            )
        self._current_global_size = global_size
        self._current_local_size  = local_size
        self._resolution_src.set(global_size, local_size)
        log.info("Resolution updated: global=%d local=%d", global_size, local_size)

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Current crop resolution as (global_size, local_size)."""
        return (self._current_global_size, self._current_local_size)

    @property
    def aug_spec(self) -> AugmentationSpec:
        """The active augmentation spec."""
        return self._aug_spec

    @property
    def current_weights(self) -> list[float]:
        """Current normalised mixing weights."""
        return self._source.current_weights

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update mixing weights (re-normalised automatically)."""
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Update one dataset's weight by name."""
        self._source.set_weight_by_name(name, weight)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def checkpoint(self, step: int) -> None:
        """Save a checkpoint (rank 0 only, every N steps)."""
        state = CheckpointState(
            step             = step,
            epoch            = getattr(self._source, "_epoch", 0),
            dataset_names    = self._source.dataset_names,
            mixing_weights   = self._source.current_weights,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
        )
        self._ckpt.save(state)

    def state_dict(self) -> dict:
        """Return checkpoint state as a plain dict."""
        if not self._cfg.stateful_dataloader:
            msg = "state_dict() requires stateful_dataloader=True in LoaderConfig."
            raise RuntimeError(msg)
        return self._ckpt.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        """Restore from a state dict produced by state_dict()."""
        self._ckpt.load_state_dict(sd)

    # ── Iteration protocol ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        # [FIX-ITER] Lock prevents TOCTOU race when __iter__ called concurrently.
        with self._iter_lock:
            if self._active_iter:
                msg = (
                    "DINODataLoader: __iter__ called while already iterating. "
                    "Call set_epoch() before starting a new epoch loop."
                )
                raise RuntimeError(msg)
            self._active_iter = True

        try:
            yield from self._raw_iter()
        finally:
            with self._iter_lock:
                self._active_iter = False

    def _raw_iter(self) -> Iterator[Batch]:
        """Core iteration loop — iterates directly over the DALI iterator.

        DALI's internal prefetch queues (cpu_queue / gpu_queue) overlap I/O and
        GPU decode behind this loop naturally, without any application-level
        threading.  Set dali_cpu_queue ≥ 16 in LoaderConfig to ensure the
        queue is deep enough to hide Lustre / extraction latency.
        """
        metrics       = get_registry()
        stall_timeout = self._cfg.stall_timeout_s
        got_first     = False

        for dali_out in self._dali_iter:
            got_first = True
            t0        = time.perf_counter()

            views    = [dali_out[0][name] for name in self._aug_spec.output_map]
            metadata = self._source.pop_last_metadata()

            batch = self._build_batch(views, metadata)

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if metrics:
                metrics.inc("loader_batches_yielded", 1)
                metrics.inc("pipeline_yield_time_ms", elapsed_ms)
                metrics.heartbeat()

            yield batch

        if not got_first and stall_timeout > 0:
            if os.environ.get("DINO_DISABLE_EMPTY_CHECK"):
                log.warning(
                    "DINODataLoader rank %d: no batch produced but "
                    "DINO_DISABLE_EMPTY_CHECK is set — continuing silently.",
                    self._rank,
                )
            else:
                msg = (
                    f"DINODataLoader (rank {self._rank}): no batch produced. "
                    "Possible causes: corrupted shards, /dev/shm full, "
                    "sample_predicate rejected every sample, Lustre MDS slow start. "
                    "Disable: DINO_DISABLE_EMPTY_CHECK=1 or stall_timeout_s=0."
                )
                raise RuntimeError(msg)

    def _build_batch(
        self,
        views:    list[Any],
        metadata: list[dict | None],
    ) -> Batch:
        """Build a Batch from pipeline output views, routing through H2D and FP8."""
        if isinstance(self._aug_spec, DinoV2AugSpec):
            n_global     = self._aug_spec.aug_cfg.n_global_crops
            global_views = views[:n_global]
            local_views  = views[n_global:]
        elif isinstance(self._aug_spec, EvalAugSpec):
            global_views = views
            local_views  = []
        elif isinstance(self._aug_spec, LeJEPAAugSpec):
            global_views = [views[0]]
            local_views  = views[1:]
        elif isinstance(self._aug_spec, UserAugSpec):
            mid          = max(1, len(views) // 2)
            global_views = views[:mid]
            local_views  = views[mid:]
        else:
            global_views = views
            local_views  = []

        with self._h2d.transfer({"global": global_views, "local": local_views}) as gpu:
            g_gpu = gpu["global"]
            l_gpu = gpu["local"]

        if self._fp8 is not None:
            g_gpu = [self._fp8.quantise(t) for t in g_gpu]
            l_gpu = [self._fp8.quantise(t) for t in l_gpu]

        masks = None
        if self._mask_generator is not None and isinstance(self._aug_spec, DinoV2AugSpec):
            n_tokens = (self._current_global_size // 14) ** 2
            masks    = self._mask_generator(n_tokens)

        return Batch(
            global_crops = g_gpu,
            local_crops  = l_gpu,
            metadata     = metadata,
            masks        = masks,
        )

    def __len__(self) -> int:
        """Return steps_per_epoch if set; raises TypeError otherwise."""
        if self._steps_per_epoch is None:
            msg = "len(loader) requires steps_per_epoch to be set at construction."
            raise TypeError(msg)
        return self._steps_per_epoch

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_shard_coverage(self, specs: list[DatasetSpec]) -> None:
        for spec in specs:
            n_shards = len(spec.shards)
            if n_shards < self._world_size:
                log.warning(
                    "DatasetSpec '%s': only %d shards for %d ranks. "
                    "Ranks %d..%d will receive no shards from this dataset. "
                    "Consider shard_sampling='resampled' for small datasets.",
                    spec.name, n_shards, self._world_size,
                    n_shards, self._world_size - 1,
                )

    def _restore(self) -> None:
        state = self._ckpt.load()
        if state is None:
            return
        if state.dataset_names != self._source.dataset_names:
            log.warning(
                "Checkpoint dataset names %s do not match current specs %s — "
                "skipping mixing weight restore.",
                state.dataset_names,
                self._source.dataset_names,
            )
        else:
            self._source.set_weights(state.mixing_weights)
        if state.global_crop_size != self._current_global_size:
            self.set_resolution(state.global_crop_size, state.local_crop_size)

    def _start_prometheus(self, port: int) -> None:
        try:
            import prometheus_client  # noqa: PLC0415
            t = threading.Thread(
                target  = prometheus_client.start_http_server,
                args    = (port,),
                name    = "prometheus-server",
                daemon  = True,
            )
            t.start()
            log.info("Prometheus metrics server started on port %d (rank 0)", port)
        except Exception as exc:
            log.warning("Could not start Prometheus server: %s", exc)

    @staticmethod
    def _infer_rank() -> int:
        for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _infer_world_size() -> int:
        for var in ("WORLD_SIZE", "SLURM_NTASKS"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _infer_local_world_size() -> int:
        for var in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        return 1
