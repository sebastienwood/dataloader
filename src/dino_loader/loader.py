"""dino_loader.loader
==================
DINODataLoader: the single public entry point for training code.

[LD-AUG-1] aug_spec parameter accepts any AugmentationSpec subclass.
           The legacy aug_cfg kwarg is accepted and silently wrapped in
           DinoV2AugSpec(aug_cfg=aug_cfg) for backward compatibility.
[LD-AUG-2] sample_predicate: early filtering before DALI decode.
[B3-FIX]   set_epoch() is protected by threading.Lock.
[LD-13]    current_resolution public property.
[FIX-ENV]  DINODataLoader asserts dino_env.init() before construction in DALI mode.
[FIX-ITER] _active_iter is protected by a threading.Lock.
[CFG-PIPE] PipelineConfig utilisé pour transmettre les paramètres de pipeline.
[CFG-POOL] SharedExtractionPoolConfig transmis à MixingSource.
[FIX-CKPT] state_dict() utilise l'état en mémoire (plus de re-lecture disque).
[FIX-BUILD] _build_batch délégue le split views à AugmentationSpec pour
            éviter les isinstance chains à chaque batch.

Post-processing API
-------------------
Utiliser ``wrap_loader()`` depuis ``dino_loader.pipeline_graph``::

    from dino_loader.pipeline_graph import wrap_loader

    pipeline = (
        wrap_loader(DINODataLoader(...))
        .map(apply_ibot_masks)
        .select(quality_ok)
        .with_epoch(steps_per_epoch)
    )
"""

import logging
import os
import threading
import time
from collections.abc import Iterator, Sequence
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
from dino_loader.config import (
    CheckpointState,
    DINOAugConfig,
    LoaderConfig,
    PipelineConfig,
)
from dino_loader.memory import Batch
from dino_loader.mixing_source import MixingSource, ResolutionSource
from dino_loader.monitor.metrics import get_registry, init_registry

log = logging.getLogger(__name__)


class DINODataLoader:
    """HPC-grade data loader for DINO-style self-supervised training.

    Pour la post-processing pipeline, utiliser ``wrap_loader()`` depuis
    ``dino_loader.pipeline_graph``.

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
        """Construct a DINODataLoader."""
        if aug_spec is None:
            effective_aug_cfg: DINOAugConfig  = aug_cfg if aug_cfg is not None else DINOAugConfig()
            self._aug_spec: AugmentationSpec  = DinoV2AugSpec(aug_cfg=effective_aug_cfg)
        else:
            if aug_cfg is not None:
                log.warning(
                    "DINODataLoader: both aug_spec and aug_cfg provided; "
                    "aug_cfg is ignored (aug_spec takes precedence).",
                )
            self._aug_spec = aug_spec

        self._aug_cfg: DINOAugConfig = (
            self._aug_spec.aug_cfg  # type: ignore[attr-defined]
            if isinstance(self._aug_spec, DinoV2AugSpec)
            else DINOAugConfig()
        )

        self._cfg              = config or LoaderConfig(checkpoint_dir="/tmp/dino_loader_ckpt")
        self._mask_generator   = mask_generator
        self._sample_predicate = sample_predicate
        self._steps_per_epoch  = steps_per_epoch

        self._step:  int = 0
        self._epoch: int = 0

        # [FIX-CKPT] État checkpoint maintenu en mémoire — évite les I/O disque
        # à chaque appel à state_dict().
        self._last_ckpt_state: CheckpointState | None = None

        self._active_iter = False
        self._iter_lock   = threading.Lock()
        self._epoch_lock  = threading.Lock()

        # Backend
        if isinstance(backend, str):
            self._backend: BackendProtocol = get_backend(backend)
        else:
            self._backend = backend

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
        init_registry(
            job_id     = os.environ.get("SLURM_JOB_ID", "dino_local"),
            create     = (self._local_rank == 0),
            local_rank = self._local_rank,
        )

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
            pool_cfg            = self._cfg.extraction_pool,
            seed                = self._cfg.seed,
            device_id           = device_id,
            shuffle_buffer_size = self._cfg.shuffle_buffer_size,
            debug_log_keys      = self._cfg.debug_log_keys,
            sample_predicate    = sample_predicate,
        )
        # Expose la resolution_src sur la source pour que le backend puisse y accéder.
        self._source._resolution_src = self._resolution_src  # type: ignore[attr-defined]
        self._source._batch_size     = batch_size            # type: ignore[attr-defined]

        # Stage 3: augmentation pipeline — [CFG-PIPE]
        pipeline_cfg = PipelineConfig.from_loader_config(
            cfg       = self._cfg,
            device_id = device_id,
            rank      = self._rank,
        )

        pipeline = self._backend.build_pipeline(
            source       = self._source,
            aug_spec     = self._aug_spec,
            pipeline_cfg = pipeline_cfg,
            specs        = specs,
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
            "aug=%s resolution=%dx%d pool_workers=%d",
            self._backend.name,
            self._rank, self._world_size, batch_size,
            type(self._aug_spec).__name__,
            self._current_global_size, self._current_local_size,
            self._cfg.extraction_pool.max_workers,
        )

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def backend(self) -> BackendProtocol:
        """The active backend instance."""
        return self._backend

    @property
    def aug_spec(self) -> AugmentationSpec:
        """The active augmentation spec."""
        return self._aug_spec

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Current crop resolution as (global_size, local_size)."""
        return (self._current_global_size, self._current_local_size)

    @property
    def current_weights(self) -> list[float]:
        """Current normalised mixing weights."""
        return self._source.current_weights

    # ── Augmentation spec helpers ─────────────────────────────────────────────

    def _initial_global_size(self) -> int:
        """Return the initial global crop size from the aug spec."""
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
        """Return the initial local crop size from the aug spec."""
        if isinstance(self._aug_spec, DinoV2AugSpec):
            return self._aug_spec.aug_cfg.local_crop_size
        if isinstance(self._aug_spec, EvalAugSpec):
            return self._aug_spec.crop_size
        if isinstance(self._aug_spec, LeJEPAAugSpec):
            return self._aug_spec.target_crop_size
        if isinstance(self._aug_spec, UserAugSpec):
            return self._aug_spec.decode_size
        return 96

    # ── Epoch / weight / resolution control ───────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Prepare for a new epoch. Thread-safe.

        Args:
            epoch: New epoch number (0-indexed).

        """
        with self._epoch_lock:
            self._epoch = epoch
            if isinstance(self._aug_spec, DinoV2AugSpec):
                new_global = self._aug_spec.aug_cfg.crop_size_at_epoch(epoch)
                if new_global != self._current_global_size:
                    self.set_resolution(new_global, self._current_local_size)

            self._source.set_epoch(epoch)
            self._dali_iter.reset()

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Update crop resolution (only meaningful for DinoV2AugSpec).

        Args:
            global_size: New global crop size in pixels.
            local_size: New local crop size in pixels.

        Raises:
            ValueError: If sizes exceed the pre-allocated maximums.

        """
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

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update mixing weights (re-normalised automatically)."""
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Update one dataset's weight by name."""
        self._source.set_weight_by_name(name, weight)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def checkpoint(self, step: int) -> None:
        """Save a checkpoint (rank 0 only, every N steps).

        Maintient également l'état en mémoire pour que state_dict() n'ait
        pas à relire le fichier depuis le disque.

        Args:
            step: Current training step.

        """
        self._step = step
        state = CheckpointState(
            step             = step,
            epoch            = self._epoch,
            dataset_names    = self._source.dataset_names,
            mixing_weights   = self._source.current_weights,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
        )
        # [FIX-CKPT] Met à jour l'état en mémoire avant l'écriture disque.
        self._last_ckpt_state = state
        self._ckpt.save(state)

    def state_dict(self) -> dict:
        """Return checkpoint state as a plain dict.

        [FIX-CKPT] Utilise l'état en mémoire — pas de re-lecture disque.

        Raises:
            RuntimeError: If ``stateful_dataloader=False`` in ``LoaderConfig``.

        """
        if not self._cfg.stateful_dataloader:
            msg = "state_dict() requires stateful_dataloader=True in LoaderConfig."
            raise RuntimeError(msg)

        if self._last_ckpt_state is not None:
            return self._last_ckpt_state.to_dict()

        # Pas encore de checkpoint sauvegardé — retourne l'état courant.
        return CheckpointState(
            step             = self._step,
            epoch            = self._epoch,
            dataset_names    = self._source.dataset_names,
            mixing_weights   = self._source.current_weights,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
        ).to_dict()

    def load_state_dict(self, sd: dict) -> None:
        """Restore from a state dict produced by state_dict().

        Args:
            sd: State dict from a prior :meth:`state_dict` call.

        """
        if "epoch" in sd:
            self._epoch = sd["epoch"]
        if "step" in sd:
            self._step = sd["step"]
        if "global_crop_size" in sd:
            self.set_resolution(
                sd["global_crop_size"],
                sd.get("local_crop_size", self._current_local_size),
            )
        if "mixing_weights" in sd and "dataset_names" in sd:
            if sd["dataset_names"] == self._source.dataset_names:
                self._source.set_weights(sd["mixing_weights"])

    # ── Iteration protocol ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over training batches.

        Raises:
            RuntimeError: If called while already iterating.

        """
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
        """Core iteration loop — drives the DALI/CPU iterator directly."""
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
        """Build a Batch from pipeline output views.

        [FIX-BUILD] Le split views→global/local est délégué à aug_spec pour
        éviter les isinstance chains répétées à chaque batch dans le hot path.

        Args:
            views: List of tensors from the augmentation pipeline.
            metadata: Per-sample sidecar dicts from MixingSource.

        Returns:
            A fully assembled ``Batch`` on the target device.

        """
        global_views, local_views = self._aug_spec_split_views(views)

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

    def _aug_spec_split_views(
        self,
        views: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Split the flat views list into (global_crops, local_crops).

        Chaque sous-type d'AugmentationSpec a une convention différente.
        Cette méthode centralise la logique pour éviter de la répéter dans
        _build_batch avec des isinstance à chaque batch.

        Args:
            views: Flat list of tensors from the augmentation pipeline.

        Returns:
            ``(global_views, local_views)`` — two lists of tensors.

        """
        if isinstance(self._aug_spec, DinoV2AugSpec):
            n_global = self._aug_spec.aug_cfg.n_global_crops
            return views[:n_global], views[n_global:]
        if isinstance(self._aug_spec, EvalAugSpec):
            return views, []
        if isinstance(self._aug_spec, LeJEPAAugSpec):
            return [views[0]], views[1:]
        if isinstance(self._aug_spec, UserAugSpec):
            mid = max(1, len(views) // 2)
            return views[:mid], views[mid:]
        return views, []

    def __len__(self) -> int:
        """Return steps_per_epoch if set; raises TypeError otherwise."""
        if self._steps_per_epoch is None:
            msg = "len(loader) requires steps_per_epoch to be set at construction."
            raise TypeError(msg)
        return self._steps_per_epoch

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_shard_coverage(self, specs: list[DatasetSpec]) -> None:
        """Warn if any dataset has fewer shards than ranks."""
        for spec in specs:
            n_shards = len(spec.shards)
            if n_shards < self._world_size:
                log.warning(
                    "DatasetSpec '%s': only %d shards for %d ranks. "
                    "Consider shard_sampling='resampled' for small datasets.",
                    spec.name, n_shards, self._world_size,
                )

    def _restore(self) -> None:
        """Load the latest checkpoint and apply its state."""
        state = self._ckpt.load()
        if state is None:
            return
        self._last_ckpt_state = state
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
        self._epoch = state.epoch
        self._step  = state.step

    def _start_prometheus(self, port: int) -> None:
        """Start a Prometheus HTTP metrics server on *port* (rank 0 only)."""
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
        """Infer global rank from environment variables."""
        for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _infer_world_size() -> int:
        """Infer world size from environment variables."""
        for var in ("WORLD_SIZE", "SLURM_NTASKS"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _infer_local_world_size() -> int:
        """Infer local world size from environment variables."""
        for var in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        return 1
