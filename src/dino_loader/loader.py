"""dino_loader.loader
==================
DINODataLoader: the single public entry point for training code.

Architecture finale (Phase 1/3 + wrap_loader implicite)
---------------------------------------------------------
DINODataLoader délègue entièrement à ``NodePipeline`` pour l'itération.
Toute la logique de boucle, métriques et stall detection vit dans
``_DALINode`` (``pipeline_graph.py``).

Flux de construction :

    DINODataLoader.__init__
        → ShardReaderNode           (stage 1-2 : I/O + mixing)
        → _ReaderAdapter            (bridge ShardReaderNode → callable DALI)
        → BackendProtocol.build_*   (stage 3-5 : DALI/CPU + H2D + FP8)
        → _DALINode                 (BaseNode : pilote DALI, assemble Batch)
        → NodePipeline              (self._pipeline = wrap_loader(self))

Flux d'itération :

    for batch in loader:          # délègue à self._pipeline
        ...

Corrections
-----------
[FIX-ENV] L'objet ``DistribEnv`` retourné par ``init_distributed()`` est
    conservé tel quel (self._env) plutôt que d'en dépouiller chaque attribut.
    Les propriétés délèguent à self._env pour éviter la duplication.
[FIX-ACTIVE-ITER] Guard contre le double-iter : __iter__ lève RuntimeError
    si une itération est déjà en cours.
[FIX-RESET-ITER] set_epoch() appelle _dali_node.reset_iter() (méthode
    thread-safe avec lock interne) au lieu d'accéder à _iter directement.
[FIX-META-FIFO] _ReaderAdapter utilise une queue.Queue pour aligner les
    métadonnées sur les batches dans l'ordre FIFO strict.
[FIX-FUTURE] from __future__ import annotations supprimé (Python ≥ 3.12 natif).
"""

import logging
import os
import queue
import threading
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
from dino_loader.config import (
    CheckpointState,
    DINOAugConfig,
    LoaderConfig,
    PipelineConfig,
)
from dino_loader.memory import Batch
from dino_loader.mixing_source import ResolutionSource
from dino_loader.monitor.metrics import init_registry
from dino_loader.nodes import ShardReaderNode
from dino_loader.pipeline_graph import NodePipeline, _DALINode, wrap_loader

log = logging.getLogger(__name__)


class DINODataLoader:
    """HPC-grade data loader for DINO-style self-supervised training.

    L'itération est entièrement déléguée à un ``NodePipeline`` interne.

    Usage minimal::

        loader = DINODataLoader(specs, batch_size=512, ...)
        for epoch in range(100):
            loader.set_epoch(epoch)
            for batch in loader:
                train_step(batch)

    Usage avec composition::

        pipeline = (
            loader.as_pipeline()
            .map(apply_ibot_masks)
            .select(quality_ok)
            .with_epoch(steps_per_epoch)
        )
        for epoch in range(100):
            pipeline.set_epoch(epoch)
            for batch in pipeline:
                train_step(batch)

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
        # Rétrocompat : wrapping du legacy aug_cfg.
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

        self._cfg = config or LoaderConfig(stateful_dataloader=False, checkpoint_dir="")
        self._mask_generator  = mask_generator
        self._steps_per_epoch = steps_per_epoch

        self._step:  int = 0
        self._epoch: int = 0
        self._last_ckpt_state: CheckpointState | None = None

        self._epoch_lock  = threading.Lock()
        # [FIX-ACTIVE-ITER] Flag thread-safe pour détecter les double-iter.
        self._active_iter = False
        self._active_iter_lock = threading.Lock()

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

        # [FIX-ENV] Conserver l'objet DistribEnv complet plutôt que d'en
        # dépouiller chaque attribut individuellement.
        self._env = self._backend.init_distributed(
            rank             = rank             if rank             is not None else self._infer_rank(),
            world_size       = world_size       if world_size       is not None else self._infer_world_size(),
            local_rank       = local_rank       if local_rank       is not None else device_id,
            local_world_size = local_world_size if local_world_size is not None else self._infer_local_world_size(),
            force_topology   = self._cfg.force_topology,
        )

        init_registry(
            job_id     = os.environ.get("SLURM_JOB_ID", "dino_local"),
            create     = (self._env.local_rank == 0),
            local_rank = self._env.local_rank,
        )

        if self._cfg.prometheus_port is not None and self._env.rank == 0:
            self._start_prometheus(self._cfg.prometheus_port)

        # Resolution tracking
        self._current_global_size = self._initial_global_size()
        self._current_local_size  = self._initial_local_size()
        self._resolution_src      = ResolutionSource(
            self._current_global_size,
            self._current_local_size,
        )

        # Stage 1 : shard cache
        node_master = (self._env.local_rank == 0)
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

        # Stage 1-2 : ShardReaderNode (source unique)
        self._reader = ShardReaderNode(
            specs               = specs,
            batch_size          = batch_size,
            cache               = shard_cache,
            rank                = self._env.rank,
            world_size          = self._env.world_size,
            pool_cfg            = self._cfg.extraction_pool,
            seed                = self._cfg.seed,
            device_id           = device_id,
            shuffle_buffer_size = self._cfg.shuffle_buffer_size,
            debug_log_keys      = self._cfg.debug_log_keys,
            sample_predicate    = sample_predicate,
        )
        self._reader.reset()

        # Adaptateur ShardReaderNode → callable DALI/CPU
        self._dali_source = _ReaderAdapter(
            reader         = self._reader,
            resolution_src = self._resolution_src,
            batch_size     = batch_size,
        )

        # Stage 3 : pipeline d'augmentation
        pipeline_cfg = PipelineConfig.from_loader_config(
            cfg       = self._cfg,
            device_id = device_id,
            rank      = self._env.rank,
        )
        dali_pipeline = self._backend.build_pipeline(
            source       = self._dali_source,
            aug_spec     = self._aug_spec,
            pipeline_cfg = pipeline_cfg,
            specs        = specs,
        )
        dali_iter = self._backend.build_pipeline_iterator(
            pipeline   = dali_pipeline,
            aug_spec   = self._aug_spec,
            output_map = self._aug_spec.output_map,
            batch_size = batch_size,
        )

        # Stage 4-5 : H2D + FP8
        device   = torch.device(f"cuda:{device_id}") if self._backend.supports_gpu else torch.device("cpu")
        dali_fp8 = self._cfg.use_fp8_output and self._cfg.dali_fp8_output
        h2d_topo = getattr(self._env, "topology", None)
        self._h2d = self._backend.build_h2d_stream(device=device, topo=h2d_topo)
        self._fp8 = (
            self._backend.build_fp8_formatter()
            if self._cfg.use_fp8_output and not dali_fp8
            else None
        )

        # _DALINode : pilote DALI, assemble Batch, métriques, stall watchdog
        self._dali_node = _DALINode(
            dali_iter_factory = lambda: dali_iter,
            pop_metadata_fn   = self._dali_source.pop_last_metadata,
            build_batch_fn    = self._build_batch,
            output_map        = self._aug_spec.output_map,
            stall_timeout_s   = self._cfg.stall_timeout_s,
            rank              = self._env.rank,
        )

        # NodePipeline interne : c'est ce que __iter__ utilise.
        self._pipeline: NodePipeline = wrap_loader(self)

        # Checkpointing
        ckpt_dir   = self._cfg.checkpoint_dir if self._cfg.stateful_dataloader else "/tmp"
        self._ckpt = DataLoaderCheckpointer(
            ckpt_dir      = ckpt_dir,
            every_n_steps = self._cfg.checkpoint_every_steps,
            rank          = self._env.rank,
        )

        if resume:
            self._restore()

        log.info(
            "DINODataLoader ready: backend=%s rank=%d/%d batch=%d "
            "aug=%s resolution=%dx%d pool_workers=%d",
            self._backend.name,
            self._env.rank, self._env.world_size, batch_size,
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
        return self._reader.current_weights

    # ── Convenience accessors for distributed env ─────────────────────────────
    # [FIX-ENV] Plutôt que de dupliquer chaque attribut de DistribEnv, on
    # délègue via des propriétés.  Le code interne utilise self._env directement
    # pour les accès groupés.

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return self._env.rank

    @property
    def world_size(self) -> int:
        """Total number of processes."""
        return self._env.world_size

    @property
    def local_rank(self) -> int:
        """Local rank within the node."""
        return self._env.local_rank

    @property
    def local_world_size(self) -> int:
        """Number of processes on this node."""
        return self._env.local_world_size

    # ── Composition API ───────────────────────────────────────────────────────

    def as_pipeline(self) -> NodePipeline:
        """Return the internal NodePipeline for composition."""
        return self._pipeline

    def map(self, fn: Callable[[Batch], Batch], *, label: str = "<map>") -> NodePipeline:
        """Shortcut : ``loader.map(fn)`` équivaut à ``loader.as_pipeline().map(fn)``."""
        return self._pipeline.map(fn, label=label)

    def select(
        self,
        predicate: Callable[[Batch], bool],
        *,
        label: str = "<filter>",
    ) -> NodePipeline:
        """Shortcut : ``loader.select(pred)`` équivaut à ``loader.as_pipeline().select(pred)``."""
        return self._pipeline.select(predicate, label=label)

    def with_epoch(self, n_steps: int) -> NodePipeline:
        """Shortcut : ``loader.with_epoch(n)`` équivaut à ``loader.as_pipeline().with_epoch(n)``."""
        return self._pipeline.with_epoch(n_steps)

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
        """Prepare for a new epoch.

        Args:
            epoch: New epoch number (0-indexed).

        """
        with self._epoch_lock:
            self._epoch = epoch
            if isinstance(self._aug_spec, DinoV2AugSpec):
                new_global = self._aug_spec.aug_cfg.crop_size_at_epoch(epoch)
                if new_global != self._current_global_size:
                    self.set_resolution(new_global, self._current_local_size)
            self._reader.set_epoch(epoch)
            # [FIX-RESET-ITER] Utilise la méthode thread-safe au lieu d'accéder
            # à _iter directement depuis l'extérieur.
            self._dali_node.reset_iter()

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Update crop resolution.

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
                "set_resolution called on a %s — no effect for non-DinoV2 specs.",
                type(self._aug_spec).__name__,
            )
        self._current_global_size = global_size
        self._current_local_size  = local_size
        self._resolution_src.set(global_size, local_size)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update mixing weights (re-normalised automatically)."""
        self._reader.set_weights(list(weights))

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Update one dataset's weight by name."""
        self._reader.set_weight_by_name(name, weight)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def checkpoint(self, step: int) -> None:
        """Save a checkpoint (rank 0 only, every N steps)."""
        self._step = step
        state = CheckpointState(
            step             = step,
            epoch            = self._epoch,
            dataset_names    = self._reader.dataset_names,
            mixing_weights   = self._reader.current_weights,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
        )
        self._last_ckpt_state = state
        self._ckpt.save(state)

    def state_dict(self) -> dict:
        """Return checkpoint state as a plain dict (from memory, no disk I/O).

        Raises:
            RuntimeError: If ``stateful_dataloader=False`` in ``LoaderConfig``.

        """
        if not self._cfg.stateful_dataloader:
            msg = "state_dict() requires stateful_dataloader=True in LoaderConfig."
            raise RuntimeError(msg)

        if self._last_ckpt_state is not None:
            return self._last_ckpt_state.to_dict()

        return CheckpointState(
            step             = self._step,
            epoch            = self._epoch,
            dataset_names    = self._reader.dataset_names,
            mixing_weights   = self._reader.current_weights,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
        ).to_dict()

    def load_state_dict(self, sd: dict) -> None:
        """Restore from a state dict produced by state_dict()."""
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
            if sd["dataset_names"] == self._reader.dataset_names:
                self._reader.set_weights(sd["mixing_weights"])

    # ── Iteration — délégué au NodePipeline interne ───────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over training batches via the internal NodePipeline.

        [FIX-ACTIVE-ITER] Lève RuntimeError si une itération est déjà active.
        Cela prévient les bugs silencieux où deux boucles for consomment le
        même flux de batches de façon entremêlée.

        Raises:
            RuntimeError: Si ``__iter__`` est appelé pendant une itération active.

        """
        with self._active_iter_lock:
            if self._active_iter:
                msg = (
                    "DINODataLoader is already iterating. "
                    "Cannot start a second iteration on the same loader. "
                    "Call set_epoch() to start a new epoch, or use "
                    "loader.as_pipeline() for concurrent-safe composition."
                )
                raise RuntimeError(msg)
            self._active_iter = True

        try:
            yield from self._pipeline
        finally:
            with self._active_iter_lock:
                self._active_iter = False

    def __len__(self) -> int:
        """Return steps_per_epoch if set; raises TypeError otherwise."""
        if self._steps_per_epoch is None:
            msg = "len(loader) requires steps_per_epoch to be set at construction."
            raise TypeError(msg)
        return self._steps_per_epoch

    # ── Batch assembly ────────────────────────────────────────────────────────

    def _build_batch(
        self,
        views:    list[Any],
        metadata: list[dict | None],
    ) -> Batch:
        """Build a Batch from pipeline output views.

        Args:
            views: Flat list of tensors from the augmentation pipeline.
            metadata: Per-sample sidecar dicts.

        Returns:
            A fully assembled ``Batch`` on the target device.

        """
        global_views, local_views = self._split_views(views)

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

    def _split_views(self, views: list[Any]) -> tuple[list[Any], list[Any]]:
        """Split the flat views list into (global_crops, local_crops)."""
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_shard_coverage(self, specs: list[DatasetSpec]) -> None:
        """Warn if any dataset has fewer shards than ranks."""
        for spec in specs:
            n_shards = len(spec.shards)
            if n_shards < self._env.world_size:
                log.warning(
                    "DatasetSpec '%s': only %d shards for %d ranks. "
                    "Consider shard_sampling='resampled' for small datasets.",
                    spec.name, n_shards, self._env.world_size,
                )

    def _restore(self) -> None:
        """Load the latest checkpoint and apply its state."""
        state = self._ckpt.load()
        if state is None:
            return
        self._last_ckpt_state = state
        if state.dataset_names != self._reader.dataset_names:
            log.warning(
                "Checkpoint dataset names %s do not match current specs %s — "
                "skipping mixing weight restore.",
                state.dataset_names, self._reader.dataset_names,
            )
        else:
            self._reader.set_weights(state.mixing_weights)
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


# ---------------------------------------------------------------------------
# _ReaderAdapter — bridge ShardReaderNode → callable DALI/CPU ExternalSource
# ---------------------------------------------------------------------------


class _ReaderAdapter:
    """Adaptateur exposant ShardReaderNode comme callable DALI/CPU source.

    [FIX-META-FIFO] _last_metadata est une queue.Queue (FIFO) plutôt qu'une
    variable scalaire.  Cela garantit l'alignement strict entre les batches
    DALI et leurs métadonnées même quand DALI appelle ``__call__()`` de façon
    préemptive depuis son thread de prefetch.

    Chaque appel à ``__call__()`` pousse les métadonnées dans la queue.
    ``pop_last_metadata()`` dépile la valeur la plus ancienne, préservant
    ainsi l'ordre FIFO entre batches.
    """

    # Taille maximale de la queue de métadonnées.  En pratique, DALI ne
    # précharge pas plus de cpu_queue + gpu_queue batches en avance.
    _META_QUEUE_MAXSIZE: int = 32

    def __init__(
        self,
        reader:         ShardReaderNode,
        resolution_src: ResolutionSource,
        batch_size:     int,
    ) -> None:
        """Initialise the reader adapter."""
        self._reader         = reader
        self._resolution_src = resolution_src
        self._batch_size     = batch_size
        # [FIX-META-FIFO] Queue FIFO pour l'alignement métadonnées ↔ batches.
        self._meta_queue: queue.Queue[list[dict | None]] = queue.Queue(
            maxsize=self._META_QUEUE_MAXSIZE,
        )

    def __call__(self) -> list:
        """Return one batch of JPEG arrays (called by DALI per step)."""
        jpegs, metadata = self._reader.next()
        # [FIX-META-FIFO] Enqueue pour alignement FIFO avec le batch DALI.
        # put_nowait() : si la queue est pleine (DALI prefetch très agressif),
        # on lève queue.Full plutôt que de corrompre silencieusement l'ordre.
        try:
            self._meta_queue.put_nowait(metadata)
        except queue.Full:
            log.warning(
                "_ReaderAdapter: metadata queue full (%d slots). "
                "DALI prefetch depth may exceed _META_QUEUE_MAXSIZE=%d. "
                "Increase _META_QUEUE_MAXSIZE or reduce DALI queue depths.",
                self._META_QUEUE_MAXSIZE, self._META_QUEUE_MAXSIZE,
            )
            # Fallback : vider un slot pour ne pas bloquer DALI.
            try:
                self._meta_queue.get_nowait()
            except queue.Empty:
                pass
            self._meta_queue.put_nowait(metadata)
        return jpegs

    def pop_last_metadata(self) -> list[dict | None]:
        """Return metadata for the oldest unconsumed batch (FIFO).

        [FIX-META-FIFO] Dépile dans l'ordre FIFO pour garantir l'alignement
        avec les batches retournés par DALI.
        """
        try:
            return self._meta_queue.get_nowait()
        except queue.Empty:
            log.debug("_ReaderAdapter.pop_last_metadata: queue empty, returning []")
            return []

    def register_dataset_index_callback(self, cb: Any) -> None:
        """Propagate NormSource callbacks to the inner MixingSource."""
        source = self._reader._source  # type: ignore[attr-defined]
        if source is not None and hasattr(source, "register_dataset_index_callback"):
            source.register_dataset_index_callback(cb)
        else:
            log.warning(
                "_ReaderAdapter: register_dataset_index_callback called before "
                "ShardReaderNode.reset() — callback will not be registered.",
            )
