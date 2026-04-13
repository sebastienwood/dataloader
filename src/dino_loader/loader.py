"""dino_loader.loader
==================
DINODataLoader : point d'entrée public unique pour le code d'entraînement.

Architecture
------------
``DINODataLoader`` orchestre les 5 stages du pipeline sans contenir de logique
d'augmentation ni de post-traitement.  Son constructeur est décomposé en
méthodes privées nommées pour la lisibilité.

Flux de construction ::

    __init__
      → _build_cache()            stage 1 : NodeSharedShardCache
      → _build_reader()           stage 1-2 : ShardReaderNode + _ReaderAdapter
      → _build_augmentation()     stage 3 : pipeline DALI/CPU + itérateur
      → _build_transfer()         stage 4-5 : H2D + FP8
      → _build_dali_node()        _DALINode : pilote le pipeline, assemble Batch
      → _build_pipeline()         NodePipeline interne

Flux d'itération ::

    for batch in loader:   # délègue à self._pipeline
        ...

Séparation des responsabilités
--------------------------------
- La logique de split des vues (global/local) vit dans ``AugmentationSpec``
  via ``spec.split_views(views)`` — aucun ``isinstance`` dans loader.py.
- La logique de sérialisation des checkpoints vit dans ``checkpoint.py``.
- ``_DALINode`` (pilotage de l'itérateur backend) vit dans ``dali_node.py``.
- ``_ReaderAdapter`` (bridge ShardReaderNode → callable) vit dans ``shard_reader.py``.

Corrections intégrées
---------------------
[FIX-ENV]          DistribEnv conservé tel quel.
[FIX-ACTIVE-ITER]  Guard contre le double-iter via _active_iter + lock.
[FIX-RESET-ITER]   set_epoch() appelle _dali_node.reset_iter() (thread-safe).
"""

import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
import torch.distributed as dist
from dino_datasets import DatasetSpec

from dino_loader.augmentation import AugmentationSpec, DinoV2AugSpec, SamplePredicate
from dino_loader.backends import get_backend
from dino_loader.backends.protocol import BackendProtocol
from dino_loader.checkpoint import DataLoaderCheckpointer
from dino_loader.config import (
    CheckpointState,
    DINOAugConfig,
    LoaderConfig,
    PipelineConfig,
)
from dino_loader.dali_node import _DALINode
from dino_loader.memory import Batch
from dino_loader.monitor.metrics import init_registry
from dino_loader.pipeline_graph import NodePipeline, wrap_loader
from dino_loader.shard_reader import ShardReaderNode, _ReaderAdapter
from dino_loader.sources.resolution import ResolutionSource

log = logging.getLogger(__name__)


class DINODataLoader:
    """Dataloader HPC pour l'entraînement auto-supervisé de style DINO.

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

    Args:
        specs:            Liste de spécifications de datasets.
        batch_size:       Taille de batch par GPU.
        aug_spec:         Stratégie d'augmentation. Défaut : DinoV2AugSpec.
        aug_cfg:          Paramètre legacy. Enveloppé dans DinoV2AugSpec si
                          aug_spec est None.
        config:           Configuration runtime du loader.
        device_id:        Index GPU local.
        rank:             Rang global (inféré depuis l'environnement si None).
        world_size:       Nombre total de rangs (inféré si None).
        local_rank:       Rang local (défaut : device_id).
        local_world_size: Rangs par nœud (inféré si None).
        resume:           Charge le dernier checkpoint à la construction.
        steps_per_epoch:  Active ``len(loader)``.
        mask_generator:   MaskingGenerator iBOT optionnel (CPU, post-DALI).
        sample_predicate: Filtre anticipé avant le décodage DALI.
        backend:          ``"auto"`` | ``"dali"`` | ``"cpu"`` | instance backend.

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
        self._aug_spec        = self._resolve_aug_spec(aug_spec, aug_cfg)
        self._cfg             = config or LoaderConfig(stateful_dataloader=False, checkpoint_dir="")
        self._mask_generator  = mask_generator
        self._steps_per_epoch = steps_per_epoch

        self._step:  int = 0
        self._epoch: int = 0
        self._last_ckpt_state: CheckpointState | None = None

        self._epoch_lock       = threading.Lock()
        self._active_iter      = False
        self._active_iter_lock = threading.Lock()

        self._backend = self._init_backend(backend)
        self._env     = self._init_distributed(
            rank, world_size, local_rank, local_world_size, device_id,
        )

        init_registry(
            job_id     = os.environ.get("SLURM_JOB_ID", "dino_local"),
            create     = (self._env.local_rank == 0),
            local_rank = self._env.local_rank,
        )

        if self._cfg.prometheus_port is not None and self._env.rank == 0:
            self._start_prometheus(self._cfg.prometheus_port)

        self._current_global_size = self._aug_spec.initial_global_size
        self._current_local_size  = self._aug_spec.initial_local_size
        self._resolution_src      = ResolutionSource(
            self._current_global_size,
            self._current_local_size,
        )

        self._validate_shard_coverage(specs)

        shard_cache              = self._build_cache(device_id)
        self._reader, self._dali_source = self._build_reader(
            specs, batch_size, shard_cache, device_id, sample_predicate,
        )
        dali_iter                = self._build_augmentation(specs, batch_size, device_id)
        self._h2d, self._fp8    = self._build_transfer(device_id)
        self._dali_node          = self._build_dali_node(dali_iter)
        self._pipeline           = wrap_loader(self)

        self._ckpt = DataLoaderCheckpointer(
            ckpt_dir      = self._cfg.checkpoint_dir if self._cfg.stateful_dataloader else "/tmp",
            every_n_steps = self._cfg.checkpoint_every_steps,
            rank          = self._env.rank,
        )

        if resume:
            self._restore()

        log.info(
            "DINODataLoader prêt : backend=%s rank=%d/%d batch=%d "
            "aug=%s résolution=%dx%d pool_workers=%d dtype=%s",
            self._backend.name,
            self._env.rank, self._env.world_size, batch_size,
            type(self._aug_spec).__name__,
            self._current_global_size, self._current_local_size,
            self._cfg.extraction_pool.max_workers,
            self._cfg.output_dtype,
        )

    # ── Propriétés publiques ──────────────────────────────────────────────────

    @property
    def backend(self) -> BackendProtocol:
        return self._backend

    @property
    def aug_spec(self) -> AugmentationSpec:
        return self._aug_spec

    @property
    def current_resolution(self) -> tuple[int, int]:
        return (self._current_global_size, self._current_local_size)

    @property
    def current_weights(self) -> list[float]:
        return self._reader.current_weights

    @property
    def rank(self) -> int:
        return self._env.rank

    @property
    def world_size(self) -> int:
        return self._env.world_size

    @property
    def local_rank(self) -> int:
        return self._env.local_rank

    @property
    def local_world_size(self) -> int:
        return self._env.local_world_size

    # ── API de composition ────────────────────────────────────────────────────

    def as_pipeline(self) -> NodePipeline:
        return self._pipeline

    def map(self, fn: Callable[[Batch], Batch], *, label: str = "<map>") -> NodePipeline:
        return self._pipeline.map(fn, label=label)

    def select(
        self,
        predicate: Callable[[Batch], bool],
        *,
        label: str = "<filter>",
    ) -> NodePipeline:
        return self._pipeline.select(predicate, label=label)

    def with_epoch(self, n_steps: int) -> NodePipeline:
        return self._pipeline.with_epoch(n_steps)

    # ── Contrôle époque / poids / résolution ──────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Prépare le loader pour une nouvelle époque."""
        with self._epoch_lock:
            self._epoch = epoch
            if isinstance(self._aug_spec, DinoV2AugSpec):
                new_global = self._aug_spec.aug_cfg.crop_size_at_epoch(epoch)
                if new_global != self._current_global_size:
                    self.set_resolution(new_global, self._current_local_size)
            self._reader.set_epoch(epoch)
            self._dali_node.reset_iter()

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Met à jour la résolution de crop.

        Raises:
            ValueError: Si les tailles dépassent les maximums pré-alloués.

        """
        if isinstance(self._aug_spec, DinoV2AugSpec):
            cfg = self._aug_spec.aug_cfg
            if global_size > cfg.max_global_crop_size:
                msg = (
                    f"set_resolution: global_size={global_size} dépasse "
                    f"max_global_crop_size={cfg.max_global_crop_size}."
                )
                raise ValueError(msg)
            if local_size > cfg.max_local_crop_size:
                msg = (
                    f"set_resolution: local_size={local_size} dépasse "
                    f"max_local_crop_size={cfg.max_local_crop_size}."
                )
                raise ValueError(msg)
        else:
            log.warning(
                "set_resolution appelé sur un %s — sans effet pour les specs non-DinoV2.",
                type(self._aug_spec).__name__,
            )
        self._current_global_size = global_size
        self._current_local_size  = local_size
        self._resolution_src.set(global_size, local_size)

    def set_weights(self, weights: Sequence[float]) -> None:
        self._reader.set_weights(list(weights))

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._reader.set_weight_by_name(name, weight)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def checkpoint(self, step: int) -> None:
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
        """Retourne l'état en dict (depuis la mémoire, sans I/O disque).

        Raises:
            RuntimeError: Si ``stateful_dataloader=False``.

        """
        if not self._cfg.stateful_dataloader:
            msg = "state_dict() requiert stateful_dataloader=True dans LoaderConfig."
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

    # ── Itération ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        """Itère via le NodePipeline interne.

        Raises:
            RuntimeError: Si une itération est déjà active.

        """
        with self._active_iter_lock:
            if self._active_iter:
                msg = (
                    "DINODataLoader est déjà en cours d'itération. "
                    "Impossible de démarrer une seconde itération simultanée."
                )
                raise RuntimeError(msg)
            self._active_iter = True

        try:
            yield from self._pipeline
        finally:
            with self._active_iter_lock:
                self._active_iter = False

    def __len__(self) -> int:
        if self._steps_per_epoch is None:
            msg = "len(loader) requiert que steps_per_epoch soit défini à la construction."
            raise TypeError(msg)
        return self._steps_per_epoch

    # ── Construction par étapes ───────────────────────────────────────────────

    @staticmethod
    def _resolve_aug_spec(
        aug_spec: AugmentationSpec | None,
        aug_cfg:  DINOAugConfig | None,
    ) -> AugmentationSpec:
        if aug_spec is not None:
            if aug_cfg is not None:
                log.warning(
                    "DINODataLoader: aug_spec et aug_cfg fournis simultanément ; "
                    "aug_cfg est ignoré (aug_spec a la priorité).",
                )
            return aug_spec
        effective = aug_cfg if aug_cfg is not None else DINOAugConfig()
        return DinoV2AugSpec(aug_cfg=effective)

    def _init_backend(self, backend: Any) -> BackendProtocol:
        if isinstance(backend, str):
            result = get_backend(backend)
        else:
            result = backend

        if result.name == "dali":
            try:
                import dino_env  # noqa: PLC0415
                dino_env.get_env()
            except RuntimeError:
                msg = (
                    "DINODataLoader avec le backend DALI requiert que "
                    "dino_env.init() soit appelé avant la construction du loader."
                )
                raise RuntimeError(msg) from None
        return result

    def _init_distributed(
        self,
        rank:             int | None,
        world_size:       int | None,
        local_rank:       int | None,
        local_world_size: int | None,
        device_id:        int,
    ) -> Any:
        return self._backend.init_distributed(
            rank             = rank             if rank             is not None else self._infer_rank(),
            world_size       = world_size       if world_size       is not None else self._infer_world_size(),
            local_rank       = local_rank       if local_rank       is not None else device_id,
            local_world_size = local_world_size if local_world_size is not None else self._infer_local_world_size(),
            force_topology   = self._cfg.force_topology,
        )

    def _build_cache(self, device_id: int) -> Any:
        """Stage 1 : construit le cache de shards."""
        node_master = (self._env.local_rank == 0)
        job_id      = os.environ.get("SLURM_JOB_ID", "dino_local")
        return self._backend.build_shard_cache(
            job_id            = job_id,
            node_master       = node_master,
            max_gb            = self._cfg.node_shm_gb,
            prefetch_window   = self._cfg.shard_prefetch_window,
            timeout_s         = self._cfg.shard_timeout_s,
            warn_threshold    = self._cfg.shm_warn_threshold,
            heartbeat_stale_s = self._cfg.heartbeat_stale_s,
        )

    def _build_reader(
        self,
        specs:            list[DatasetSpec],
        batch_size:       int,
        shard_cache:      Any,
        device_id:        int,
        sample_predicate: SamplePredicate | None,
    ) -> tuple[ShardReaderNode, _ReaderAdapter]:
        """Stages 1-2 : construit le ShardReaderNode et son adaptateur."""
        reader = ShardReaderNode(
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
        reader.reset()

        adapter = _ReaderAdapter(
            reader         = reader,
            resolution_src = self._resolution_src,
            batch_size     = batch_size,
        )
        return reader, adapter

    def _build_augmentation(
        self,
        specs:      list[DatasetSpec],
        batch_size: int,
        device_id:  int,
    ) -> Any:
        """Stage 3 : construit le pipeline d'augmentation et retourne l'itérateur."""
        pipeline_cfg = PipelineConfig.from_loader_config(
            cfg       = self._cfg,
            device_id = device_id,
            rank      = self._env.rank,
        )
        pipeline = self._backend.build_pipeline(
            source       = self._dali_source,
            aug_spec     = self._aug_spec,
            pipeline_cfg = pipeline_cfg,
            specs        = specs,
        )
        return self._backend.build_pipeline_iterator(
            pipeline   = pipeline,
            aug_spec   = self._aug_spec,
            output_map = self._aug_spec.output_map,
            batch_size = batch_size,
        )

    def _build_transfer(self, device_id: int) -> tuple[Any, Any]:
        """Stages 4-5 : construit le stream H2D et le formatter FP8."""
        device   = torch.device(f"cuda:{device_id}") if self._backend.supports_gpu else torch.device("cpu")
        dali_fp8 = self._cfg.use_fp8_output and self._cfg.dali_fp8_output
        h2d_topo = getattr(self._env, "topology", None)
        h2d      = self._backend.build_h2d_stream(device=device, topo=h2d_topo)
        fp8      = (
            self._backend.build_fp8_formatter()
            if self._cfg.use_fp8_output and not dali_fp8
            else None
        )
        return h2d, fp8

    def _build_dali_node(self, dali_iter: Any) -> _DALINode:
        """Construit le _DALINode qui pilote l'itérateur et assemble les Batch."""
        return _DALINode(
            dali_iter_factory = lambda: dali_iter,
            pop_metadata_fn   = self._dali_source.pop_last_metadata,
            build_batch_fn    = self._assemble_batch,
            output_map        = self._aug_spec.output_map,
            stall_timeout_s   = self._cfg.stall_timeout_s,
            rank              = self._env.rank,
        )

    # ── Assemblage du Batch ───────────────────────────────────────────────────

    def _assemble_batch(
        self,
        views:    list[Any],
        metadata: list[dict | None],
    ) -> Batch:
        """Assemble un Batch depuis les vues du pipeline."""
        global_views, local_views = self._aug_spec.split_views(views)

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

    # ── Helpers internes ──────────────────────────────────────────────────────

    def _validate_shard_coverage(self, specs: list[DatasetSpec]) -> None:
        for spec in specs:
            if len(spec.shards) < self._env.world_size:
                log.warning(
                    "DatasetSpec '%s' : seulement %d shard(s) pour %d rangs. "
                    "Envisager shard_sampling='resampled'.",
                    spec.name, len(spec.shards), self._env.world_size,
                )

    def _restore(self) -> None:
        """Charge le dernier checkpoint et applique son état."""
        state = self._ckpt.load()
        if state is None:
            return
        self._last_ckpt_state = state
        if state.dataset_names != self._reader.dataset_names:
            log.warning(
                "Les noms de datasets du checkpoint %s ne correspondent pas "
                "aux specs courantes %s — restauration des poids ignorée.",
                state.dataset_names, self._reader.dataset_names,
            )
        else:
            self._reader.set_weights(state.mixing_weights)

        if (
            state.global_crop_size != self._current_global_size
            or state.local_crop_size != self._current_local_size
        ):
            self.set_resolution(state.global_crop_size, state.local_crop_size)

        self._epoch = state.epoch
        self._step  = state.step

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
            log.info("Serveur Prometheus démarré sur le port %d (rank 0)", port)
        except Exception as exc:
            log.warning("Impossible de démarrer le serveur Prometheus : %s", exc)

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
