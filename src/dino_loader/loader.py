"""dino_loader.loader
==================
DINODataLoader : point d'entrée public unique pour le code d'entraînement.

Architecture (Phase 1/3 + wrap_loader implicite)
-------------------------------------------------
DINODataLoader délègue entièrement à ``NodePipeline`` pour l'itération.
Toute la logique de boucle, métriques et stall detection vit dans
``_DALINode`` (``pipeline_graph.py``).

Flux de construction ::

    DINODataLoader.__init__
        → ShardReaderNode           (stage 1-2 : I/O + mixing)
        → _ReaderAdapter            (bridge ShardReaderNode → callable DALI)
        → BackendProtocol.build_*   (stage 3-5 : DALI/CPU + H2D + FP8)
        → _DALINode                 (BaseNode : pilote DALI, assemble Batch)
        → NodePipeline              (self._pipeline = wrap_loader(self))

Flux d'itération ::

    for batch in loader:          # délègue à self._pipeline
        ...

Corrections intégrées
---------------------
[FIX-ENV]          DistribEnv conservé tel quel ; propriétés par délégation.
[FIX-ACTIVE-ITER]  Guard contre le double-iter via _active_iter + lock.
[FIX-RESET-ITER]   set_epoch() appelle _dali_node.reset_iter() (thread-safe).
[FIX-META-FIFO]    _ReaderAdapter utilise queue.Queue pour l'alignement FIFO.
[FIX-RESTORE-LOCAL] _restore() appelle set_resolution() dès que l'une des
                    deux dimensions diffère (global OU local).
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
from dino_loader.monitor.metrics import init_registry
from dino_loader.pipeline_graph import NodePipeline, _DALINode, wrap_loader
from dino_loader.shard_reader import ShardReaderNode
from dino_loader.sources.protocol import SourceProtocol
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
        for epoch in range(100):
            pipeline.set_epoch(epoch)
            for batch in pipeline:
                train_step(batch)

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
        backend:          "auto" | "dali" | "cpu" | instance BackendProtocol.

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
        """Construit un DINODataLoader."""
        # Rétrocompat : enveloppement du legacy aug_cfg.
        if aug_spec is None:
            effective_aug_cfg: DINOAugConfig = aug_cfg if aug_cfg is not None else DINOAugConfig()
            self._aug_spec: AugmentationSpec = DinoV2AugSpec(aug_cfg=effective_aug_cfg)
        else:
            if aug_cfg is not None:
                log.warning(
                    "DINODataLoader: aug_spec et aug_cfg fournis simultanément ; "
                    "aug_cfg est ignoré (aug_spec a la priorité).",
                )
            self._aug_spec = aug_spec

        self._aug_cfg: DINOAugConfig = (
            self._aug_spec.aug_cfg  # type: ignore[attr-defined]
            if isinstance(self._aug_spec, DinoV2AugSpec)
            else DINOAugConfig()
        )

        self._cfg             = config or LoaderConfig(stateful_dataloader=False, checkpoint_dir="")
        self._mask_generator  = mask_generator
        self._steps_per_epoch = steps_per_epoch

        self._step:  int = 0
        self._epoch: int = 0
        self._last_ckpt_state: CheckpointState | None = None

        self._epoch_lock       = threading.Lock()
        self._active_iter      = False
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
                    "DINODataLoader avec le backend DALI requiert que "
                    "dino_env.init() soit appelé avant la construction du loader."
                )
                raise RuntimeError(msg) from None

        # [FIX-ENV] Conserver l'objet DistribEnv complet.
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

        # Suivi de la résolution courante
        self._current_global_size = self._initial_global_size()
        self._current_local_size  = self._initial_local_size()
        self._resolution_src      = ResolutionSource(
            self._current_global_size,
            self._current_local_size,
        )

        # Stage 1 : cache de shards
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

        # Stage 1-2 : ShardReaderNode
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

        # NodePipeline interne
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
        """Instance du backend actif."""
        return self._backend

    @property
    def aug_spec(self) -> AugmentationSpec:
        """Spec d'augmentation active."""
        return self._aug_spec

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Résolution de crop courante sous la forme (global_size, local_size)."""
        return (self._current_global_size, self._current_local_size)

    @property
    def current_weights(self) -> list[float]:
        """Poids de mixage normalisés courants."""
        return self._reader.current_weights

    # ── Accesseurs vers DistribEnv ────────────────────────────────────────────

    @property
    def rank(self) -> int:
        """Rang global de ce processus."""
        return self._env.rank

    @property
    def world_size(self) -> int:
        """Nombre total de processus."""
        return self._env.world_size

    @property
    def local_rank(self) -> int:
        """Rang local au sein du nœud."""
        return self._env.local_rank

    @property
    def local_world_size(self) -> int:
        """Nombre de processus sur ce nœud."""
        return self._env.local_world_size

    # ── API de composition ────────────────────────────────────────────────────

    def as_pipeline(self) -> NodePipeline:
        """Retourne le NodePipeline interne pour la composition."""
        return self._pipeline

    def map(self, fn: Callable[[Batch], Batch], *, label: str = "<map>") -> NodePipeline:
        """Raccourci : ``loader.map(fn)`` ≡ ``loader.as_pipeline().map(fn)``."""
        return self._pipeline.map(fn, label=label)

    def select(
        self,
        predicate: Callable[[Batch], bool],
        *,
        label: str = "<filter>",
    ) -> NodePipeline:
        """Raccourci : ``loader.select(pred)`` ≡ ``loader.as_pipeline().select(pred)``."""
        return self._pipeline.select(predicate, label=label)

    def with_epoch(self, n_steps: int) -> NodePipeline:
        """Raccourci : ``loader.with_epoch(n)`` ≡ ``loader.as_pipeline().with_epoch(n)``."""
        return self._pipeline.with_epoch(n_steps)

    # ── Helpers spec d'augmentation ───────────────────────────────────────────

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

    # ── Contrôle époque / poids / résolution ──────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Prépare le loader pour une nouvelle époque.

        Args:
            epoch: Numéro de la nouvelle époque (0-indexé).

        """
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

        Args:
            global_size: Nouvelle taille du crop global en pixels.
            local_size:  Nouvelle taille du crop local en pixels.

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
        """Met à jour les poids de mixage (re-normalisés automatiquement)."""
        self._reader.set_weights(list(weights))

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids d'un dataset par son nom."""
        self._reader.set_weight_by_name(name, weight)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def checkpoint(self, step: int) -> None:
        """Sauvegarde un checkpoint (rank 0 uniquement, tous les N steps)."""
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
        """Retourne l'état sous forme de dict (depuis la mémoire, sans I/O disque).

        Raises:
            RuntimeError: Si ``stateful_dataloader=False`` dans ``LoaderConfig``.

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
        """Restaure depuis un state dict produit par state_dict()."""
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
        """Itère sur les batches d'entraînement via le NodePipeline interne.

        Raises:
            RuntimeError: Si une itération est déjà active sur ce loader.

        """
        with self._active_iter_lock:
            if self._active_iter:
                msg = (
                    "DINODataLoader est déjà en cours d'itération. "
                    "Impossible de démarrer une seconde itération simultanée. "
                    "Appeler set_epoch() pour démarrer une nouvelle époque, ou "
                    "utiliser loader.as_pipeline() pour une composition sûre."
                )
                raise RuntimeError(msg)
            self._active_iter = True

        try:
            yield from self._pipeline
        finally:
            with self._active_iter_lock:
                self._active_iter = False

    def __len__(self) -> int:
        """Retourne steps_per_epoch si défini ; lève TypeError sinon."""
        if self._steps_per_epoch is None:
            msg = "len(loader) requiert que steps_per_epoch soit défini à la construction."
            raise TypeError(msg)
        return self._steps_per_epoch

    # ── Assemblage du Batch ───────────────────────────────────────────────────

    def _build_batch(
        self,
        views:    list[Any],
        metadata: list[dict | None],
    ) -> Batch:
        """Assemble un Batch depuis les vues du pipeline.

        Args:
            views:    Liste plate de tenseurs issus du pipeline d'augmentation.
            metadata: Dicts JSON par sample.

        Returns:
            ``Batch`` assemblé sur le device cible.

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
        """Sépare la liste plate de vues en (global_crops, local_crops)."""
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

    # ── Helpers internes ──────────────────────────────────────────────────────

    def _validate_shard_coverage(self, specs: list[DatasetSpec]) -> None:
        """Avertit si un dataset a moins de shards que de rangs."""
        for spec in specs:
            n_shards = len(spec.shards)
            if n_shards < self._env.world_size:
                log.warning(
                    "DatasetSpec '%s' : seulement %d shard(s) pour %d rangs. "
                    "Envisager shard_sampling='resampled' pour les petits datasets.",
                    spec.name, n_shards, self._env.world_size,
                )

    def _restore(self) -> None:
        """Charge le dernier checkpoint et applique son état.

        [FIX-RESTORE-LOCAL] set_resolution() est appelé si l'une OU l'autre
        dimension (global ou local) diffère de la valeur courante.
        """
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
        """Démarre un serveur HTTP Prometheus sur *port* (rank 0 uniquement)."""
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
        """Infère le rang global depuis les variables d'environnement."""
        for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _infer_world_size() -> int:
        """Infère le world size depuis les variables d'environnement."""
        for var in ("WORLD_SIZE", "SLURM_NTASKS"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _infer_local_world_size() -> int:
        """Infère le local world size depuis les variables d'environnement."""
        for var in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"):
            v = os.environ.get(var)
            if v is not None:
                return int(v)
        return 1


# ---------------------------------------------------------------------------
# _ReaderAdapter — bridge ShardReaderNode → callable DALI/CPU ExternalSource
# ---------------------------------------------------------------------------


class _ReaderAdapter:
    """Adaptateur exposant ``ShardReaderNode`` comme callable source DALI/CPU.

    Respecte ``SourceProtocol`` pour les callbacks NormSource.

    [FIX-META-FIFO] ``_last_metadata`` est une ``queue.Queue`` (FIFO) plutôt
    qu'une variable scalaire, garantissant l'alignement strict entre les batches
    DALI et leurs métadonnées même quand DALI appelle ``__call__()`` de façon
    préemptive depuis son thread de prefetch.

    Attributs de convention
    -----------------------
    ``_batch_size`` et ``_resolution_src`` sont lus par les backends via
    ``getattr`` pour inférer batch_size et la source de résolution.
    """

    # Taille maximale de la queue de métadonnées.
    # DALI ne précharge pas plus de cpu_queue + gpu_queue batches en avance.
    _META_QUEUE_MAXSIZE: int = 32

    def __init__(
        self,
        reader:         ShardReaderNode,
        resolution_src: ResolutionSource,
        batch_size:     int,
    ) -> None:
        """Initialise l'adaptateur."""
        self._reader         = reader
        self._resolution_src = resolution_src
        self._batch_size     = batch_size
        self._meta_queue: queue.Queue[list[dict | None]] = queue.Queue(
            maxsize=self._META_QUEUE_MAXSIZE,
        )

    def __call__(self) -> list:
        """Retourne un batch de tableaux JPEG (appelé par DALI à chaque step)."""
        jpegs, metadata = self._reader.next()
        try:
            self._meta_queue.put_nowait(metadata)
        except queue.Full:
            log.warning(
                "_ReaderAdapter: queue de métadonnées pleine (%d slots). "
                "La profondeur de prefetch DALI dépasse _META_QUEUE_MAXSIZE=%d. "
                "Augmenter _META_QUEUE_MAXSIZE ou réduire les profondeurs de queue DALI.",
                self._META_QUEUE_MAXSIZE, self._META_QUEUE_MAXSIZE,
            )
            try:
                self._meta_queue.get_nowait()
            except queue.Empty:
                pass
            self._meta_queue.put_nowait(metadata)
        return jpegs

    def pop_last_metadata(self) -> list[dict | None]:
        """Retourne les métadonnées du batch le plus ancien (FIFO)."""
        try:
            return self._meta_queue.get_nowait()
        except queue.Empty:
            log.debug("_ReaderAdapter.pop_last_metadata: queue vide, retour []")
            return []

    def register_dataset_index_callback(self, cb: Any) -> None:
        """Propage les callbacks NormSource vers la MixingSource interne."""
        source = self._reader._source  # type: ignore[attr-defined]
        if source is not None and hasattr(source, "register_dataset_index_callback"):
            source.register_dataset_index_callback(cb)
        else:
            log.warning(
                "_ReaderAdapter: register_dataset_index_callback appelé avant "
                "ShardReaderNode.reset() — callback non enregistré.",
            )