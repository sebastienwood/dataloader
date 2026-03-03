"""
dino_loader.loader
==================
DINODataLoader: the single public entry point for training code.

Changes vs previous version
----------------------------
[LD-1]  StatefulDataLoader interface (torchdata ≥ 0.8 / PyTorch 2.3+).
        When LoaderConfig.stateful_dataloader=True, DINODataLoader exposes
        state_dict() and load_state_dict() that match the protocol expected
        by PyTorch Lightning, torchtitan, and other frameworks.  Internally
        these delegate to DataLoaderCheckpointer, so the on-disk format is
        unchanged (JSON, backward-compatible).

[LD-2]  set_resolution(global_size, local_size) — zero-downtime resolution change.
        Writes to ResolutionSource (thread-safe), which is consumed by the DALI
        pipeline's ExternalSource.  No DALI rebuild; takes effect on the next
        batch boundary.  CheckpointState persists the current resolution.

[LD-3]  Resolution schedule auto-apply.
        If DINOAugConfig.resolution_schedule is set, set_epoch() automatically
        calls set_resolution() for the new epoch's dictated size.  Removes
        boilerplate from training scripts.

[LD-4]  Batch.metadata — per-sample metadata list.
        MixingSource.pop_last_metadata() is called after each DALI batch and
        stored in Batch.metadata (List[Optional[Dict]]).  None for samples
        from shards without .json sidecars, or when metadata_key=None.

[LD-5]  MaskingGenerator integration (DinoV3 / iBOT).
        DINODataLoader accepts an optional mask_generator.  When provided,
        collate() calls it to produce token masks and packs them into Batch,
        matching the collate_data_and_cast pattern from dinov3/data/collate.py.
        Mask generation runs on CPU (rank 0) immediately after DALI output
        before H2D transfer, adding negligible latency.

[LD-6]  ResolutionSource wired into build_pipeline call.
        The pipeline now receives a ResolutionSource instead of static ints.

[LD-7]  Backend abstraction.
        DINODataLoader now accepts an optional ``backend`` parameter of type
        ``BackendProtocol``.  When not provided, ``get_backend("auto")`` is
        used: DALI on GPU, CPU backend as fallback.

        This enables full end-to-end testing on any machine:
        ::
            from dino_loader.backends import get_backend
            loader = DINODataLoader(..., backend=get_backend("cpu"))

        All five pipeline stages (shard cache, augmentation pipeline,
        pipeline iterator, H2D stream, FP8 formatter) are constructed via
        the backend, so the rest of loader.py is backend-agnostic.

        The ``HAS_DALI`` guard and direct ``DALIGenericIterator`` import are
        removed from this module; the backend encapsulates those details.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist

from dino_loader.backends         import get_backend, BackendName
from dino_loader.backends.protocol import BackendProtocol
from dino_loader.checkpoint       import DataLoaderCheckpointer
from dino_loader.config           import CheckpointState, DatasetSpec, DINOAugConfig, LoaderConfig
from dino_loader.distributed      import ClusterTopology, detect_topology
from dino_loader.memory           import Batch
from dino_loader.mixing_source    import MixingSource, ResolutionSource
from dino_loader.monitor.metrics  import get_registry, init_registry

log = logging.getLogger(__name__)


class DINODataLoader:
    """
    HPC-grade DINOv3 data loader for B200 / GB200 NVL72 clusters.

    Parameters
    ----------
    specs            : List of DatasetSpec (name, shards, weight, quality config).
    batch_size       : Per-GPU batch size.
    aug_cfg          : DINOAugConfig — augmentation and resolution hyper-parameters.
    config           : LoaderConfig — all infrastructure knobs.
    device_id        : Local GPU index.
    rank, world_size, local_rank, local_world_size
                     : Distributed identity.  Inferred from dist when initialised.
    resume           : If True, attempt to load the latest dataloader checkpoint.
    steps_per_epoch  : Optional; enables ``len(loader)``.
    mask_generator   : Optional MaskingGenerator from dinov3.data.masking.
                       When provided, token masks are generated and added to Batch.
    backend          : BackendProtocol instance, or a backend name string
                       ("auto" | "dali" | "cpu").  Defaults to "auto"
                       (DALI when available, CPU otherwise).  [LD-7]

    StatefulDataLoader interface  [LD-1]
    ------------------------------------
    When LoaderConfig.stateful_dataloader=True::

        sd = loader.state_dict()          # → dict, JSON-serialisable
        loader.load_state_dict(sd)        # resume from dict

    Resolution scheduling  [LD-2, LD-3]
    ------------------------------------
    Manual::

        loader.set_resolution(448, 192)   # takes effect next batch, no rebuild

    Automatic (via DINOAugConfig.resolution_schedule)::

        aug_cfg = DINOAugConfig(
            resolution_schedule=[(0, 224), (10, 448), (30, 518)],
            max_global_crop_size=518,
        )
        # set_epoch(epoch) auto-applies the scheduled resolution.

    CPU testing  [LD-7]
    -------------------
    ::

        from dino_loader.backends import get_backend
        loader = DINODataLoader(
            specs      = specs,
            batch_size = 4,
            backend    = get_backend("cpu"),
        )
        for batch in loader:
            assert batch.global_crops[0].shape == (4, 3, 224, 224)
    """

    def __init__(
        self,
        specs:            List[DatasetSpec],
        batch_size:       int,
        aug_cfg:          Optional[DINOAugConfig] = None,
        config:           Optional[LoaderConfig]  = None,
        device_id:        int  = 0,
        rank:             Optional[int] = None,
        world_size:       Optional[int] = None,
        local_rank:       Optional[int] = None,
        local_world_size: Optional[int] = None,
        resume:           bool = False,
        steps_per_epoch:  Optional[int] = None,
        mask_generator:   Optional[Any] = None,
        backend:          Optional[Any] = None,   # BackendProtocol | BackendName | None  [LD-7]
    ) -> None:
        # ── Backend selection [LD-7] ──────────────────────────────────────────
        if backend is None:
            self._backend: BackendProtocol = get_backend("auto")
        elif isinstance(backend, str):
            self._backend = get_backend(backend)  # type: ignore[arg-type]
        else:
            self._backend = backend

        log.info("DINODataLoader: using backend '%s'", self._backend.name)

        self._aug_cfg         = aug_cfg or DINOAugConfig()
        self._cfg             = config  or LoaderConfig()
        self._batch_size      = batch_size
        self._steps_per_epoch = steps_per_epoch
        self._mask_generator  = mask_generator
        self._active_iter     = False
        self._epoch           = 0
        self._step            = 0

        # ── Distributed identity ─────────────────────────────────────────────
        if dist.is_available() and dist.is_initialized():
            rank             = rank             or dist.get_rank()
            world_size       = world_size       or dist.get_world_size()
            local_rank       = local_rank       or int(os.environ.get("LOCAL_RANK", device_id))
            local_world_size = local_world_size or int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        else:
            rank             = rank             if rank             is not None else 0
            world_size       = world_size       if world_size       is not None else 1
            local_rank       = local_rank       if local_rank       is not None else 0
            local_world_size = local_world_size if local_world_size is not None else 1

        self._rank             = rank
        self._world_size       = world_size
        self._local_rank       = local_rank
        self._local_world_size = local_world_size
        self._device_id        = device_id
        self._is_node_master   = (local_rank == 0)

        # ── Metrics registry ─────────────────────────────────────────────────
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        init_registry(job_id=job_id, local_rank=local_rank)

        # ── Topology detection ────────────────────────────────────────────────
        # For the CPU backend, topology is a StubClusterTopology.
        # For the DALI backend, detect_topology probes real hardware.
        if self._backend.name == "cpu":
            from dino_loader.backends.cpu import StubClusterTopology
            self._topo = StubClusterTopology()
        else:
            self._topo: ClusterTopology = detect_topology(
                force=self._cfg.force_topology,
            )

        # ── Stage 1: Shard cache ──────────────────────────────────────────────
        self._shard_cache = self._backend.build_shard_cache(
            job_id          = job_id,
            node_master     = self._is_node_master,
            max_gb          = self._cfg.node_shm_gb,
            prefetch_window = self._cfg.shard_prefetch_window,
            timeout_s       = self._cfg.shard_timeout_s,
            warn_threshold  = self._cfg.shm_warn_threshold,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # ── Stage 2: Mixing source ────────────────────────────────────────────
        self._source = MixingSource(
            specs                = specs,
            batch_size           = batch_size,
            cache                = self._shard_cache,
            rank                 = rank,
            world_size           = world_size,
            prefetch_ahead       = self._cfg.shard_prefetch_window,
            num_workers          = self._cfg.shard_extraction_workers,
            seed                 = self._cfg.seed,
            device_id            = device_id,
            cpu_affinity_enabled = self._cfg.cpu_affinity_enabled,
            shuffle_buffer_size  = self._cfg.shuffle_buffer_size,
        )

        # ── [LD-2] Resolution source ──────────────────────────────────────────
        self._resolution_src = ResolutionSource(
            global_size = self._aug_cfg.global_crop_size,
            local_size  = self._aug_cfg.local_crop_size,
        )
        self._current_global_size = self._aug_cfg.global_crop_size
        self._current_local_size  = self._aug_cfg.local_crop_size

        # ── Stage 3: Augmentation pipeline ───────────────────────────────────
        self._pipe = self._backend.build_pipeline(
            source          = self._source,
            aug_cfg         = self._aug_cfg,
            batch_size      = batch_size,
            num_threads     = self._cfg.dali_num_threads,
            device_id       = device_id,
            resolution_src  = self._resolution_src,
            hw_decoder_load = self._cfg.hw_decoder_load,
            cpu_queue       = self._cfg.dali_cpu_queue,
            gpu_queue       = self._cfg.dali_gpu_queue,
            seed            = self._cfg.seed + rank,
        )

        _view_names = [f"view_{i}" for i in range(self._aug_cfg.n_views)]
        self._dali_iter = self._backend.build_pipeline_iterator(
            pipeline   = self._pipe,
            output_map = _view_names,
            batch_size = batch_size,
        )

        # ── Stage 4 & 5: H2D + FP8 ───────────────────────────────────────────
        device    = torch.device(f"cuda:{device_id}") if self._backend.supports_gpu else torch.device("cpu")
        self._h2d = self._backend.build_h2d_stream(device=device, topo=self._topo)
        self._fp8 = self._backend.build_fp8_formatter() if self._cfg.use_fp8_output else None

        # ── Checkpointing ─────────────────────────────────────────────────────
        self._ckpt = DataLoaderCheckpointer(
            ckpt_dir     = self._cfg.checkpoint_dir,
            every_n_steps= self._cfg.checkpoint_every_steps,
            rank         = rank,
        )

        if resume:
            self._restore()

        log.info(
            "DINODataLoader ready: backend=%s rank=%d/%d, batch=%d, "
            "resolution=%dx%d (max %dx%d), shuffle_buf=%d",
            self._backend.name,
            rank, world_size, batch_size,
            self._current_global_size, self._current_local_size,
            self._aug_cfg.max_global_crop_size, self._aug_cfg.max_local_crop_size,
            self._cfg.shuffle_buffer_size,
        )

    # ── Iteration protocol ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Batch]:
        if self._active_iter:
            raise RuntimeError(
                "DINODataLoader: __iter__ called while already iterating. "
                "Call set_epoch() before starting a new epoch loop."
            )
        self._active_iter = True
        try:
            yield from self._iter_batches()
        finally:
            self._active_iter = False

    def _iter_batches(self) -> Iterator[Batch]:
        metrics = get_registry()
        for dali_out in self._dali_iter:
            t0 = time.perf_counter()

            views = [dali_out[0][f"view_{i}"] for i in range(self._aug_cfg.n_views)]
            n_global = self._aug_cfg.n_global_crops
            global_views = views[:n_global]
            local_views  = views[n_global:]

            # [LD-4] Retrieve per-sample metadata from MixingSource
            metadata = self._source.pop_last_metadata()

            # [LD-5] Token mask generation (DinoV3 iBOT pattern)
            masks = None
            if self._mask_generator is not None:
                n_tokens = (self._current_global_size // 14) ** 2  # ViT patch=14
                masks    = self._mask_generator(n_tokens)

            # H2D transfer
            with self._h2d.transfer({"global": global_views, "local": local_views}) as gpu:
                g_gpu = gpu["global"]
                l_gpu = gpu["local"]

            # Optional FP8 quantisation
            if self._fp8 is not None:
                g_gpu = [self._fp8.quantise(t) for t in g_gpu]
                l_gpu = [self._fp8.quantise(t) for t in l_gpu]

            batch = Batch(
                global_crops = g_gpu,
                local_crops  = l_gpu,
                metadata     = metadata,
                masks        = masks,
            )

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if metrics:
                metrics.inc("loader_batches_yielded", 1)
                metrics.inc("pipeline_yield_time_ms", elapsed_ms)
                metrics.set("heartbeat_ts", int(time.time()))

            yield batch

    def __len__(self) -> int:
        if self._steps_per_epoch is None:
            raise TypeError(
                "len(loader) requires steps_per_epoch to be set at construction. "
                "Pass steps_per_epoch=total_images // (batch_size * world_size)."
            )
        return self._steps_per_epoch

    def __del__(self):
        self._active_iter = False

    # ── Epoch / resolution control ────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """
        Re-shuffle shards for the new epoch and apply the resolution schedule.

        Must be called at the start of each epoch.
        """
        self._active_iter = False
        self._epoch       = epoch
        self._source.set_epoch(epoch)
        self._dali_iter.reset()

        # [LD-3] Auto-apply resolution schedule
        if self._aug_cfg.resolution_schedule:
            new_global = self._aug_cfg.crop_size_at_epoch(epoch)
            ratio     = self._aug_cfg.local_crop_size / self._aug_cfg.global_crop_size
            new_local = max(int(new_global * ratio), 32)
            if new_global != self._current_global_size:
                self.set_resolution(new_global, new_local)
                log.info(
                    "Resolution schedule: epoch %d → global=%d local=%d",
                    epoch, new_global, new_local,
                )

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """
        Change the crop resolution without rebuilding the augmentation pipeline.  [LD-2]

        Thread-safe.  Takes effect on the next batch boundary.

        Parameters
        ----------
        global_size : Target pixel size for global crops (e.g. 224, 448, 518).
        local_size  : Target pixel size for local crops (e.g. 96, 192).
        """
        if global_size > self._aug_cfg.max_global_crop_size:
            raise ValueError(
                f"global_size={global_size} exceeds max_global_crop_size="
                f"{self._aug_cfg.max_global_crop_size}.  Rebuild the pipeline "
                f"or increase max_global_crop_size."
            )
        self._resolution_src.set(global_size, local_size)
        self._current_global_size = global_size
        self._current_local_size  = local_size
        log.info("set_resolution: global=%d local=%d", global_size, local_size)

    # ── Dataset mixing control ────────────────────────────────────────────────

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update dataset mixing weights (thread-safe, next-batch effective)."""
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Update a single dataset's mixing weight by name."""
        self._source.set_weight_by_name(name, weight)

    @property
    def current_weights(self) -> List[float]:
        return self._source.current_weights

    # ── Checkpointing (manual API) ────────────────────────────────────────────

    def checkpoint(self, step: int) -> None:
        """Save dataloader state.  Rank 0 only; every N steps."""
        state = CheckpointState(
            step             = step,
            epoch            = self._epoch,
            dataset_names    = self._source.dataset_names,
            mixing_weights   = self._source.current_weights,
            global_crop_size = self._current_global_size,
            local_crop_size  = self._current_local_size,
        )
        self._ckpt.save(state)

        util = self._shard_cache.utilisation
        if util >= self._cfg.shm_warn_threshold and self._rank == 0:
            log.warning(
                "checkpoint step=%d: /dev/shm utilisation %.1f%% ≥ threshold %.0f%%",
                step, util * 100, self._cfg.shm_warn_threshold * 100,
            )

    # ── StatefulDataLoader interface  [LD-1] ──────────────────────────────────

    def state_dict(self) -> Dict:
        """
        Return a JSON-serialisable state dict.

        Compatible with the torchdata StatefulDataLoader protocol so that
        Lightning / torchtitan can call this automatically.
        """
        if not self._cfg.stateful_dataloader:
            raise RuntimeError(
                "state_dict() requires LoaderConfig.stateful_dataloader=True."
            )
        return {
            "step":             self._step,
            "epoch":            self._epoch,
            "dataset_names":    self._source.dataset_names,
            "mixing_weights":   self._source.current_weights,
            "global_crop_size": self._current_global_size,
            "local_crop_size":  self._current_local_size,
        }

    def load_state_dict(self, state: Dict) -> None:
        """
        Restore state from a dict previously returned by state_dict().

        Compatible with the torchdata StatefulDataLoader protocol.
        """
        if not self._cfg.stateful_dataloader:
            raise RuntimeError(
                "load_state_dict() requires LoaderConfig.stateful_dataloader=True."
            )
        cs = CheckpointState(**state)
        self._apply_checkpoint(cs)

    def _restore(self) -> None:
        """Load the latest on-disk checkpoint (called from __init__ when resume=True)."""
        state = self._ckpt.load()
        if state is None:
            log.info("No dataloader checkpoint found — starting from scratch.")
            return
        self._apply_checkpoint(state)

    def _apply_checkpoint(self, state: CheckpointState) -> None:
        self._step  = state.step
        self._epoch = state.epoch
        self.set_epoch(state.epoch)

        if (state.global_crop_size != self._current_global_size or
                state.local_crop_size != self._current_local_size):
            self.set_resolution(state.global_crop_size, state.local_crop_size)

        if (state.mixing_weights and
                state.dataset_names == self._source.dataset_names):
            self._source.set_weights(state.mixing_weights)
        elif state.dataset_names != self._source.dataset_names:
            log.warning(
                "Checkpoint dataset names %s do not match current %s — "
                "mixing weights NOT restored.",
                state.dataset_names, self._source.dataset_names,
            )
        log.info(
            "Resumed dataloader: epoch=%d step=%d resolution=%dx%d",
            self._epoch, self._step,
            self._current_global_size, self._current_local_size,
        )

    # ── Monitoring ────────────────────────────────────────────────────────────

    @property
    def shard_cache_utilisation(self) -> float:
        """Current cache utilisation fraction (0–1)."""
        return self._shard_cache.utilisation

    @property
    def backend(self) -> BackendProtocol:
        """The active backend instance."""
        return self._backend
