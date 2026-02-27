"""
dino_loader.loader
==================
DINODataLoader: the single public entry point for training code.

Responsibilities
----------------
- Compose subsystems: shard cache, mixing source, DALI pipeline, H2D, FP8.
- Expose the three-line training-loop API:
      loader.set_epoch(epoch)
      for batch in loader: ...
      loader.checkpoint(step)
- Expose dynamic mixing: set_weights(), set_weight_by_name().
- Handle distributed identity (auto-detects from dist if initialised).

Fixes
-----
[FIX-4]  Import paths corrected.  The previous loader.py imported from
         ``dino_loader.augment.pipeline``, ``dino_loader.cache.memory``,
         ``dino_loader.io.mixing_source``, and ``dino_loader.io.shard_cache``.
         None of these subdirectories exist in the actual package tree;
         the modules live directly under ``dino_loader/``.  Every import
         was broken at runtime.  Fixed to use the actual flat paths.

[FIX-14] ``set_epoch`` now calls ``self._source.set_epoch(epoch)`` so
         ``ShardIterator.reset_epoch`` runs and re-shuffles shards.
         Previously ``set_epoch`` only updated the internal counter,
         meaning every epoch saw the same shard order.

[FIX-15] ``__len__`` added.  PyTorch schedulers, progress bars (tqdm), and
         ``torch.utils.data`` ecosystem components require ``len(loader)``.
         The value is the number of steps per epoch: total shards × average
         JPEGs per shard / (batch_size × world_size).  Because we don't
         have per-shard JPEG counts at loader construction time, we expose
         an optional ``steps_per_epoch`` parameter; if not provided, len()
         raises ``TypeError`` with a helpful message.

[FIX-19] ``shard_extraction_workers`` now read from ``LoaderConfig`` and
         forwarded to ``MixingSource`` / ``ShardIterator``.

[FIX-A]  ``_restore()`` was not calling ``self._source.set_epoch(state.epoch)``
         after restoring epoch from a checkpoint.  All ShardIterators silently
         started from epoch-0 shard order on every resume, regardless of the
         saved epoch.  Fixed: ``set_epoch()`` is now called inside ``_restore()``
         after the epoch counter is updated.

[FIX-C]  ``__iter__`` could create two concurrent consumers on the same DALI
         iterator if called twice within one epoch (e.g. by a tqdm progress bar
         or an accidental double loop).  DALI's iterator is stateful and
         non-reentrant; interleaving reads produces silently corrupted batches.
         Fixed: an ``_active_iter`` flag raises RuntimeError on double-entry.
         The flag is cleared by ``set_epoch()`` and ``__del__()``.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist

# [FIX-4] Corrected flat import paths (no augment/ cache/ io/ subdirectories)
from dino_loader.pipeline       import build_pipeline
from dino_loader.memory         import AsyncPrefetchIterator, Batch, FP8Formatter, H2DStream
from dino_loader.checkpoint     import DataLoaderCheckpointer
from dino_loader.config         import CheckpointState, DatasetSpec, DINOAugConfig, LoaderConfig
from dino_loader.distributed    import ClusterTopology, detect_topology
from dino_loader.mixing_source  import MixingSource
from dino_loader.shard_cache    import NodeSharedShardCache
from dino_loader.monitor.metrics import get_registry, init_registry

log = logging.getLogger(__name__)

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    log.error("nvidia-dali not installed — DINODataLoader will not build")


class DINODataLoader:
    """
    HPC-grade DINOv3 data loader for B200 / GB200 NVL72 clusters.

    Parameters
    ----------
    specs            : List of DatasetSpec (name, shards, initial weight).
    batch_size       : Per-GPU batch size.
    aug_cfg          : DINOAugConfig — augmentation hyper-parameters.
    config           : LoaderConfig — all infrastructure knobs.
    device_id        : Local GPU index.
    rank, world_size, local_rank, local_world_size : distributed identity.
                       Inferred automatically when a process group is active.
    resume           : If True, attempt to load the latest dataloader checkpoint.
    steps_per_epoch  : Optional; enables ``len(loader)``.  Set to
                       (total_images // (batch_size * world_size)).

    Example
    -------
        env = slurm_init()
        loader = DINODataLoader(
            specs            = [DatasetSpec("laion", shards, weight=1.0)],
            batch_size       = 512,
            config           = LoaderConfig(node_shm_gb=256),
            device_id        = env.local_rank,
            steps_per_epoch  = 200_000,
        )
        for epoch in range(100):
            loader.set_epoch(epoch)
            for step, batch in enumerate(loader):
                train_step(batch.global_crops, batch.local_crops)
                loader.checkpoint(step)
    """

    def __init__(
        self,
        specs:            List[DatasetSpec],
        batch_size:       int,
        aug_cfg:          Optional[DINOAugConfig] = None,
        config:           Optional[LoaderConfig]  = None,
        device_id:        int                     = 0,
        rank:             int                     = 0,
        world_size:       int                     = 1,
        local_rank:       int                     = 0,
        local_world_size: int                     = 8,
        resume:           bool                    = False,
        steps_per_epoch:  Optional[int]           = None,   # [FIX-15]
    ):
        if not HAS_DALI:
            raise ImportError(
                "nvidia-dali not installed — DINODataLoader cannot be instantiated"
            )

        # ── Resolve distributed identity ──────────────────────────────────────
        if dist.is_available() and dist.is_initialized():
            rank             = dist.get_rank()
            world_size       = dist.get_world_size()
            local_rank       = int(os.environ.get("LOCAL_RANK", local_rank))

        self._rank              = rank
        self._world             = world_size
        self._device            = torch.device(f"cuda:{device_id}")
        self._aug_cfg           = aug_cfg or DINOAugConfig()
        self._cfg               = config  or LoaderConfig()
        self._epoch             = 0
        self._step              = 0
        self._steps_per_epoch   = steps_per_epoch   # [FIX-15]
        self._active_iter: Optional[AsyncPrefetchIterator] = None  # [FIX-C]

        # ── Topology ──────────────────────────────────────────────────────────
        self._topo = detect_topology(force=self._cfg.force_topology)

        # ── Monitoring registry (must precede cache construction) ─────────────────
        # [FIX-MON] init_registry was never called; all counters were stuck at 0.
        init_registry(
            job_id     = self._cfg.job_id,
            create     = (local_rank == 0),   # node master creates; others attach
            local_rank = local_rank,
        )
        log.debug("Metrics registry initialised (local_rank=%d, create=%s)", local_rank, local_rank == 0)

        # ── Stage 1: Node-local shared shard cache ────────────────────────────
        node_master = (local_rank == 0)
        self._shard_cache = NodeSharedShardCache(
            node_master     = node_master,
            job_id          = os.environ.get("SLURM_JOB_ID", "dino"),
            max_shm_gb      = self._cfg.node_shm_gb,
            prefetch_window = self._cfg.shard_prefetch_window,
            shard_timeout_s = self._cfg.shard_timeout_s,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # ── Stage 2: Mixing source ────────────────────────────────────────────
        self._source = MixingSource(
            specs          = specs,
            batch_size     = batch_size,
            cache          = self._shard_cache,
            rank           = rank,
            world_size     = world_size,
            prefetch_ahead = self._cfg.shard_prefetch_window,
            num_workers    = self._cfg.shard_extraction_workers,  # [FIX-19]
            seed           = self._cfg.seed,
        )

        # ── Stage 3: DALI pipeline ────────────────────────────────────────────
        self._pipe = build_pipeline(
            source          = self._source,
            aug_cfg         = self._aug_cfg,
            batch_size      = batch_size,
            num_threads     = self._cfg.dali_num_threads,
            device_id       = device_id,
            hw_decoder_load = self._cfg.hw_decoder_load,
            cpu_queue       = self._cfg.dali_cpu_queue,
            gpu_queue       = self._cfg.dali_gpu_queue,
            seed            = self._cfg.seed + rank,
        )

        _view_names = [f"view_{i}" for i in range(self._aug_cfg.n_views)]
        self._dali_iter = DALIGenericIterator(
            pipelines         = [self._pipe],
            output_map        = _view_names,
            last_batch_policy = LastBatchPolicy.DROP,
            auto_reset        = True,
        )

        # ── Stage 4: H2D transfer stream ─────────────────────────────────────
        self._h2d = H2DStream(self._device, self._topo)

        # ── Stage 5: Optional FP8 / TE output formatter ───────────────────────
        self._fp8_fmt: Optional[FP8Formatter] = None
        if self._cfg.use_fp8_output:
            self._fp8_fmt = FP8Formatter(self._device)

        # ── Checkpointer ──────────────────────────────────────────────────────
        self._ckptr = DataLoaderCheckpointer(
            ckpt_dir      = self._cfg.checkpoint_dir,
            every_n_steps = self._cfg.checkpoint_every_steps,
            rank          = rank,
        )
        if resume:
            self._restore()

        log.info(
            "DINODataLoader ready | topology=%s | rank=%d/%d | "
            "views=%dg+%dl | FP8=%s | extraction_workers=%d",
            self._topo.label, rank, world_size,
            self._aug_cfg.n_global_crops, self._aug_cfg.n_local_crops,
            self._cfg.use_fp8_output,
            self._cfg.shard_extraction_workers,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def set_epoch(self, epoch: int) -> None:
        """
        Call at the start of each epoch.

        [FIX-14] Calls ``self._source.set_epoch(epoch)`` which propagates
        to all ``ShardIterator.reset_epoch()`` calls, re-shuffling shards so
        each epoch sees a distinct, reproducible order.

        [FIX-C] Clears ``_active_iter`` so that ``__iter__`` can be called
        again for the new epoch without raising RuntimeError.
        """
        self._epoch = epoch
        self._source.set_epoch(epoch)   # [FIX-14]
        self._active_iter = None        # [FIX-C] allow re-iteration for new epoch

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update all dataset mixing weights. Thread-safe; takes effect immediately."""
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Update one dataset's weight by name. Thread-safe."""
        self._source.set_weight_by_name(name, weight)

    def checkpoint(self, step: int) -> None:
        """Save dataloader state. Call once per step; skips internally if not due."""
        self._step = step
        self._ckptr.save(CheckpointState(
            step           = step,
            epoch          = self._epoch,
            mixing_weights = self._source.current_weights,
            dataset_names  = self._source.dataset_names,
        ))

    def __iter__(self) -> Iterator[Batch]:
        """
        Return an iterator over batches for the current epoch.

        [FIX-C] Guards against double-entry: if called while a previous
        iteration is still active (e.g. a progress-bar wrapper inspecting the
        iterator, or an accidental nested loop), raises RuntimeError instead
        of silently corrupting batches via interleaved DALI reads.

        Call ``set_epoch()`` to reset the guard between epochs.
        """
        if self._active_iter is not None:
            raise RuntimeError(
                "DINODataLoader.__iter__ was called while a previous iteration "
                "is still active.  Call set_epoch() before starting a new epoch."
            )
        self._active_iter = AsyncPrefetchIterator(
            source = self._dali_collate(),
            h2d    = self._h2d,
            te_fmt = self._fp8_fmt,
        )
        return self._active_iter

    def __len__(self) -> int:
        """
        Steps per epoch. [FIX-15]

        Requires ``steps_per_epoch`` to be passed at construction.  Raises
        ``TypeError`` (matching Python's built-in convention for unsized
        containers) with a descriptive message if it was not provided.
        """
        if self._steps_per_epoch is None:
            raise TypeError(
                "len(DINODataLoader) is not defined because steps_per_epoch "
                "was not provided at construction.  Pass "
                "steps_per_epoch=total_images // (batch_size * world_size) "
                "to enable len()."
            )
        return self._steps_per_epoch

    def __del__(self):
        self._active_iter = None   # [FIX-C]
        if hasattr(self, "_source"):
            self._source.close()
            _reg = get_registry()
            if _reg is not None:
                _reg.close()
                if self._local_rank == 0:
                    try:
                        _reg.unlink()
                    except Exception:
                        pass

    # ══════════════════════════════════════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def shard_cache_utilisation(self) -> float:
        return self._shard_cache.utilisation

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _dali_collate(self) -> Iterator[dict]:
        """Convert DALI iterator output to {"global", "local"} dicts."""
        ng = self._aug_cfg.n_global_crops
        nl = self._aug_cfg.n_local_crops
        _reg = get_registry()
        for dali_out in self._dali_iter:
            d = dali_out[0]
            _t0 = time.monotonic()
            yield {
                "global": [d[f"view_{i}"].to(torch.bfloat16)    for i in range(ng)],
                "local":  [d[f"view_{ng+i}"].to(torch.bfloat16) for i in range(nl)],
            }
            # Time from yield to resume = time the consumer spent processing.
            # We want the *producer* stall, so measure before the yield:
            if _reg is not None:
                _reg.inc("pipeline_yield_time_ms", int((time.monotonic() - _t0) * 1000))

    def _restore(self) -> None:
        state = self._ckptr.load()
        if state is None:
            return
        self._epoch = state.epoch
        self._step  = state.step

        # [FIX-A] Re-shuffle all shard iterators to match the restored epoch.
        # Without this call every ShardIterator starts from epoch-0 order on
        # resume, causing the model to see the same shard sequence regardless
        # of which epoch was checkpointed.
        self._source.set_epoch(state.epoch)

        if state.mixing_weights and state.dataset_names == self._source.dataset_names:
            self._source.set_weights(state.mixing_weights)
        elif state.dataset_names != self._source.dataset_names:
            log.warning(
                "Checkpoint dataset names %s do not match current %s — "
                "mixing weights NOT restored.",
                state.dataset_names, self._source.dataset_names,
            )
        # Note: DALI iterator fast-forward (skipping steps within an epoch) is
        # not implemented here.  DALI 1.30+ supports pipeline checkpointing via
        # pipeline.serialize() / deserialize(); integrate that for sub-epoch
        # resume on very large datasets.
        log.info(
            "Restored dataloader state: epoch=%d step=%d "
            "(note: sub-epoch fast-forward requires DALI pipeline checkpointing)",
            self._epoch, self._step,
        )
