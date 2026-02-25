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

What this class does NOT do
----------------------------
- It does not contain any I/O, augmentation, or memory logic.
  Those live in dino_loader.io, dino_loader.augment, dino_loader.cache.
- It does not call dist.init_process_group(); that is the caller's job
  (use dino_loader.distributed.slurm_init for SLURM jobs).
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist

from dino_loader.augment.pipeline  import build_pipeline
from dino_loader.cache.memory      import (AsyncPrefetchIterator, Batch,
                                           FP8Formatter, H2DStream)
from dino_loader.checkpoint        import DataLoaderCheckpointer
from dino_loader.config            import (CheckpointState, DatasetSpec,
                                           DINOAugConfig, LoaderConfig)
from dino_loader.distributed       import ClusterTopology, detect_topology
from dino_loader.io.mixing_source  import MixingSource
from dino_loader.io.shard_cache    import NodeSharedShardCache

from dino_loader.monitor.metrics   import init_registry, get_registry
from dino_loader.monitor.tracing   import trace

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
    specs       : List of DatasetSpec (name, shards, initial weight).
    batch_size  : Per-GPU batch size.
    aug_cfg     : DINOAugConfig — augmentation hyper-parameters.
    config      : LoaderConfig — all infrastructure knobs.
    device_id   : Local GPU index.
    rank, world_size, local_rank, local_world_size : distributed identity.
                  If a process group is already initialised, these are
                  inferred automatically.
    resume      : If True, attempt to load the latest dataloader checkpoint.

    Example
    -------
        env = slurm_init()
        loader = DINODataLoader(
            specs      = [DatasetSpec("laion", shards, weight=1.0)],
            batch_size = 512,
            config     = LoaderConfig(node_shm_gb=256),
            device_id  = env.local_rank,
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
        aug_cfg:          Optional[DINOAugConfig]  = None,
        config:           Optional[LoaderConfig]   = None,
        device_id:        int                      = 0,
        rank:             int                      = 0,
        world_size:       int                      = 1,
        local_rank:       int                      = 0,
        local_world_size: int                      = 8,
        resume:           bool                     = False,
    ):
        if not HAS_DALI:
            raise ImportError("nvidia-dali not installed — DINODataLoader cannot be instantiated")
        # ── Resolve distributed identity ──────────────────────────────────────
        if dist.is_available() and dist.is_initialized():
            rank             = dist.get_rank()
            world_size       = dist.get_world_size()
            local_rank       = int(os.environ.get("LOCAL_RANK", local_rank))

        self._rank       = rank
        self._world      = world_size
        self._device     = torch.device(f"cuda:{device_id}")
        self._aug_cfg    = aug_cfg or DINOAugConfig()
        self._cfg        = config  or LoaderConfig()
        self._epoch      = 0
        self._step       = 0

        # ── Topology (already detected & NCCL configured by slurm_init;
        #    detect again here for standalone / non-SLURM use)  ─────────────
        self._topo = detect_topology(force=self._cfg.force_topology)

        # ── Initialize Metrics ────────────────────────────────────────────────
        init_registry(
            job_id=os.environ.get("SLURM_JOB_ID", "dino"),
            create=(local_rank == 0),
            local_rank=local_rank
        )

        # ── Stage 1: Node-local shared shard cache ────────────────────────────
        node_master = (local_rank == 0)
        self._shard_cache = NodeSharedShardCache(
            node_master     = node_master,
            job_id          = os.environ.get("SLURM_JOB_ID", "dino"),
            max_shm_gb      = self._cfg.node_shm_gb,
            prefetch_window = self._cfg.shard_prefetch_window,
            shard_timeout_s = self._cfg.shard_timeout_s,
        )
        # Barrier: ensure node master has initialised /dev/shm before others read
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
            pipelines        = [self._pipe],
            output_map       = _view_names,
            last_batch_policy= LastBatchPolicy.DROP,
            auto_reset       = True,
        )

        # ── Stage 4: H2D transfer stream ─────────────────────────────────────
        self._h2d = H2DStream(self._device, self._topo)

        # ── Stage 5: Optional FP8 / TE output formatter ───────────────────────
        self._fp8_fmt: Optional[FP8Formatter] = None
        if self._cfg.use_fp8_output:
            self._fp8_fmt = FP8Formatter(self._device)

        # ── Checkpointer ──────────────────────────────────────────────────────
        self._ckptr = DataLoaderCheckpointer(
            ckpt_dir     = self._cfg.checkpoint_dir,
            every_n_steps= self._cfg.checkpoint_every_steps,
            rank         = rank,
        )
        if resume:
            self._restore()

        log.info(
            "DINODataLoader ready | topology=%s | rank=%d/%d | "
            "views=%dg+%dl | FP8=%s",
            self._topo.label, rank, world_size,
            self._aug_cfg.n_global_crops, self._aug_cfg.n_local_crops,
            self._cfg.use_fp8_output,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch."""
        self._epoch = epoch

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
        return AsyncPrefetchIterator(
            source  = self._dali_collate(),
            h2d     = self._h2d,
            te_fmt  = self._fp8_fmt,
        )

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
        """Convert DALI iterator output (list of dicts) to {"global", "local"} dicts."""
        ng = self._aug_cfg.n_global_crops
        nl = self._aug_cfg.n_local_crops
        registry = get_registry()

        import time
        while True:
            t0 = time.time()
            with trace("dali_wait", "pipeline"):
                try:
                    dali_out = next(self._dali_iter)
                except StopIteration:
                    break
            
            if registry:
                registry.set("pipeline_yield_time_ms", float((time.time() - t0) * 1000.0))
                registry.inc("loader_batches_yielded", 1)

            # d is a dict of strings to DALI tensors
            d = dali_out[0]
            # Cast DALI FLOAT16 → PyTorch bfloat16 for B200 tensor cores
            yield {
                "global": [d[f"view_{i}"].to(torch.bfloat16)      for i in range(ng)],
                "local":  [d[f"view_{ng+i}"].to(torch.bfloat16)   for i in range(nl)],
            }

    def _restore(self) -> None:
        state = self._ckptr.load()
        if state is None:
            return
        self._epoch = state.epoch
        self._step  = state.step
        if state.mixing_weights and state.dataset_names == self._source.dataset_names:
            self._source.set_weights(state.mixing_weights)
        elif state.dataset_names != self._source.dataset_names:
            log.warning(
                "Checkpoint dataset names %s do not match current %s — "
                "mixing weights NOT restored.",
                state.dataset_names, self._source.dataset_names,
            )
