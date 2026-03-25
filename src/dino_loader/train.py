r"""train.py — Reference DINOv3 training script using dino_loader.

Demonstrates all improvements introduced in this version:

  1. [CFG-S1] shard_sampling='resampled' for small curated dataset
  2. [CFG-S2] prob= alias aligned with wds.RandomMix API
  3. [CFG-S3] debug_log_keys — per-sample key auditing
  4. [CFG-S4] fuse_normalization — per-dataset norm fused in DALI graph
  5. [CFG-S5] dali_fp8_output — optional in-graph FP8 cast (comment shows trade-off)
  6. [LD-8]   PostProcessPipeline — fluid interface for post-DALI transforms
  7. [LD-10]  empty_check watchdog — automatic stall detection

Also retained from previous version:
  - Resolution schedule (zero-downtime, automatic via set_epoch)
  - Per-shard quality filtering
  - MaskingGenerator (iBOT token masking)
  - StatefulDataLoader interface (state_dict / load_state_dict)
  - Curriculum dataset weight schedule

Why iBOT masking stays outside DALI
--------------------------------------
MaskingGenerator operates on ViT patch-level indices (a bool grid of shape
gridxgrid where grid = img_size // patch_size), not on image pixels.
DALI's computation graph only processes dense image tensors, so there is no
way to express patch-index masking as a DALI operator.  The CPU overhead is
~0.3 ms for a 37x37 grid — negligible compared to a 40 ms DALI decode step.

SLURM submission examples
--------------------------
GB200 NVL72 (72 GPUs / rack, 4 racks = 288 ranks):
    sbatch --nodes=4 --ntasks-per-node=72 --gres=gpu:72 \\
           --cpus-per-task=4 --mem=2048G                  \\
           --wrap="python train.py"

B200 PCIe (8 GPUs / node, 32 nodes = 256 ranks):
    sbatch --nodes=32 --ntasks-per-node=8 --gres=gpu:8  \\
           --cpus-per-task=16 --mem=512G                 \\
           --wrap="python train.py"
"""

from __future__ import annotations

import logging
import os

import torch
from dino_datasets import DatasetSpec
from dino_env import slurm_init

from dino_loader import (
    DINOAugConfig,
    DINODataLoader,
    LoaderConfig,
)
from dino_loader.masking import MaskingGenerator
from dino_loader.nodes import MaskMapNode

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("train")

_TOTAL_IMAGES = 50_000 * 10_000 + 30_000 * 10_000 + 5_000 * 10_000  # ~850 M


def main() -> None:
    # ── 1. Distributed init ────────────────────────────────────────────────────
    env    = slurm_init()
    device = torch.device(f"cuda:{env.local_rank % torch.cuda.device_count()}")  # noqa: F841

    # ── 2. Dataset catalogue ──────────────────────────────────────────────────
    #
    # [CFG-S1] imagenet22k uses shard_sampling='resampled' — it is small (5k
    #          shards) and we want to over-sample it proportionally to its
    #          weight=0.2 without cycling epochs manually.
    #
    # [CFG-S2] datacomp uses prob= (wds.RandomMix alias) instead of weight=
    #          to demonstrate API alignment.  Both are equivalent.
    #
    specs = [
        DatasetSpec(
            name               = "laion2b",
            shards             = [f"/lustre/laion2b/shard-{i:06d}.tar"  for i in range(50_000)],
            weight             = 0.5,
            shard_sampling     = "epoch",       # default — full deterministic pass
            min_sample_quality = 0.25,
            metadata_key       = "json",
        ),
        DatasetSpec(
            name               = "datacomp1b",
            shards             = [f"/lustre/datacomp/shard-{i:06d}.tar" for i in range(30_000)],
            prob               = 0.3,           # [CFG-S2] wds.RandomMix alias
            shard_sampling     = "epoch",
            min_sample_quality = 0.20,
            metadata_key       = "json",
            # Per-dataset normalisation stats (DataComp has slightly different dist.)
            mean               = (0.490, 0.460, 0.420),
            std                = (0.235, 0.228, 0.232),
        ),
        DatasetSpec(
            name           = "imagenet22k",
            shards         = [f"/lustre/in22k/shard-{i:06d}.tar" for i in range(5_000)],
            weight         = 0.2,
            shard_sampling = "resampled",   # [CFG-S1] infinite with-replacement
            metadata_key   = None,          # no sidecars — skip extraction
        ),
    ]

    # ── 3. Augmentation config ────────────────────────────────────────────────
    aug_cfg = DINOAugConfig(
        n_local_crops        = 8,
        preserve_aspect_ratio= True,
        resolution_schedule  = [(0, 224), (10, 448), (30, 518)],
        max_global_crop_size = 518,
        max_local_crop_size  = 224,
    )

    # ── 4. Loader config ──────────────────────────────────────────────────────
    batch_size = 512

    cfg = LoaderConfig(
        node_shm_gb              = 256,
        shard_prefetch_window    = 128,
        shard_extraction_workers = 8,
        dali_cpu_queue           = 8,
        dali_gpu_queue           = 6,
        hw_decoder_load          = 0.90,
        shuffle_buffer_size      = 512,

        # [CFG-S4] Fuse per-dataset mean/std into DALI graph — reduces one
        # kernel launch by merging normalize → cast → transpose.
        fuse_normalization       = True,

        # [CFG-S5] In-graph FP8 cast.
        # Set dali_fp8_output=True to fuse the FP8 cast into the DALI graph
        # (normalize → cast FP8 → transpose in one kernel).
        # Trade-off: FP8TensorMeta (TE rolling amax) is NOT available.
        # Use False (default) when you need te.fp8_autocast compatibility.
        use_fp8_output           = True,
        dali_fp8_output          = False,   # False → FP8Formatter post-DALI with TE meta

        stateful_dataloader      = True,
        checkpoint_dir           = f"/lustre/ckpts/{os.environ.get('SLURM_JOB_ID', 'local')}/dl",
        checkpoint_every_steps   = 500,

        # [CFG-S3] Debug: log sample keys to file (disable in production)
        debug_log_keys           = None,  # e.g. "/lustre/logs/sample_keys.tsv"
    )

    # ── 5. iBOT MaskingGenerator (CPU, post-DALI) ────────────────────────────
    #
    # iBOT masking cannot be fused into DALI: MaskingGenerator produces a
    # boolean grid over ViT patch indices (shape gridxgrid), not pixel-level
    # operations.  DALI only processes dense image tensors.  CPU overhead is
    # ~0.3 ms per batch — negligible vs 40 ms DALI decode.
    #
    patch_size = 14
    gen = MaskingGenerator(
       input_size=(aug_cfg.max_global_crop_size // patch_size,
                   aug_cfg.max_global_crop_size // patch_size),
       num_masking_patches=int(0.5 * (aug_cfg.max_global_crop_size // patch_size) ** 2),
   )

    # ── 6. Build loader ───────────────────────────────────────────────────────
    steps_per_epoch = _TOTAL_IMAGES // (batch_size * env.world_size)

    base_loader = DINODataLoader(
        specs            = specs,
        batch_size       = batch_size,
        aug_cfg          = aug_cfg,
        config           = cfg,
        device_id        = env.local_rank % torch.cuda.device_count(),
        local_rank       = env.local_rank,
        local_world_size = env.local_world_size,
        resume           = True,
        steps_per_epoch  = steps_per_epoch,
    )

    # ── 7. [LD-8] PostProcessPipeline — fluid API ─────────────────────────────
    #
    # Chain post-DALI transforms in a readable, composable style:
    #
    #   .map(fn)       — apply fn(Batch) → Batch on every batch
    #   .select(pred)  — drop batches where pred(Batch) is False
    #   .with_epoch(n) — limit to n steps per epoch (replaces steps_per_epoch)
    #
    # Each call returns a new PostProcessPipeline; the base_loader is unchanged.
    # The pipeline is lazy — transforms run as batches flow through.
    #
    # The empty_check watchdog [LD-10] fires inside _raw_iter() if no batch
    # is produced within 120s, giving a diagnostic message instead of hanging.
    #
    def quality_filter(batch) -> bool:
        """Example: skip batches where all metadata is None (no quality signal)."""  # noqa: D401
        if not batch.metadata:
            return True
        return any(m is not None for m in batch.metadata)

    loader = (
        base_loader
        .select(quality_filter)     # skip all-None-metadata batches
        .map(MaskMapNode.as_transform(gen))
        .with_epoch(steps_per_epoch)
    )

    # ── 8. Model (placeholder) ────────────────────────────────────────────────
    # model = TE_ViT_Giant(...).to(device) # noqa: ERA001
    # optim = torch.optim.AdamW(model.parameters(), lr=1e-3) # noqa: ERA001

    # ── 9. Training loop ──────────────────────────────────────────────────────
    for epoch in range(100):
        loader.set_epoch(epoch)

        # Curriculum: shift toward curated data over time
        if epoch == 20:  # noqa: PLR2004
            log.info("Epoch %d: curriculum shift — boosting imagenet22k weight", epoch)
            loader.set_weight_by_name("imagenet22k", 0.35)

        for step, batch in enumerate(loader):  # noqa: B007
            # batch.global_crops : list[Tensor[B,3,H,W]]  — BF16 (or FP8) on GPU
            # batch.local_crops  : list[Tensor[B,3,h,w]] # noqa: ERA001
            # batch.metadata     : list[Optional[dict]]   — per-sample JSON sidecar
            # batch.masks        : Optional[Tensor]        — iBOT patch masks

            # loss = model(batch) # noqa: ERA001
            # loss.backward() # noqa: ERA001
            # optim.step()  # noqa: ERA001

            loader.checkpoint(step)

            if step % 100 == 0 and env.rank == 0:
                log.info("epoch=%d step=%d/%d", epoch, step, steps_per_epoch)


if __name__ == "__main__":
    main()
