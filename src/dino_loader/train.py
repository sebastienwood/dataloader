"""
train.py — Reference DINOv3 training script using dino_loader.

Demonstrates all major features introduced in this version:
  - Resolution schedule (zero-downtime, automatic via set_epoch)
  - Per-shard quality filtering
  - Intra-shard shuffle buffer
  - MaskingGenerator (iBOT token masking)
  - StatefulDataLoader interface (state_dict / load_state_dict)
  - Curriculum dataset weight schedule

SLURM submission examples
--------------------------
GB200 NVL72 (72 GPUs / rack, 4 racks):
    sbatch --nodes=4 --ntasks-per-node=72 --gres=gpu:72 \\
           --cpus-per-task=4 --mem=2048G                 \\
           --wrap="python train.py"

B200 PCIe (8 GPUs / node, 32 nodes):
    sbatch --nodes=32 --ntasks-per-node=8 --gres=gpu:8 \\
           --cpus-per-task=16 --mem=512G                \\
           --wrap="python train.py"
"""

import logging
import os
import time

import torch
import transformer_engine.pytorch as te

from dino_loader import (
    DatasetSpec,
    DINOAugConfig,
    DINODataLoader,
    LoaderConfig,
    slurm_init,
)

# DinoV3 MaskingGenerator — import if available, skip gracefully otherwise
try:
    from dinov3.data.masking import MaskingGenerator
    HAS_MASKING = True
except ImportError:
    HAS_MASKING = False

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("train")

_TOTAL_IMAGES = 50_000 * 10_000 + 30_000 * 10_000 + 5_000 * 10_000  # ~850 M


def main():
    # ── 1. Distributed init ────────────────────────────────────────────────────
    env    = slurm_init()
    device = torch.device(f"cuda:{env.local_rank % torch.cuda.device_count()}")

    # ── 2. Dataset catalogue ──────────────────────────────────────────────────
    #
    # shard_quality_scores: optional per-shard score [0,1] for weighted sampling
    # min_sample_quality:   drop samples whose .json "quality_score" < threshold
    # metadata_key:         sidecar extension (default "json"); None = legacy fast path
    #
    specs = [
        DatasetSpec(
            name               = "laion2b",
            shards             = [f"/lustre/laion2b/shard-{i:06d}.tar"  for i in range(50_000)],
            weight             = 0.5,
            min_sample_quality = 0.25,      # discard lowest-quality 25%
            metadata_key       = "json",
        ),
        DatasetSpec(
            name               = "datacomp1b",
            shards             = [f"/lustre/datacomp/shard-{i:06d}.tar" for i in range(30_000)],
            weight             = 0.3,
            min_sample_quality = 0.20,
            metadata_key       = "json",
        ),
        DatasetSpec(
            name               = "imagenet22k",
            shards             = [f"/lustre/in22k/shard-{i:06d}.tar"   for i in range(5_000)],
            weight             = 0.2,
            metadata_key       = None,      # no sidecars — skip extraction
        ),
    ]

    # ── 3. Augmentation config — with progressive resolution schedule ─────────
    #
    # Resolution schedule: start at 224, move to 448 at epoch 10, 518 at epoch 30.
    # max_global_crop_size must be set to the ceiling to pre-allocate DALI buffers.
    # No DALI rebuild occurs during training — ResolutionSource handles it.
    #
    aug_cfg = DINOAugConfig(
        n_local_crops        = 8,
        preserve_aspect_ratio= True,
        resolution_schedule  = [(0, 224), (10, 448), (30, 518)],
        max_global_crop_size = 518,
        max_local_crop_size  = 224,   # scales proportionally
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
        shuffle_buffer_size      = 512,     # intra-shard sample shuffle
        use_fp8_output           = True,
        stateful_dataloader      = True,    # enable state_dict / load_state_dict
        checkpoint_dir           = f"/lustre/ckpts/{os.environ['SLURM_JOB_ID']}/dl",
        checkpoint_every_steps   = 500,
    )

    # ── 5. Optional MaskingGenerator (iBOT token masking) ────────────────────
    mask_generator = None
    if HAS_MASKING:
        patch_size = 14
        img_size   = aug_cfg.max_global_crop_size
        grid       = img_size // patch_size
        mask_generator = MaskingGenerator(
            input_size       = (grid, grid),
            max_num_patches  = int(0.5 * grid * grid),
        )

    # ── 6. Build loader ───────────────────────────────────────────────────────
    steps_per_epoch = _TOTAL_IMAGES // (batch_size * env.world_size)

    loader = DINODataLoader(
        specs            = specs,
        batch_size       = batch_size,
        aug_cfg          = aug_cfg,
        config           = cfg,
        device_id        = env.local_rank % torch.cuda.device_count(),
        local_rank       = env.local_rank,
        local_world_size = env.local_world_size,
        resume           = True,
        steps_per_epoch  = steps_per_epoch,
        mask_generator   = mask_generator,
    )

    # ── 7. (Placeholder) ViT model ────────────────────────────────────────────
    # model = TE_ViT_Giant(...).to(device)
    # optim = ...

    # ── 8. Training loop ──────────────────────────────────────────────────────
    for epoch in range(100):
        # set_epoch re-shuffles shards AND auto-applies resolution schedule
        loader.set_epoch(epoch)

        # Manual data curriculum (alternative to resolution_schedule for weights)
        if epoch == 10:
            loader.set_weights([0.4, 0.4, 0.2])
        elif epoch == 30:
            loader.set_weights([0.2, 0.4, 0.4])

        t_epoch = time.perf_counter()

        for step, batch in enumerate(loader):
            # batch.global_crops : 2 × (FP8Tensor, FP8Meta) ready for TE
            # batch.local_crops  : 8 × (FP8Tensor, FP8Meta)
            # batch.metadata     : list[Optional[dict]] — quality_score etc.
            # batch.masks        : Optional[BoolTensor(B, N_tokens)]

            with te.fp8_autocast(enabled=True):
                # g0, g0_meta = batch.global_crops[0]
                # student_out = model(g0, g0_meta)
                pass

            # Backward, optimiser step, EMA teacher update ...

            # ── Checkpoint via manual API ─────────────────────────────────────
            loader.checkpoint(step)

            # ── Or via StatefulDataLoader interface (e.g. Lightning calls this) ──
            # sd = loader.state_dict()
            # ... serialise sd to disk ...
            # loader.load_state_dict(sd)

            if env.rank == 0 and step % 200 == 0:
                elapsed      = time.perf_counter() - t_epoch
                imgs_per_sec = (batch_size * env.world_size * (step + 1)) / max(elapsed, 1e-6)
                log.info(
                    "E%03d S%06d/%06d | %.1fk img/s | shm %.0f%% | res %dx%d",
                    epoch, step, len(loader),
                    imgs_per_sec / 1000,
                    loader.shard_cache_utilisation * 100,
                    loader._current_global_size,
                    loader._current_local_size,
                )


if __name__ == "__main__":
    main()
