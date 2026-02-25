"""
train.py — Minimal DINOv3 training script using dino_loader.

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

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("train")


def main():
    # ── 1. Distributed init (topology detection + NCCL config inside) ─────────
    env = slurm_init()                     # raises clearly if SLURM vars missing
    device = torch.device(f"cuda:{env.local_rank % torch.cuda.device_count()}")

    # ── 2. Dataset catalogue ──────────────────────────────────────────────────
    specs = [
        DatasetSpec(
            name    = "laion2b",
            shards  = [f"/lustre/laion2b/shard-{i:06d}.tar" for i in range(50_000)],
            weight  = 0.5,
        ),
        DatasetSpec(
            name    = "datacomp1b",
            shards  = [f"/lustre/datacomp/shard-{i:06d}.tar" for i in range(30_000)],
            weight  = 0.3,
        ),
        DatasetSpec(
            name    = "imagenet22k",
            shards  = [f"/lustre/in22k/shard-{i:06d}.tar"   for i in range(5_000)],
            weight  = 0.2,
        ),
    ]

    # ── 3. Config ─────────────────────────────────────────────────────────────
    aug_cfg = DINOAugConfig(n_local_crops=8)
    cfg     = LoaderConfig(
        node_shm_gb            = 256,   # GB200 NVL72 nodes have 2 TB RAM
        shard_prefetch_window  = 128,
        dali_cpu_queue         = 8,
        dali_gpu_queue         = 6,
        hw_decoder_load        = 0.90,
        use_fp8_output         = True,
        checkpoint_dir         = f"/lustre/ckpts/{os.environ['SLURM_JOB_ID']}/dl",
        checkpoint_every_steps = 500,
    )

    # ── 4. Loader ─────────────────────────────────────────────────────────────
    loader = DINODataLoader(
        specs            = specs,
        batch_size       = 512,           # per-GPU; 512 × 288 GPUs = 147k global
        aug_cfg          = aug_cfg,
        config           = cfg,
        device_id        = env.local_rank % torch.cuda.device_count(),
        local_rank       = env.local_rank,
        local_world_size = env.local_world_size,
        resume           = True,          # safe to always pass True
    )

    # ── 5. (Placeholder) ViT model ────────────────────────────────────────────
    # model = TE_ViT_Giant(...).to(device)
    # ...

    # ── 6. Training loop ──────────────────────────────────────────────────────
    for epoch in range(100):
        loader.set_epoch(epoch)

        # Example: data curriculum — shift toward curated data over time
        if epoch == 10:
            loader.set_weights([0.4, 0.4, 0.2])
        elif epoch == 30:
            loader.set_weights([0.2, 0.4, 0.4])

        t_epoch = time.perf_counter()
        for step, batch in enumerate(loader):
            # batch.global_crops : 2 × (FP8Tensor, FP8Meta) ready for TE
            # batch.local_crops  : 8 × (FP8Tensor, FP8Meta)

            with te.fp8_autocast(enabled=True):
                # forward pass through TE-wrapped ViT
                # g0, g0_meta = batch.global_crops[0]
                # student_out = model(g0, g0_meta)
                pass

            # Backward, optimiser step, EMA teacher update …

            loader.checkpoint(step)

            if env.rank == 0 and step % 200 == 0:
                elapsed = time.perf_counter() - t_epoch
                imgs_per_sec = (512 * env.world_size * (step + 1)) / max(elapsed, 1e-6)
                log.info(
                    "E%03d S%06d | %.1fk img/s | shm %.0f%%",
                    epoch, step,
                    imgs_per_sec / 1000,
                    loader.shard_cache_utilisation * 100,
                )


if __name__ == "__main__":
    main()
