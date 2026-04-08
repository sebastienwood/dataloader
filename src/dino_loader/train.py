r"""train.py — Script de référence DINOv3 utilisant dino_loader.

Illustre les fonctionnalités principales :

  1. Spécification multi-dataset avec poids et filtres qualité
  2. ``shard_sampling='resampled'`` pour les petits datasets curatés
  3. ``fuse_normalization`` — normalisation par-dataset fusionnée dans DALI
  4. ``dali_fp8_output`` — cast FP8 optionnel dans le graphe DALI
  5. API de composition fluide (``map``, ``select``, ``with_epoch``)
  6. Curriculum de résolution progressive via ``resolution_schedule``
  7. Masquage de patches iBOT (``MaskingGenerator``, CPU post-DALI)
  8. Interface ``StatefulDataLoader`` (``state_dict`` / ``load_state_dict``)
  9. Curriculum de poids de datasets

Pourquoi le masquage iBOT reste hors de DALI
----------------------------------------------
``MaskingGenerator`` opère sur des indices de patches ViT (grille bool de forme
``grid × grid`` où ``grid = img_size // patch_size``), et non sur des pixels.
Le graphe DALI ne traite que des tenseurs image denses, rendant impossible
l'expression d'opérations sur des indices de patches. Le surcoût CPU est
d'environ 0,3 ms pour une grille 37×37 — négligeable face aux ~40 ms du
décodage DALI.

Exemples de soumission SLURM
-----------------------------
GB200 NVL72 (72 GPU / rack, 4 racks = 288 rangs) :

    sbatch --nodes=4 --ntasks-per-node=72 --gres=gpu:72 \\
           --cpus-per-task=4 --mem=2048G                  \\
           --wrap="python train.py"

B200 PCIe (8 GPU / nœud, 32 nœuds = 256 rangs) :

    sbatch --nodes=32 --ntasks-per-node=8 --gres=gpu:8  \\
           --cpus-per-task=16 --mem=512G                 \\
           --wrap="python train.py"
"""

from __future__ import annotations

import logging
import os

import torch
from dino_datasets import DatasetSpec
from dino_env import init

from dino_loader import (
    DINOAugConfig,
    DINODataLoader,
    LoaderConfig,
)
from dino_loader.masking import MaskingGenerator
from dino_loader.pipeline_graph import MaskMapNode

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("train")

_TOTAL_IMAGES = 50_000 * 10_000 + 30_000 * 10_000 + 5_000 * 10_000  # ~850 M


def main() -> None:
    # ── 1. Init distribué ─────────────────────────────────────────────────────
    env    = init()
    device = torch.device(f"cuda:{env.local_rank % torch.cuda.device_count()}")  # noqa: F841

    # ── 2. Catalogue des datasets ─────────────────────────────────────────────
    #
    # imagenet22k utilise shard_sampling='resampled' : petit dataset (5k shards)
    # sur-échantillonné proportionnellement à son poids sans cycling manuel.
    #
    # datacomp utilise une normalisation par-dataset (stats distinctes d'ImageNet).
    specs = [
        DatasetSpec(
            name               = "laion2b",
            shards             = [f"/lustre/laion2b/shard-{i:06d}.tar"  for i in range(50_000)],
            weight             = 0.5,
            shard_sampling     = "epoch",
            min_sample_quality = 0.25,
            metadata_key       = "json",
        ),
        DatasetSpec(
            name               = "datacomp1b",
            shards             = [f"/lustre/datacomp/shard-{i:06d}.tar" for i in range(30_000)],
            weight             = 0.3,
            shard_sampling     = "epoch",
            min_sample_quality = 0.20,
            metadata_key       = "json",
            mean               = (0.490, 0.460, 0.420),
            std                = (0.235, 0.228, 0.232),
        ),
        DatasetSpec(
            name           = "imagenet22k",
            shards         = [f"/lustre/in22k/shard-{i:06d}.tar" for i in range(5_000)],
            weight         = 0.2,
            shard_sampling = "resampled",
            metadata_key   = None,
        ),
    ]

    # ── 3. Config d'augmentation ──────────────────────────────────────────────
    aug_cfg = DINOAugConfig(
        n_local_crops         = 8,
        preserve_aspect_ratio = True,
        resolution_schedule   = [(0, 224), (10, 448), (30, 518)],
        max_global_crop_size  = 518,
        max_local_crop_size   = 224,
    )

    # ── 4. Config du loader ───────────────────────────────────────────────────
    batch_size = 512

    cfg = LoaderConfig(
        node_shm_gb           = 256,
        shard_prefetch_window = 128,
        dali_cpu_queue        = 8,
        dali_gpu_queue        = 6,
        hw_decoder_load       = 0.90,
        shuffle_buffer_size   = 512,

        # Fusionner la normalisation par-dataset dans le graphe DALI :
        # réduit un kernel launch en fusionnant normalize → cast → transpose.
        fuse_normalization    = True,

        # Cast FP8 post-DALI via Transformer Engine (rolling amax, compatible
        # te.fp8_autocast). Mettre dali_fp8_output=True pour fusionner le cast
        # dans le graphe DALI (plus rapide, mais sans FP8TensorMeta TE).
        use_fp8_output        = True,
        dali_fp8_output       = False,

        stateful_dataloader   = True,
        checkpoint_dir        = f"/lustre/ckpts/{os.environ.get('SLURM_JOB_ID', 'local')}/dl",
        checkpoint_every_steps = 500,
    )

    # ── 5. MaskingGenerator iBOT (CPU, post-DALI) ─────────────────────────────
    patch_size = 14
    gen = MaskingGenerator(
        input_size           = (aug_cfg.max_global_crop_size // patch_size,
                                aug_cfg.max_global_crop_size // patch_size),
        num_masking_patches  = int(
            0.5 * (aug_cfg.max_global_crop_size // patch_size) ** 2
        ),
    )

    # ── 6. Construction du loader ─────────────────────────────────────────────
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

    # ── 7. Composition du pipeline ────────────────────────────────────────────
    #
    # L'API fluide permet de chaîner des transforms post-DALI :
    #   .map(fn)       → applique fn(Batch) → Batch sur chaque batch
    #   .select(pred)  → ignore les batches où pred(Batch) est False
    #   .with_epoch(n) → limite à n steps par époque
    #
    # Chaque appel retourne un nouveau NodePipeline ; base_loader est intact.
    # Les transforms s'exécutent paresseusement au fil de l'itération.
    def quality_filter(batch) -> bool:
        """Ignore les batches sans aucun signal de qualité (tout None)."""
        if not batch.metadata:
            return True
        return any(m is not None for m in batch.metadata)

    loader = (
        base_loader
        .select(quality_filter)
        .map(MaskMapNode.as_transform(gen))
        .with_epoch(steps_per_epoch)
    )

    # ── 8. Modèle (placeholder) ───────────────────────────────────────────────
    # model = TE_ViT_Giant(...).to(device)
    # optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ── 9. Boucle d'entraînement ──────────────────────────────────────────────
    for epoch in range(100):
        loader.set_epoch(epoch)

        # Curriculum : augmenter le poids du dataset curatés au fil du temps
        if epoch == 20:  # noqa: PLR2004
            log.info("Époque %d : curriculum — augmentation du poids imagenet22k", epoch)
            loader.set_weight_by_name("imagenet22k", 0.35)

        for step, batch in enumerate(loader):  # noqa: B007
            # batch.global_crops : list[Tensor[B,3,H,W]]  — BF16 (ou FP8) sur GPU
            # batch.local_crops  : list[Tensor[B,3,h,w]]
            # batch.metadata     : list[Optional[dict]]   — sidecar JSON par sample
            # batch.masks        : Optional[Tensor]        — masques de patches iBOT

            # loss = model(batch)
            # loss.backward()
            # optim.step()

            loader.checkpoint(step)

            if step % 100 == 0 and env.rank == 0:
                log.info("epoch=%d step=%d/%d", epoch, step, steps_per_epoch)


if __name__ == "__main__":
    main()