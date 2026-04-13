"""dino_loader.norm_utils
========================
Utilitaires de normalisation partagés entre les pipelines DALI statique et
dynamique.

Ces fonctions encapsulent la logique de construction des tableaux ``mean``
et ``std`` par sample à partir d'une table de ``NormStats`` indexée par
dataset.  Elles sont utilisées par :

- ``dino_loader.pipeline.NormSource`` (DALI statique, ExternalSource callback)
- ``dino_loader.experimental.dynamic_pipeline._make_dinov2_aug_fn``

Toutes les valeurs sont en échelle [0, 255] (DALI scale), converties une seule
fois depuis les ``NormStats`` stockées en [0, 1].
"""

import numpy as np

from dino_loader.config import DINOAugConfig, NormStats


def build_norm_table(
    aug_cfg: DINOAugConfig,
    specs:   list,
) -> list[NormStats]:
    """Construit une table de ``NormStats`` indexée par dataset.

    Utilise les stats globales de ``aug_cfg`` comme fallback pour les datasets
    sans override.

    Args:
        aug_cfg: Config d'augmentation globale (fournit le fallback mean/std).
        specs:   Spécifications de datasets (peuvent porter des mean/std propres).

    Returns:
        Liste de ``NormStats`` alignée avec ``specs`` (index i → specs[i]).
        Les stats sont en échelle [0, 1].

    """
    global_stats = aug_cfg.norm_stats
    return [
        NormStats.from_config(
            mean     = spec.mean,
            std      = spec.std,
            fallback = global_stats,
        )
        for spec in specs
    ]


def build_norm_arrays(
    indices:    list[int],
    norm_table: list[NormStats],
    fallback:   NormStats,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit des tableaux ``mean`` et ``std`` par sample en échelle [0, 255].

    Chaque sample reçoit les stats de son dataset source — pas un scalaire
    partagé sur tout le batch.

    Args:
        indices:    Indices de dataset par sample (longueur = batch_size).
        norm_table: Table de NormStats en [0, 1], indexée par dataset.
        fallback:   Stats utilisées si ``indices`` est vide ou si un indice
                    dépasse la longueur de la table.

    Returns:
        Paire ``(batch_means, batch_stds)`` de shape ``(B, 3)`` en float32,
        en échelle [0, 255].

    """
    if not indices:
        b            = 1
        mean_255, std_255 = fallback.to_dali_scale()
        batch_means  = np.tile(np.array(mean_255, dtype=np.float32), (b, 1))
        batch_stds   = np.tile(np.array(std_255,  dtype=np.float32), (b, 1))
        return batch_means, batch_stds

    n = len(norm_table)
    batch_means = np.stack([
        np.array(norm_table[min(i, n - 1)].to_dali_scale()[0], dtype=np.float32)
        for i in indices
    ])
    batch_stds = np.stack([
        np.array(norm_table[min(i, n - 1)].to_dali_scale()[1], dtype=np.float32)
        for i in indices
    ])
    return batch_means, batch_stds
