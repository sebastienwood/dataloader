"""
dino_loader
===========
HPC-grade DINOv3 DALI data pipeline for B200 / GB200 NVL72 clusters.

Public API
----------
    from dino_loader import (
        DatasetSpec, DINOAugConfig, LoaderConfig,
        DINODataLoader, Batch, slurm_init,
    )

    loader = DINODataLoader(specs, batch_size=512, config=LoaderConfig())
    for batch in loader:
        # batch.global_crops : list[Tensor]  — BF16 on GPU (or FP8+meta with TE)
        # batch.local_crops  : list[Tensor]
        # batch.metadata     : list[Optional[dict]]  — .json sidecar per sample
        # batch.masks        : Optional[Tensor]      — iBOT token masks
        ...

New in this version
--------------------
- DatasetSpec: shard_quality_scores, min_sample_quality, metadata_key, mean/std
- DINOAugConfig: preserve_aspect_ratio, resolution_schedule, max_*_crop_size
- LoaderConfig: shuffle_buffer_size, stateful_dataloader
- DINODataLoader.set_resolution(global, local) — zero-downtime resolution change
- DINODataLoader.state_dict() / load_state_dict() — StatefulDataLoader interface
- Batch.metadata, Batch.masks
- ResolutionSource exposed for advanced pipeline customisation
"""

from dino_loader.config import DINOAugConfig, LoaderConfig

import logging
log = logging.getLogger(__name__)

__all__ = [
    "DINOAugConfig",
    "LoaderConfig",
]

try:
    from dino_loader.loader        import DINODataLoader
    from dino_loader.memory        import Batch
    from dino_loader.mixing_source import ResolutionSource
    __all__.extend([
        "DINODataLoader",
        "Batch",
        "ResolutionSource",
    ])
except ImportError as e:
    log.debug(
        "Could not import loader/distributed modules — expected if "
        "nvidia-dali is missing: %s", e
    )
