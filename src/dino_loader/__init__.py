"""
dino_loader
===========
HPC-grade DINOv3 DALI data pipeline for B200 / GB200 NVL72 clusters.

Public API
----------
    from dino_loader import DatasetSpec, DINOAugConfig, LoaderConfig, DINODataLoader

    loader = DINODataLoader(specs, batch_size=512, config=LoaderConfig())
    for batch in loader:
        # batch.global_crops : list[Tensor]  — BF16 on GPU, or (FP8, FP8Meta) with TE
        # batch.local_crops  : list[Tensor]
        ...

Fix vs previous version
------------------------
[FIX-4] Import paths corrected to match the actual flat package structure.
        The previous __init__.py tried ``from dino_loader.loader import ...``
        which in turn imported from non-existent sub-packages
        (``dino_loader.augment.pipeline``, ``dino_loader.cache.memory``, etc.),
        making the entire public API dead on import.
"""

from dino_loader.config import DatasetSpec, DINOAugConfig, LoaderConfig

import logging
log = logging.getLogger(__name__)

__all__ = [
    "DatasetSpec",
    "DINOAugConfig",
    "LoaderConfig",
]

try:
    from dino_loader.loader      import DINODataLoader
    from dino_loader.memory      import Batch
    from dino_loader.distributed import (
        slurm_init,
        configure_nccl,
        ClusterTopology,
        detect_topology,
    )
    __all__.extend([
        "DINODataLoader",
        "Batch",
        "slurm_init",
        "configure_nccl",
        "ClusterTopology",
        "detect_topology",
    ])
except ImportError as e:
    # Expected when nvidia-dali or transformer-engine are absent
    # (e.g. in a plain CPU CI environment or on the login node).
    log.debug(
        "Could not import loader/distributed modules — expected if "
        "nvidia-dali is missing: %s", e
    )
