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
"""

from dino_loader.config   import DatasetSpec, DINOAugConfig, LoaderConfig
from dino_loader.loader   import DINODataLoader, Batch
from dino_loader.distributed import slurm_init, configure_nccl, ClusterTopology, detect_topology

__all__ = [
    "DatasetSpec",
    "DINOAugConfig",
    "LoaderConfig",
    "DINODataLoader",
    "Batch",
    "slurm_init",
    "configure_nccl",
    "ClusterTopology",
    "detect_topology",
]
