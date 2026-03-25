"""dino_loader
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

Phase 1 — torchdata.nodes integration
--------------------------------------
    from dino_loader.nodes import ShardReaderNode, build_reader_graph

    loader, reader = build_reader_graph(specs, batch_size=512, cache=cache, ...)
    for epoch in range(100):
        reader.set_epoch(epoch)
        for jpegs, meta in loader:
            my_augment(jpegs)

Phase 3 — NodePipeline (torchdata-backed PostProcessPipeline)
--------------------------------------------------------------
    from dino_loader.pipeline_graph import wrap_loader

    pipeline = (
        wrap_loader(DINODataLoader(...))
        .map(apply_ibot_masks)
        .select(quality_ok)
        .with_epoch(steps_per_epoch)
    )

New in this version
--------------------
- DatasetSpec: shard_quality_scores, min_sample_quality, metadata_key, mean/std
- DINOAugConfig: preserve_aspect_ratio, resolution_schedule, max_*_crop_size
- LoaderConfig: shuffle_buffer_size, stateful_dataloader
- DINODataLoader.set_resolution(global, local) — zero-downtime resolution change
- DINODataLoader.state_dict() / load_state_dict() — StatefulDataLoader interface
- Batch.metadata, Batch.masks
- ResolutionSource exposed for advanced pipeline customisation
- ShardReaderNode / build_reader_graph — torchdata.nodes integration (Phase 1)
- wrap_loader / NodePipeline — composable stateful pipeline (Phase 3)
"""

import logging

from dino_loader.config import DINOAugConfig, LoaderConfig
from dino_loader.loader import DINODataLoader
from dino_loader.memory import Batch
from dino_loader.mixing_source import ResolutionSource
from dino_loader.nodes import (
    MaskMapNode,
    MetadataNode,
    ShardReaderNode,
    build_reader_graph,
)
from dino_loader.pipeline_graph import (  # noqa: F401
    BatchFilterNode,
    BatchMapNode,
    NodePipeline,
    wrap_loader,
)

log = logging.getLogger(__name__)

__all__ = [
    "Batch",
    "BatchFilterNode",
    "BatchMapNode",
    "DINOAugConfig",
    "DINODataLoader",
    "LoaderConfig",
    "MaskMapNode",
    "MetadataNode",
    "NodePipeline",
    "ResolutionSource",
    "ShardReaderNode",
    "build_reader_graph",
]
