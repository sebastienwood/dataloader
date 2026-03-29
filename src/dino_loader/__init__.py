"""dino_loader
===========
HPC-grade DINOv3 DALI data pipeline for B200 / GB200 NVL72 clusters.

Public API
----------
    from dino_loader import (
        DatasetSpec, DINOAugConfig, LoaderConfig, NormStats,
        DINODataLoader, Batch,
    )
    from dino_loader.pipeline_graph import wrap_loader

    loader = DINODataLoader(specs, batch_size=512, config=LoaderConfig())

    # Compose post-DALI transforms in a stateful, checkpointable graph:
    pipeline = (
        wrap_loader(loader)
        .map(apply_ibot_masks)          # fn(Batch) → Batch
        .select(quality_ok)             # predicate(Batch) → bool
        .with_epoch(steps_per_epoch)    # limit steps per epoch
    )

    for epoch in range(100):
        pipeline.set_epoch(epoch)
        for batch in pipeline:
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

Phase 3 — NodePipeline (torchdata-backed stateful post-processing)
------------------------------------------------------------------
    from dino_loader.pipeline_graph import wrap_loader

    pipeline = (
        wrap_loader(DINODataLoader(...))
        .map(apply_ibot_masks)
        .select(quality_ok)
        .with_epoch(steps_per_epoch)
    )

New in this version
--------------------
- NormStats dataclass — canonical [0,1] normalisation stats with
  to_dali_scale() / to_numpy() helpers; eliminates ad-hoc ×255 conversions.
- MixingWeights — stable __init__(names, weights) + from_specs() classmethod.
- AugmentationSpec.norm_stats — all spec subclasses expose a NormStats property.
- PostProcessPipeline removed — wrap_loader() is the single post-processing
  entry point, backed by torchdata.nodes with full state_dict support.
- DINODataLoader.backend public property.
- DINODataLoader._step / ._epoch explicitly initialised and maintained.
- MaskMapNode now properly subclasses BaseNode; raises ValueError on empty
  global_crops instead of silently producing wrong-shaped masks.
"""

import logging

from dino_loader.config import DINOAugConfig, LoaderConfig, NormStats
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
    "NormStats",
    "ResolutionSource",
    "ShardReaderNode",
    "build_reader_graph",
    "wrap_loader",
]
