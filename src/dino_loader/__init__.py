"""dino_loader
===========
HPC-grade DINOv3 DALI data pipeline pour clusters B200 / GB200 NVL72.

Public API
----------
    from dino_loader import (
        DatasetSpec, DINOAugConfig, LoaderConfig, NormStats,
        DINODataLoader, Batch,
    )
    from dino_loader.pipeline_graph import wrap_loader

    loader = DINODataLoader(specs, batch_size=512, config=LoaderConfig())

    # Pipeline post-DALI composable et stateful :
    pipeline = (
        wrap_loader(loader)
        .map(apply_ibot_masks)
        .select(quality_ok)
        .with_epoch(steps_per_epoch)
    )

    for epoch in range(100):
        pipeline.set_epoch(epoch)
        for batch in pipeline:
            ...

Sources
-------
::

    from dino_loader.sources import MixingSource, WDSSource, SourceProtocol

Graph I/O (stages 1-2)
-----------------------
::

    from dino_loader.shard_reader import ShardReaderNode, build_reader_graph

    loader, reader = build_reader_graph(specs, batch_size=512, cache=cache, ...)
    reader.set_epoch(0)
    for jpegs, meta in loader:
        my_augment(jpegs)

"""

import logging

from dino_loader.config import DINOAugConfig, LoaderConfig, NormStats
from dino_loader.loader import DINODataLoader
from dino_loader.memory import Batch
from dino_loader.pipeline_graph import (
    BatchFilterNode,
    BatchMapNode,
    MaskMapNode,
    MetadataNode,
    NodePipeline,
    wrap_loader,
)
from dino_loader.shard_reader import ShardReaderNode, build_reader_graph
from dino_loader.sources import MixingSource, SourceProtocol, WDSSource
from dino_loader.sources.resolution import ResolutionSource

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
    "MixingSource",
    "NodePipeline",
    "NormStats",
    "ResolutionSource",
    "ShardReaderNode",
    "SourceProtocol",
    "WDSSource",
    "build_reader_graph",
    "wrap_loader",
]
