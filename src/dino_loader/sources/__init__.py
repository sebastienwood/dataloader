"""dino_loader.sources
====================
Sources de données pour le pipeline dino_loader.

Ce module expose deux stratégies de lecture de shards WebDataset, toutes
deux conformes à ``SourceProtocol`` et interchangeables du point de vue
du reste du codebase.

HPC source (``hpc_source``)
    Conçue pour les clusters HPC avec Lustre et beaucoup de rangs par nœud.
    Combine un cache ``/dev/shm`` (``NodeSharedShardCache``) avec un double-
    buffering strict I/O + extraction.  C'est la source de production par
    défaut sur B200 / GB200 NVL72.

WDS source (``wds_source``)
    Alternative basée sur ``webdataset``, plus simple, recommandée quand
    les shards sont déjà en mémoire rapide (NVMe local, Lustre MDS rapide)
    ou quand la simplicité prime sur la latence absolue.

Interface commune
-----------------
``SourceProtocol`` garantit que les deux sources sont interchangeables.
Tout consommateur de source (``ShardReaderNode``, ``_ReaderAdapter``,
backends) doit typer ses arguments avec ce protocol.

Partagé entre les deux sources
    ``MixingWeights`` (``_weights``) — vecteur de poids normalisé thread-safe.
    ``ResolutionSource`` (``resolution``) — holder thread-safe de la résolution
    de crop courante, utilisé comme callback DALI ExternalSource.

Exports publics
---------------
    from dino_loader.sources import (
        SourceProtocol,
        MixingWeights,
        ResolutionSource,
        SampleRecord,
        ShardIterator,
        MixingSource,
        WDSSource,
        WDSShardReaderNode,
    )
"""

from dino_loader.sources._weights import MixingWeights
from dino_loader.sources.hpc_source import MixingSource, SampleRecord, ShardIterator
from dino_loader.sources.protocol import SourceProtocol
from dino_loader.sources.resolution import ResolutionSource
from dino_loader.sources.wds_source import WDSShardReaderNode, WDSSource

__all__ = [
    "MixingSource",
    "MixingWeights",
    "ResolutionSource",
    "SampleRecord",
    "ShardIterator",
    "SourceProtocol",
    "WDSShardReaderNode",
    "WDSSource",
]
