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

[FIX-SAMPLE-RECORD-EXPORT] SampleRecord est défini dans ``augmentation.py``
(c'est le contrat entre les sources et le stage de filtrage/augmentation).
Il est ré-exporté ici pour la commodité des consommateurs du package
``sources``, mais l'import canonique reste ``dino_loader.augmentation``.
``ShardIterator`` est une implémentation interne de ``hpc_source`` et ne
devrait pas être importé directement depuis l'extérieur du package sources —
seul ``MixingSource`` fait partie de l'API publique.

Exports publics
---------------
    from dino_loader.sources import (
        SourceProtocol,
        MixingWeights,
        ResolutionSource,
        SampleRecord,       # canonical: dino_loader.augmentation.SampleRecord
        MixingSource,
        WDSSource,
        WDSShardReaderNode,
    )
"""

# [FIX-SAMPLE-RECORD-EXPORT] SampleRecord lives in augmentation.py.
# Re-exported here for backwards-compatibility and convenience, but the
# canonical import path is dino_loader.augmentation.SampleRecord.
from dino_loader.augmentation import SampleRecord
from dino_loader.sources._weights import MixingWeights
from dino_loader.sources.hpc_source import MixingSource, ShardIterator
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
