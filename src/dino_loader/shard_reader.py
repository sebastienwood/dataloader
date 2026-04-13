"""dino_loader.shard_reader
===========================
``ShardReaderNode`` — nœud ``torchdata.nodes`` encapsulant les stages 1-2
du pipeline dino_loader (I/O shards + mixing multi-dataset).

``_ReaderAdapter`` — bridge entre ``ShardReaderNode`` et le callable attendu
par les backends (DALI ExternalSource ou pipeline CPU).

Pourquoi ce module est séparé
------------------------------
``ShardReaderNode`` est une primitive d'I/O qui ne dépend ni de DALI ni de
l'assemblage des batches.  Il peut être utilisé de façon autonome pour
alimenter n'importe quel backend d'augmentation, ou dans des scripts de
débogage et benchmarks sans instancier le loader complet.

Le séparer de ``pipeline_graph.py`` (transforms) évite un couplage inutile
et clarifie les responsabilités :

- ``shard_reader`` → stages 1-2 : lecture des shards, mixing, prédication.
- ``pipeline_graph`` → stages post-DALI : Batch assembly, masking, filtering.

``_ReaderAdapter``
-------------------
Adaptateur exposant ``ShardReaderNode`` comme callable source DALI/CPU.
Il vit dans ce module car il n'a aucune dépendance sur ``DINODataLoader``
et doit être accessible sans importer ``loader.py``.

[FIX-META-FIFO] ``_meta_queue`` est une ``queue.Queue`` (FIFO) garantissant
l'alignement strict entre les batches DALI et leurs métadonnées, même
quand DALI appelle ``__call__()`` de façon préemptive depuis son thread
de prefetch.

Public API
----------
::

    from dino_loader.shard_reader import ShardReaderNode, _ReaderAdapter, build_reader_graph

    cache  = InProcessShardCache(max_gb=1.0)
    loader, reader = build_reader_graph(
        specs=[spec], batch_size=4, cache=cache, rank=0, world_size=1,
    )
    reader.set_epoch(0)
    for jpegs, meta in loader:
        my_augment(jpegs)

Note sur reset()
-----------------
``ShardReaderNode.reset()`` ne recrée pas ``MixingSource`` à chaque appel.
``tn.Loader(restart_on_stop_iteration=True)`` appelle ``reset()`` à chaque
époque ; recréer tous les threads ``ShardIterator`` serait coûteux et inutile.
``reset()`` appelle ``set_epoch()`` sur la source existante. La source n'est
créée qu'une seule fois à la première construction.
"""

from __future__ import annotations

import logging
import queue
from typing import Any

import numpy as np
import torchdata.nodes as tn
from dino_datasets import DatasetSpec
from torchdata.nodes import BaseNode

from dino_loader.augmentation import SamplePredicate
from dino_loader.config import SharedExtractionPoolConfig
from dino_loader.sources.hpc_source import MixingSource
from dino_loader.sources.protocol import SourceProtocol
from dino_loader.sources.resolution import ResolutionSource

log = logging.getLogger(__name__)

# Type alias : un batch brut retourné par ShardReaderNode.
ReaderBatch = tuple[list[np.ndarray], list[dict[str, Any] | None]]


class ShardReaderNode(BaseNode):  # type: ignore[misc]
    """torchdata BaseNode encapsulant les stages 1-2 du pipeline dino_loader.

    Stage 1 — ``NodeSharedShardCache`` / ``InProcessShardCache`` :
        Lit les bytes de shard depuis Lustre (rank 0) ou /dev/shm (autres rangs).

    Stage 2 — ``SourceProtocol`` (``MixingSource`` par défaut) :
        Mixing multi-dataset pondéré avec filtrage qualité par shard.

    Chaque appel à ``next()`` retourne un ``ReaderBatch`` :
    ``(list[np.ndarray], list[dict | None])`` — bytes JPEG bruts et métadonnées
    JSON optionnelles, de longueur ``batch_size``.

    La source peut être n'importe quel objet conforme à ``SourceProtocol``.
    Par défaut, ``MixingSource`` (HPC, cache /dev/shm) est utilisée.  Pour
    passer ``WDSSource`` ou une source custom, utiliser le paramètre ``source``.

    Args:
        specs:               Liste ordonnée de spécifications de datasets.
        batch_size:          Nombre de samples par batch.
        cache:               Instance de cache de shards.
        rank:                Rang global de ce processus.
        world_size:          Nombre total de rangs.
        source:              Source custom conforme à ``SourceProtocol``.
                             Si ``None``, ``MixingSource`` est instanciée.
        pool_cfg:            Config du pool d'extraction partagé.
        seed:                Graine RNG de base.
        device_id:           Index GPU local.
        shuffle_buffer_size: Profondeur du réservoir de shuffle par shard.
        debug_log_keys:      Chemin optionnel vers le log d'audit de clés.
        sample_predicate:    Filtre anticipé appliqué avant le décodage DALI.

    """

    _KEY_EPOCH   = "epoch"
    _KEY_WEIGHTS = "mixing_weights"
    _KEY_NAMES   = "dataset_names"

    def __init__(
        self,
        specs:               list[DatasetSpec],
        batch_size:          int,
        cache:               Any,
        rank:                int,
        world_size:          int,
        *,
        source:              SourceProtocol | None      = None,
        pool_cfg:            SharedExtractionPoolConfig | None = None,
        seed:                int                        = 0,
        device_id:           int                        = 0,
        shuffle_buffer_size: int                        = 512,
        debug_log_keys:      str | None                = None,
        sample_predicate:    SamplePredicate | None    = None,
    ) -> None:
        super().__init__()

        self._specs         = specs
        self._batch_size    = batch_size
        self._cache         = cache
        self._rank          = rank
        self._world_size    = world_size
        self._pool_cfg      = pool_cfg
        self._seed          = seed
        self._device_id     = device_id
        self._shuffle_buf   = shuffle_buffer_size
        self._debug_log     = debug_log_keys
        self._predicate     = sample_predicate

        self._epoch: int = 0

        # Source injectée ou créée paresseusement au premier reset().
        self._source: SourceProtocol | None = source

    # ------------------------------------------------------------------
    # BaseNode protocol
    # ------------------------------------------------------------------

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Prépare la source pour une nouvelle époque.

        La source ``MixingSource`` n'est créée qu'à la première invocation,
        évitant de recréer tous les threads d'extraction à chaque époque.
        Pour une source injectée, ``set_epoch()`` est simplement appelé.

        Args:
            initial_state: État optionnel issu d'un appel précédent à
                :meth:`get_state`.

        """
        super().reset(initial_state)

        if self._source is None:
            self._source = MixingSource(
                specs               = self._specs,
                batch_size          = self._batch_size,
                cache               = self._cache,
                rank                = self._rank,
                world_size          = self._world_size,
                pool_cfg            = self._pool_cfg,
                seed                = self._seed,
                device_id           = self._device_id,
                shuffle_buffer_size = self._shuffle_buf,
                debug_log_keys      = self._debug_log,
                sample_predicate    = self._predicate,
            )
            log.debug(
                "ShardReaderNode: source créée, rank=%d/%d datasets=%s",
                self._rank, self._world_size,
                [s.name for s in self._specs],
            )

        if initial_state is not None:
            self._restore_state(initial_state)
        else:
            self._source.set_epoch(self._epoch)
            log.debug("ShardReaderNode.reset: set_epoch(%d)", self._epoch)

    def next(self) -> ReaderBatch:
        """Retourne le prochain batch sous la forme (jpegs, metadata).

        Raises:
            AssertionError: Si appelé avant :meth:`reset`.

        """
        assert self._source is not None, "reset() must be called before next()"
        jpegs = self._source()
        meta  = self._source.pop_last_metadata()
        return jpegs, meta

    def get_state(self) -> dict[str, Any]:
        """Persiste l'époque et les poids (la position intra-époque n'est pas sauvegardée)."""
        weights = self._source.current_weights if self._source is not None else []
        names   = self._source.dataset_names   if self._source is not None else []
        return {
            self._KEY_EPOCH:   self._epoch,
            self._KEY_WEIGHTS: weights,
            self._KEY_NAMES:   names,
        }

    # ------------------------------------------------------------------
    # API de contrôle
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Avance à une nouvelle époque (re-shuffle l'ordre des shards)."""
        self._epoch = epoch
        if self._source is not None:
            self._source.set_epoch(epoch)

    def set_weights(self, weights: list[float]) -> None:
        """Met à jour les poids de mixage (re-normalisés automatiquement)."""
        if self._source is not None:
            self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids d'un dataset par son nom."""
        if self._source is not None:
            self._source.set_by_name(name, weight)

    @property
    def current_weights(self) -> list[float]:
        """Poids de mixage normalisés courants."""
        if self._source is None:
            return []
        return self._source.current_weights

    @property
    def dataset_names(self) -> list[str]:
        """Liste ordonnée des noms de datasets."""
        if self._source is None:
            return [s.name for s in self._specs]
        return self._source.dataset_names

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _restore_state(self, state: dict[str, Any]) -> None:
        """Applique un state dict précédemment persisté."""
        assert self._source is not None

        saved_names: list[str] = state.get(self._KEY_NAMES, [])
        if saved_names and saved_names != self._source.dataset_names:
            log.warning(
                "ShardReaderNode: les noms de datasets du checkpoint %s "
                "diffèrent des specs courantes %s — restauration des poids ignorée.",
                saved_names, self._source.dataset_names,
            )
        else:
            saved_weights: list[float] = state.get(self._KEY_WEIGHTS, [])
            if saved_weights:
                self._source.set_weights(saved_weights)

        self._epoch = state.get(self._KEY_EPOCH, 0)
        self._source.set_epoch(self._epoch)

    def __del__(self) -> None:
        """Libère la source à la destruction (best-effort)."""
        source = getattr(self, "_source", None)
        if source is not None:
            try:
                source.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# _ReaderAdapter — bridge ShardReaderNode → callable DALI/CPU ExternalSource
# ---------------------------------------------------------------------------


class _ReaderAdapter:
    """Adaptateur exposant ``ShardReaderNode`` comme callable source DALI/CPU.

    [FIX-META-FIFO] ``_meta_queue`` est une ``queue.Queue`` (FIFO) garantissant
    l'alignement strict entre les batches DALI et leurs métadonnées, même
    quand DALI appelle ``__call__()`` de façon préemptive depuis son thread
    de prefetch.

    Attributs de convention
    -----------------------
    ``_batch_size`` et ``_resolution_src`` sont lus par les backends via
    ``getattr`` pour inférer batch_size et la source de résolution.

    """

    _META_QUEUE_MAXSIZE: int = 32

    def __init__(
        self,
        reader:         ShardReaderNode,
        resolution_src: ResolutionSource,
        batch_size:     int,
    ) -> None:
        self._reader         = reader
        self._resolution_src = resolution_src
        self._batch_size     = batch_size
        self._meta_queue: queue.Queue[list[dict | None]] = queue.Queue(
            maxsize=self._META_QUEUE_MAXSIZE,
        )

    def __call__(self) -> list:
        """Retourne un batch de tableaux JPEG (appelé par DALI à chaque step)."""
        jpegs, metadata = self._reader.next()
        try:
            self._meta_queue.put_nowait(metadata)
        except queue.Full:
            log.warning(
                "_ReaderAdapter: queue de métadonnées pleine (%d slots). "
                "Augmenter _META_QUEUE_MAXSIZE ou réduire les profondeurs de queue DALI.",
                self._META_QUEUE_MAXSIZE,
            )
            try:
                self._meta_queue.get_nowait()
            except queue.Empty:
                pass
            self._meta_queue.put_nowait(metadata)
        return jpegs

    def pop_last_metadata(self) -> list[dict | None]:
        """Retourne les métadonnées du batch le plus ancien (FIFO)."""
        try:
            return self._meta_queue.get_nowait()
        except queue.Empty:
            log.debug("_ReaderAdapter.pop_last_metadata: queue vide, retour []")
            return []

    def register_dataset_index_callback(self, cb: Any) -> None:
        """Propage les callbacks NormSource vers la MixingSource interne."""
        source = self._reader._source  # type: ignore[attr-defined]
        if source is not None and hasattr(source, "register_dataset_index_callback"):
            source.register_dataset_index_callback(cb)
        else:
            log.warning(
                "_ReaderAdapter: register_dataset_index_callback appelé avant "
                "ShardReaderNode.reset() — callback non enregistré.",
            )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_reader_graph(
    specs:               list[DatasetSpec],
    batch_size:          int,
    cache:               Any,
    rank:                int,
    world_size:          int,
    *,
    source:              SourceProtocol | None      = None,
    pool_cfg:            SharedExtractionPoolConfig | None = None,
    seed:                int                        = 0,
    device_id:           int                        = 0,
    shuffle_buffer_size: int                        = 512,
    debug_log_keys:      str | None                = None,
    sample_predicate:    SamplePredicate | None    = None,
    prefetch_factor:     int                        = 2,
) -> tuple[tn.Loader, ShardReaderNode]:
    """Construit un ``tn.Loader`` prêt à l'emploi sur le pipeline de lecture.

    Args:
        specs:               Spécifications des datasets.
        batch_size:          Samples par batch.
        cache:               Cache de shards.
        rank:                Rang global.
        world_size:          Nombre total de rangs.
        source:              Source custom conforme à ``SourceProtocol``.
                             Si ``None``, ``MixingSource`` est utilisée.
        pool_cfg:            Configuration du pool d'extraction partagé.
        seed:                Graine RNG de base.
        device_id:           Index GPU.
        shuffle_buffer_size: Profondeur du réservoir par shard.
        debug_log_keys:      Chemin optionnel vers le log d'audit.
        sample_predicate:    Filtre anticipé optionnel.
        prefetch_factor:     Profondeur de look-ahead du ``tn.Prefetcher``.

    Returns:
        ``(loader, reader_node)`` — itérer sur ``loader``, appeler
        ``reader_node.set_epoch(e)`` à chaque époque.

    """
    reader = ShardReaderNode(
        specs               = specs,
        batch_size          = batch_size,
        cache               = cache,
        rank                = rank,
        world_size          = world_size,
        source              = source,
        pool_cfg            = pool_cfg,
        seed                = seed,
        device_id           = device_id,
        shuffle_buffer_size = shuffle_buffer_size,
        debug_log_keys      = debug_log_keys,
        sample_predicate    = sample_predicate,
    )

    prefetched = tn.Prefetcher(reader, prefetch_factor=prefetch_factor)
    loader     = tn.Loader(prefetched, restart_on_stop_iteration=True)

    return loader, reader
