"""dino_loader.wds_source
=======================
Alternative à ``MixingSource`` basée sur la bibliothèque ``webdataset``.

Motivation
----------
``MixingSource`` + ``ShardIterator`` est une implémentation custom du cycling
de shards WebDataset.  Elle est optimale pour les clusters HPC grâce au cache
``/dev/shm`` et au double-buffering strict — mais elle est aussi complexe et
introduit plusieurs points de défaillance (threads, queues, prefetch asynchrone).

``WDSSource`` est une alternative **plus simple et plus fiable** qui délègue
entièrement la gestion des shards à la bibliothèque ``webdataset`` :

- Cycling, shuffle, resampling → ``wds.ResampledShards`` / ``wds.SimpleShardList``
- Parsing du tar → ``wds.tarfile_to_samples``
- Mixing multi-dataset → ``wds.RandomMix``
- Filtrage qualité → ``wds.select``
- Buffer shuffle → ``wds.shuffle``

Ce que ``WDSSource`` ne fait PAS (et délègue à la couche supérieure) :
- Cache ``/dev/shm`` : optionnel, peut être branché via un ``url_handler``
  personnalisé passé à webdataset.
- Augmentation DALI : toujours gérée par ``BackendProtocol.build_pipeline``.

Quand l'utiliser ?
------------------
``WDSSource`` est recommandé quand :
- Les shards sont déjà en mémoire rapide (NVMe local, Lustre MDS rapide).
- La simplicité prime sur la latence absolue.
- On veut utiliser les primitives webdataset standard (``wds.DataPipeline``,
  ``torchdata.StatefulDataLoader``, etc.) sans réimplémenter leur logique.

``MixingSource`` reste préférable pour les gros clusters HPC avec Lustre lent
et beaucoup de rangs par nœud (le cache ``/dev/shm`` y réduit les I/O Lustre
par un facteur ``local_world_size``).

Compatibilité avec le reste du loader
--------------------------------------
``WDSSource`` expose exactement la même interface que ``MixingSource`` :
- ``__call__() → list[np.ndarray]``  (bytes JPEG bruts)
- ``pop_last_metadata() → list[dict | None]``
- ``set_epoch(epoch)``
- ``set_weights(weights)`` / ``set_by_name(name, weight)``
- ``register_dataset_index_callback(cb)``
- ``current_weights`` / ``dataset_names`` properties
- ``close()``

Il peut donc être passé directement à ``_ReaderAdapter`` dans ``loader.py``.

Exemple d'utilisation
---------------------
::

    from dino_loader.wds_source import WDSSource

    source = WDSSource(
        specs      = specs,
        batch_size = 512,
        rank       = env.rank,
        world_size = env.world_size,
        seed       = cfg.seed,
    )

    # Brancher sur le loader via un ShardReaderNode custom :
    from dino_loader.nodes import ShardReaderNode
    reader = ShardReaderNode(specs=specs, ..., source_factory=source)

    # Ou directement dans _ReaderAdapter (usage avancé) :
    adapter = _ReaderAdapter(reader=..., ...)

Notes d'implémentation
----------------------
- La gestion du seed WebDataset utilise ``worker_seed`` pour garantir que chaque
  rang worker voit des shards différents tout en ayant un shuffle reproductible.
- Les pipelines WDS sont créés paresseusement dans ``__call__()`` / ``__iter__()``
  pour éviter la construction de plusieurs itérateurs avant le premier appel.
- ``set_epoch()`` recrée le pipeline avec un seed différent.  WDS ne supporte
  pas de reset en place sans recréation de l'itérateur.
- ``set_weights()`` recrée aussi le pipeline car ``wds.RandomMix`` ne supporte
  pas la mise à jour dynamique des poids.
"""

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

try:
    import webdataset as wds
    HAS_WDS = True
except ImportError:
    HAS_WDS = False

from dino_datasets import DatasetSpec

from dino_loader.mixing_source import MixingWeights

log = logging.getLogger(__name__)

# Taille du buffer de shuffle WDS par défaut.
# Réduire sur les nœuds avec peu de RAM, augmenter pour un meilleur mélange.
_DEFAULT_SHUFFLE_BUFFER = 1000


def _require_webdataset() -> None:
    """Lève ImportError si webdataset n'est pas installé."""
    if not HAS_WDS:
        msg = (
            "WDSSource requires the 'webdataset' library. "
            "Install with: pip install webdataset"
        )
        raise ImportError(msg)


class WDSSource:
    """Source de données basée sur webdataset, compatible avec l'interface MixingSource.

    Remplace ``MixingSource`` par une implémentation plus simple qui s'appuie
    entièrement sur ``webdataset`` pour le cycling, le shuffle et le mixing.

    Avantages par rapport à MixingSource
    -------------------------------------
    - Moins de code custom (~200 lignes vs ~500 lignes).
    - Pas de threads manuels ni de queues à gérer.
    - Compatible avec ``wds.DataPipeline`` et les primitives webdataset standard.
    - Le shuffle ``wds.shuffle`` est implémenté en réservoir, identique à celui
      de ``ShardIterator`` mais sans gestion manuelle.
    - ``wds.ResampledShards`` gère le resampling infini nativement.

    Limitations
    -----------
    - Pas de cache ``/dev/shm`` : chaque rang lit directement depuis Lustre.
      Sur les gros clusters (72+ rangs/nœud), ``MixingSource`` avec
      ``NodeSharedShardCache`` reste préférable.
    - ``set_weights()`` recrée l'itérateur (coût ~ms).
    - La latence de la première itération après ``set_epoch()`` est légèrement
      plus élevée (recréation du pipeline WDS).

    Args:
        specs: Liste ordonnée de spécifications de datasets.
        batch_size: Nombre de samples par batch.
        rank: Rang global de ce processus (pour le partitionnement des shards).
        world_size: Nombre total de rangs.
        seed: Graine RNG de base.
        shuffle_buffer: Taille du buffer de shuffle en nombre de samples.
        num_workers: Nombre de workers WDS pour la lecture des shards.
            0 = lecture dans le thread principal (recommandé pour les usages
            embarqués dans un ThreadPoolExecutor externe).
        url_handler: Handler URL personnalisé pour webdataset.
            Utile pour brancher un cache /dev/shm ou un client objet-store S3.
            Si None, utilise le handler par défaut (lecture directe depuis le FS).

    """

    def __init__(
        self,
        specs:          list[DatasetSpec],
        batch_size:     int,
        rank:           int,
        world_size:     int,
        seed:           int   = 0,
        shuffle_buffer: int   = _DEFAULT_SHUFFLE_BUFFER,
        num_workers:    int   = 0,
        url_handler:    Callable[[str], Any] | None = None,
    ) -> None:
        """Initialise WDSSource."""
        _require_webdataset()

        self._specs          = specs
        self._batch_size     = batch_size
        self._rank           = rank
        self._world_size     = world_size
        self._seed           = seed
        self._shuffle_buffer = shuffle_buffer
        self._num_workers    = num_workers
        self._url_handler    = url_handler

        self._weights        = MixingWeights.from_specs(specs)
        self._epoch          = 0
        self._lock           = threading.Lock()

        self._ds_index_callbacks: list[Callable[[list[int]], None]] = []
        self._last_metadata: list[dict | None] = []
        self._last_ds_indices: list[int] = []

        # Pipeline WDS — créé paresseusement au premier appel à __call__().
        self._iterator: Any | None  = None
        self._iter_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self, epoch: int) -> Any:
        """Construit un pipeline WDS pour l'époque donnée.

        Le seed est dérivé de l'époque pour garantir la reproductibilité :
        même epoch + même seed → même ordre de shards et de samples.

        Args:
            epoch: Numéro d'époque (utilisé pour dériver le seed).

        Returns:
            Un itérateur WebDataset prêt à être consommé.

        """
        epoch_seed = self._seed + epoch * 997 + self._rank

        weights = self._weights.get()

        # Construire un pipeline par dataset, puis les mixer.
        pipelines: list[Any] = []
        for i, spec in enumerate(self._specs):
            # Partitionner les shards entre les rangs.
            assigned_shards = [
                s for j, s in enumerate(spec.shards)
                if j % self._world_size == self._rank
            ]
            if not assigned_shards:
                log.warning(
                    "WDSSource: dataset '%s' has no shards assigned to rank %d/%d.",
                    spec.name, self._rank, self._world_size,
                )
                continue

            if spec.shard_sampling == "resampled":
                shard_source = wds.ResampledShards(
                    urls          = assigned_shards,
                    seed          = epoch_seed + i * 31,
                    deterministic = True,
                )
            else:
                # Mode "epoch" : shuffle déterministe des shards à chaque époque.
                shuffled = list(assigned_shards)
                rng = np.random.default_rng(epoch_seed + i * 31)
                rng.shuffle(shuffled)
                shard_source = wds.SimpleShardList(shuffled)

            # Pipeline pour ce dataset : ouvrir les tars, extraire jpg + json.
            pipe_kwargs: dict[str, Any] = {}
            if self._url_handler is not None:
                pipe_kwargs["handler"] = self._url_handler

            pipeline = (
                wds.WebDataset(shard_source, **pipe_kwargs)
                .shuffle(self._shuffle_buffer, seed=epoch_seed + i * 13)
                .decode("pil")
                .to_tuple("jpg;jpeg;png", "json", handler=wds.warn_and_continue)
            )

            # Filtre qualité si demandé.
            min_quality = spec.min_sample_quality
            if min_quality is not None:
                pipeline = pipeline.select(
                    lambda sample, mq=min_quality: (
                        sample[1].get("quality_score", 1.0) >= mq
                        if sample[1] is not None else True
                    )
                )

            pipelines.append(pipeline)

        if not pipelines:
            msg = (
                f"WDSSource: no pipelines built for rank {self._rank}/{self._world_size}. "
                "All datasets have no shards assigned to this rank."
            )
            raise RuntimeError(msg)

        if len(pipelines) == 1:
            mixed = pipelines[0]
        else:
            mixed = wds.RandomMix(pipelines, probs=weights, seed=epoch_seed)

        return iter(mixed)

    def _get_or_build_iterator(self) -> Any:
        """Retourne l'itérateur courant, le construisant si nécessaire."""
        with self._iter_lock:
            if self._iterator is None:
                self._iterator = self._build_pipeline(self._epoch)
            return self._iterator

    # ------------------------------------------------------------------
    # MixingSource-compatible interface
    # ------------------------------------------------------------------

    def __call__(self) -> list[np.ndarray]:
        """Retourne un batch de tableaux numpy de bytes JPEG.

        Compatible avec l'interface MixingSource pour être utilisé comme
        callback DALI ExternalSource.

        Returns:
            Liste de ``batch_size`` tableaux numpy de dtype uint8.

        """
        it = self._get_or_build_iterator()

        jpegs: list[np.ndarray] = []
        metadata: list[dict | None] = []
        ds_indices: list[int] = []

        while len(jpegs) < self._batch_size:
            try:
                sample = next(it)
            except StopIteration:
                # Fin d'époque : recréer l'itérateur.
                with self._iter_lock:
                    self._iterator = self._build_pipeline(self._epoch)
                    it = self._iterator
                sample = next(it)

            img, meta = sample

            # Convertir l'image PIL en bytes JPEG numpy.
            import io  # noqa: PLC0415
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            jpeg_bytes = buf.getvalue()
            jpegs.append(np.frombuffer(jpeg_bytes, dtype=np.uint8))

            metadata.append(meta if isinstance(meta, dict) else None)
            # Pour les datasets mono-source, ds_index = 0.
            # Pour RandomMix, on ne peut pas récupérer l'index directement
            # depuis l'API WDS publique — on utilise 0 comme approximation.
            ds_indices.append(0)

        with self._lock:
            self._last_metadata   = metadata
            self._last_ds_indices = ds_indices

        if self._ds_index_callbacks:
            for cb in self._ds_index_callbacks:
                cb(ds_indices)

        return jpegs

    def pop_last_metadata(self) -> list[dict | None]:
        """Retourne les métadonnées du dernier __call__. Thread-safe."""
        with self._lock:
            return list(self._last_metadata)

    def register_dataset_index_callback(
        self,
        cb: Callable[[list[int]], None],
    ) -> None:
        """Enregistre un callback qui reçoit les indices de dataset par sample.

        Note: avec ``wds.RandomMix``, les indices sont approximatifs (toujours 0).
        Les per-dataset dataset indices exacts ne sont pas exposés par l'API
        publique de webdataset.  Pour une normalisation per-dataset précise,
        utiliser ``MixingSource`` qui maintient des iterateurs séparés.
        """
        self._ds_index_callbacks.append(cb)
        if len(self._specs) > 1:
            log.warning(
                "WDSSource: register_dataset_index_callback with multiple datasets "
                "— dataset indices will be approximate (always 0). "
                "For accurate per-dataset normalization, use MixingSource instead."
            )

    def set_epoch(self, epoch: int) -> None:
        """Recrée le pipeline pour une nouvelle époque.

        Args:
            epoch: Numéro de la nouvelle époque.

        """
        with self._iter_lock:
            self._epoch    = epoch
            self._iterator = None  # Sera recréé au prochain appel.
        log.debug("WDSSource: epoch set to %d, pipeline will be rebuilt.", epoch)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Met à jour les poids de mixage.

        Recrée le pipeline car ``wds.RandomMix`` ne supporte pas la mise à
        jour dynamique des poids.

        Args:
            weights: Nouveaux poids (re-normalisés automatiquement).

        """
        self._weights.set(weights)
        with self._iter_lock:
            self._iterator = None
        log.debug("WDSSource: weights updated, pipeline will be rebuilt.")

    def set_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids d'un dataset par son nom."""
        self._weights.set_by_name(name, weight)
        with self._iter_lock:
            self._iterator = None

    @property
    def current_weights(self) -> list[float]:
        """Poids de mixage normalisés courants."""
        return self._weights.get()

    @property
    def dataset_names(self) -> list[str]:
        """Liste ordonnée des noms de datasets."""
        return list(self._weights.names)

    def close(self) -> None:
        """Libère les ressources (no-op pour WDS — pas de threads à arrêter)."""
        with self._iter_lock:
            self._iterator = None
        log.debug("WDSSource closed.")


# ---------------------------------------------------------------------------
# ShardReaderNode integration helper
# ---------------------------------------------------------------------------


def make_wds_reader_node(
    specs:          list[DatasetSpec],
    batch_size:     int,
    rank:           int,
    world_size:     int,
    seed:           int   = 0,
    shuffle_buffer: int   = _DEFAULT_SHUFFLE_BUFFER,
    url_handler:    Callable[[str], Any] | None = None,
) -> "WDSShardReaderNode":
    """Construit un nœud torchdata compatible utilisant WDSSource.

    Factory de convenance pour les utilisateurs qui veulent le graphe
    torchdata.nodes sans le cache /dev/shm.

    Args:
        specs: Spécifications des datasets.
        batch_size: Samples par batch.
        rank: Rang global.
        world_size: Taille du monde.
        seed: Graine de base.
        shuffle_buffer: Taille du buffer de shuffle WDS.
        url_handler: Handler URL optionnel.

    Returns:
        Un ``WDSShardReaderNode`` prêt à être inséré dans un graphe torchdata.

    """
    source = WDSSource(
        specs          = specs,
        batch_size     = batch_size,
        rank           = rank,
        world_size     = world_size,
        seed           = seed,
        shuffle_buffer = shuffle_buffer,
        url_handler    = url_handler,
    )
    return WDSShardReaderNode(source=source)


class WDSShardReaderNode:
    """Nœud torchdata minimal wrappant un ``WDSSource``.

    Expose la même interface que ``ShardReaderNode`` pour être utilisé
    comme drop-in replacement dans ``_ReaderAdapter``.

    Ce nœud est intentionnellement minimal : toute la logique de cycling,
    shuffle et mixing est déléguée à ``WDSSource`` / ``webdataset``.

    Args:
        source: Instance ``WDSSource`` à wraper.

    """

    def __init__(self, source: WDSSource) -> None:
        """Initialise le nœud."""
        self._source = source
        self._epoch  = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Remet le source à zéro pour une nouvelle époque."""
        if initial_state is not None:
            saved_epoch = initial_state.get("epoch", 0)
            self._epoch = saved_epoch
        self._source.set_epoch(self._epoch)

    def next(self) -> tuple[list[np.ndarray], list[dict | None]]:
        """Retourne un batch (jpegs, metadata)."""
        jpegs    = self._source()
        metadata = self._source.pop_last_metadata()
        return jpegs, metadata

    def get_state(self) -> dict[str, Any]:
        """Retourne l'état persistable."""
        return {
            "epoch":          self._epoch,
            "mixing_weights": self._source.current_weights,
            "dataset_names":  self._source.dataset_names,
        }

    def set_epoch(self, epoch: int) -> None:
        """Avance à la nouvelle époque."""
        self._epoch = epoch
        self._source.set_epoch(epoch)

    def set_weights(self, weights: list[float]) -> None:
        """Met à jour les poids de mixage."""
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids d'un dataset par son nom."""
        self._source.set_by_name(name, weight)

    @property
    def current_weights(self) -> list[float]:
        """Poids courants."""
        return self._source.current_weights

    @property
    def dataset_names(self) -> list[str]:
        """Noms des datasets."""
        return self._source.dataset_names
