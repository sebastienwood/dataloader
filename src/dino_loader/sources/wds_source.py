"""dino_loader.sources.wds_source
=================================
Source de données basée sur ``webdataset``, compatible avec l'interface de
``MixingSource``.

Motivation
----------
``MixingSource`` (``hpc_source.py``) est optimisée pour les clusters HPC avec
Lustre lent et beaucoup de rangs par nœud : elle maintient un cache ``/dev/shm``
et un double-buffering custom.  Cette complexité est justifiée quand 70+ rangs
lisent depuis ``/dev/shm`` plutôt que depuis Lustre.

``WDSSource`` est une alternative **plus simple** qui délègue le cycling, le
shuffle et le mixing à ``webdataset``, en ajoutant uniquement le suivi des
indices de dataset via ``IndexedRandomMixDataset`` (``_wds_mix.py``).

Quand l'utiliser ?
------------------
- Shards déjà en mémoire rapide (NVMe local, Lustre MDS rapide).
- Peu de rangs par nœud (≤ 8).
- Prototypage ou expériences où la simplicité prime.
- Intégration avec ``torchdata.StatefulDataLoader`` via l'API WDS standard.

Limites
-------
- Pas de cache ``/dev/shm`` : chaque rang lit directement depuis le filesystem.
- ``set_weights()`` recrée l'itérateur (coût ~ms).

Compatibilité avec ``ShardReaderNode``
---------------------------------------
``WDSShardReaderNode`` expose la même interface que ``ShardReaderNode`` et peut
être utilisé comme drop-in replacement dans ``_ReaderAdapter`` (``loader.py``).
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

from dino_loader.sources._wds_mix import IndexedRandomMixDataset
from dino_loader.sources._weights import MixingWeights

log = logging.getLogger(__name__)

_DEFAULT_SHUFFLE_BUFFER = 1000


def _require_webdataset() -> None:
    """Lève ``ImportError`` si ``webdataset`` n'est pas installé."""
    if not HAS_WDS:
        msg = (
            "WDSSource requires the 'webdataset' library. "
            "Install with: pip install webdataset"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# WDSSource
# ---------------------------------------------------------------------------


class WDSSource:
    """Source de données basée sur ``webdataset``, compatible ``MixingSource``.

    Args:
        specs:          Liste ordonnée de spécifications de datasets.
        batch_size:     Nombre de samples par batch.
        rank:           Rang global de ce processus (partitionnement des shards).
        world_size:     Nombre total de rangs.
        seed:           Graine RNG de base.
        shuffle_buffer: Taille du buffer de shuffle WDS en nombre de samples.
        num_workers:    Nombre de workers WDS (0 = thread principal).
        url_handler:    Handler URL personnalisé (ex. : client objet-store S3,
            proxy vers le cache ``/dev/shm``). Si ``None``, lecture directe.

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
        self._last_metadata:   list[dict | None] = []
        self._last_ds_indices: list[int] = []

        # Pipeline WDS créé paresseusement au premier appel à __call__().
        self._iterator: Any | None  = None
        self._iter_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction du pipeline
    # ------------------------------------------------------------------

    def _build_single_pipeline(
        self, spec: DatasetSpec, epoch_seed: int, ds_index: int,
    ) -> Any:
        """Construit un pipeline WDS pour un seul dataset.

        Args:
            spec:       Spécification du dataset.
            epoch_seed: Seed de base pour cette époque.
            ds_index:   Index du dataset dans ``self._specs`` (pour le seed).

        Returns:
            Pipeline WDS (``webdataset.WebDataset``).

        """
        assigned_shards = [
            s for j, s in enumerate(spec.shards)
            if j % self._world_size == self._rank
        ]
        if not assigned_shards:
            log.warning(
                "WDSSource: dataset '%s' has no shards assigned to rank %d/%d.",
                spec.name, self._rank, self._world_size,
            )
            return None

        shard_seed = epoch_seed + ds_index * 31
        if spec.shard_sampling == "resampled":
            shard_source = wds.ResampledShards(
                urls          = assigned_shards,
                seed          = shard_seed,
                deterministic = True,
            )
        else:
            shuffled = list(assigned_shards)
            rng = np.random.default_rng(shard_seed)
            rng.shuffle(shuffled)
            shard_source = wds.SimpleShardList(shuffled)

        pipe_kwargs: dict[str, Any] = {}
        if self._url_handler is not None:
            pipe_kwargs["handler"] = self._url_handler

        pipeline = (
            wds.WebDataset(shard_source, **pipe_kwargs)
            .shuffle(self._shuffle_buffer, seed=epoch_seed + ds_index * 13)
            .decode("pil")
            .to_tuple("jpg;jpeg;png", "json", handler=wds.warn_and_continue)
        )

        if spec.min_sample_quality is not None:
            min_quality = spec.min_sample_quality
            pipeline = pipeline.select(
                lambda sample, mq=min_quality: (
                    sample[1].get("quality_score", 1.0) >= mq
                    if sample[1] is not None else True
                ),
            )

        return pipeline

    def _build_pipeline(self, epoch: int) -> Any:
        """Construit le pipeline de mixing pour l'époque donnée.

        Le seed est dérivé de l'époque pour garantir la reproductibilité :
        même ``epoch`` + même ``seed`` → même ordre de shards et de samples.

        Avec un seul dataset, retourne son itérateur directement en yielding
        ``(sample, 0)``.  Avec plusieurs datasets, utilise
        ``IndexedRandomMixDataset`` qui expose l'indice de dataset source.

        Args:
            epoch: Numéro d'époque (utilisé pour dériver le seed).

        Returns:
            Itérateur de ``(sample, ds_index)``.

        Raises:
            RuntimeError: Si aucun shard n'est assigné à ce rang.

        """
        epoch_seed = self._seed + epoch * 997 + self._rank
        weights    = self._weights.get()

        pipelines: list[Any] = []
        active_weights: list[float] = []

        for i, spec in enumerate(self._specs):
            pipe = self._build_single_pipeline(spec, epoch_seed, ds_index=i)
            if pipe is not None:
                pipelines.append(pipe)
                active_weights.append(weights[i])

        if not pipelines:
            msg = (
                f"WDSSource: no pipelines built for rank {self._rank}/{self._world_size}. "
                "All datasets have no shards assigned to this rank."
            )
            raise RuntimeError(msg)

        if len(pipelines) == 1:
            # Un seul dataset : wrap minimal pour exposer l'indice 0.
            return ((sample, 0) for sample in pipelines[0])

        mix = IndexedRandomMixDataset(
            datasets = pipelines,
            probs    = active_weights,
            seed     = epoch_seed,
        )
        return iter(mix)

    def _get_or_build_iterator(self) -> Any:
        """Retourne l'itérateur courant, le construisant si nécessaire."""
        with self._iter_lock:
            if self._iterator is None:
                self._iterator = self._build_pipeline(self._epoch)
            return self._iterator

    # ------------------------------------------------------------------
    # Interface MixingSource
    # ------------------------------------------------------------------

    def __call__(self) -> list[np.ndarray]:
        """Retourne un batch de tableaux numpy de bytes JPEG.

        Returns:
            Liste de ``batch_size`` tableaux numpy de dtype ``uint8``.

        """
        import io  # noqa: PLC0415

        it = self._get_or_build_iterator()

        jpegs:      list[np.ndarray]  = []
        metadata:   list[dict | None] = []
        ds_indices: list[int]         = []

        while len(jpegs) < self._batch_size:
            try:
                sample, ds_idx = next(it)
            except StopIteration:
                with self._iter_lock:
                    self._iterator = self._build_pipeline(self._epoch)
                    it = self._iterator
                sample, ds_idx = next(it)

            img, meta = sample

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            jpegs.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
            metadata.append(meta if isinstance(meta, dict) else None)
            ds_indices.append(ds_idx)

        with self._lock:
            self._last_metadata   = metadata
            self._last_ds_indices = ds_indices

        for cb in self._ds_index_callbacks:
            cb(ds_indices)

        return jpegs

    def pop_last_metadata(self) -> list[dict | None]:
        """Retourne les métadonnées du dernier :meth:`__call__`. Thread-safe.

        Returns:
            Copie de la liste des métadonnées du dernier batch.

        """
        with self._lock:
            return list(self._last_metadata)

    def register_dataset_index_callback(
        self,
        cb: Callable[[list[int]], None],
    ) -> None:
        """Enregistre un callback recevant les indices de dataset par sample.

        Les indices sont exacts même avec plusieurs datasets : ils reflètent
        le pipeline source réel dans ``IndexedRandomMixDataset``.

        Args:
            cb: Callable appelé avec ``list[int]`` après chaque batch.

        """
        self._ds_index_callbacks.append(cb)

    def set_epoch(self, epoch: int) -> None:
        """Recrée le pipeline pour une nouvelle époque.

        Args:
            epoch: Numéro de la nouvelle époque.

        """
        with self._iter_lock:
            self._epoch    = epoch
            self._iterator = None
        log.debug("WDSSource: epoch set to %d, pipeline will be rebuilt.", epoch)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Met à jour les poids de mixage et recrée le pipeline.

        Args:
            weights: Nouveaux poids (re-normalisés automatiquement).

        """
        self._weights.set(weights)
        with self._iter_lock:
            self._iterator = None

    def set_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids d'un dataset par son nom.

        Args:
            name:   Nom du dataset.
            weight: Nouveau poids brut.

        """
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
        """Libère les ressources (no-op — pas de threads à arrêter)."""
        with self._iter_lock:
            self._iterator = None


# ---------------------------------------------------------------------------
# WDSShardReaderNode
# ---------------------------------------------------------------------------


class WDSShardReaderNode:
    """Nœud torchdata minimal wrappant un ``WDSSource``.

    Expose la même interface que ``ShardReaderNode`` pour être utilisé comme
    drop-in replacement dans ``_ReaderAdapter`` (``loader.py``).

    Args:
        source: Instance ``WDSSource`` à wrapper.

    """

    def __init__(self, source: WDSSource) -> None:
        """Initialise le nœud."""
        self._source = source
        self._epoch  = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Remet la source à zéro pour une nouvelle époque.

        Args:
            initial_state: État persisté optionnel (issu de :meth:`get_state`).

        """
        if initial_state is not None:
            self._epoch = initial_state.get("epoch", 0)
        self._source.set_epoch(self._epoch)

    def next(self) -> tuple[list[np.ndarray], list[dict | None]]:
        """Retourne un batch ``(jpegs, metadata)``.

        Returns:
            Tuple ``(liste de np.ndarray uint8, liste de dict | None)``.

        """
        jpegs    = self._source()
        metadata = self._source.pop_last_metadata()
        return jpegs, metadata

    def get_state(self) -> dict[str, Any]:
        """Retourne l'état persistable.

        Returns:
            Dict avec ``epoch``, ``mixing_weights``, ``dataset_names``.

        """
        return {
            "epoch":          self._epoch,
            "mixing_weights": self._source.current_weights,
            "dataset_names":  self._source.dataset_names,
        }

    def set_epoch(self, epoch: int) -> None:
        """Avance à la nouvelle époque.

        Args:
            epoch: Numéro d'époque.

        """
        self._epoch = epoch
        self._source.set_epoch(epoch)

    def set_weights(self, weights: list[float]) -> None:
        """Met à jour les poids de mixage.

        Args:
            weights: Nouveaux poids bruts.

        """
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids d'un dataset par son nom.

        Args:
            name:   Nom du dataset.
            weight: Nouveau poids brut.

        """
        self._source.set_by_name(name, weight)

    @property
    def current_weights(self) -> list[float]:
        """Poids courants."""
        return self._source.current_weights

    @property
    def dataset_names(self) -> list[str]:
        """Noms des datasets."""
        return self._source.dataset_names
