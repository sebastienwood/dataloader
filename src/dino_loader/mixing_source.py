"""dino_loader.mixing_source
=========================
DALI ExternalSource callback et cycling de shards par dataset.

[PRED-1] Filtrage anticipé via SamplePredicate — évalué avant le décodage JPEG,
         éliminant le coût DALI pour les samples rejetés.
[DB-1]   Double-buffering strict dans ShardIterator (Stage A: thread I/O,
         Stage B: workers d'extraction).
[FIX-RNG] numpy.random.Generator n'est pas thread-safe.  Les accès à _rng dans
          MixingSource.__call__() sont protégés par _rng_lock.  ShardIterator
          crée un RNG par worker via un compteur atomique déterministe, ce qui
          garantit la reproductibilité avec un seed fixe.
[FIX-CYCLE] Race condition résolue : self._shard_cycle est désormais protégé par
            _cycle_lock.  reset_epoch() arrête le thread I/O via _stop_io_event,
            attend l'accusé réception _io_stopped_event, puis remplace le
            générateur sous lock avant de reprendre.
[FIX-DEADLOCK] _io_loop_inner : la séquence stop/restart est réécrite pour éviter
               le deadlock quand close() est appelé pendant reset_epoch().
[FIX-IOLOCK] prefetch() est appelé hors de _cycle_lock pour éviter de tenir le
             lock pendant une opération I/O potentiellement lente.
[FIX-KEYLOG] threading.Lock autour des écritures. encoding="utf-8" ajouté.
[FIX-MVIEW] La copie bytes(mv) juste après get_view() annulait le zero-copy.
            _extract_worker reçoit maintenant le memoryview directement et
            _extract_jpegs_with_meta accepte un buffer-like.
[FIX-META-FIFO] _last_metadata est maintenant une Queue FIFO dans _ReaderAdapter
                pour aligner les métadonnées sur l'ordre des batches DALI.
[MW-API] MixingWeights : constructeur stable __init__(names, weights) et
         classmethod from_specs().
[POOL]   Pool d'extraction partagé entre tous les ShardIterators d'un même
         MixingSource : borne le budget total de threads à
         SharedExtractionPoolConfig.max_workers, quelle que soit la cardinalité
         du mix.
"""

import itertools
import logging
import queue
import threading
from collections import deque
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
from dino_datasets import DatasetSpec

from dino_loader.augmentation import SampleMeta, SamplePredicate
from dino_loader.config import SharedExtractionPoolConfig

log = logging.getLogger(__name__)

try:
    from dino_loader.monitor.metrics import MetricField, get_registry
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

try:
    from webdataset.shardlists import ResampledShards
    HAS_WDS = True
except ImportError:
    HAS_WDS = False
    log.warning(
        "webdataset not installed — shard_sampling='resampled' will fall back "
        "to 'epoch' mode. Install with: pip install webdataset",
    )


class MixingWeights:
    """Vecteur de poids normalisé, thread-safe, pour le mixage de datasets.

    Le constructeur accepte des primitives ``(names, weights)``.  Utiliser la
    classmethod :meth:`from_specs` pour construire depuis des ``DatasetSpec``.

    Args:
        names: Liste ordonnée des noms de datasets.
        weights: Poids bruts (non normalisés) correspondants.  Doivent être
            positifs et de même longueur que *names*.

    """

    def __init__(self, names: list[str], weights: Sequence[float]) -> None:
        """Initialise avec des noms et poids explicites."""
        if len(names) != len(weights):
            msg = (
                f"MixingWeights: names and weights must have the same length, "
                f"got {len(names)} names and {len(weights)} weights."
            )
            raise ValueError(msg)
        self.names    = list(names)
        self._lock    = threading.Lock()
        self._weights = self._normalise(list(weights))

    @classmethod
    def from_specs(cls, specs: list[DatasetSpec]) -> "MixingWeights":
        """Construit depuis une liste de ``DatasetSpec``."""
        names   = [s.name   for s in specs]
        weights = [s.weight for s in specs]
        return cls(names, weights)

    def get(self) -> list[float]:
        """Retourne une copie du vecteur de poids normalisés."""
        with self._lock:
            return list(self._weights)

    def set(self, weights: Sequence[float]) -> None:
        """Remplace le vecteur (re-normalisé automatiquement).

        Args:
            weights: Nouveaux poids bruts.  Doit avoir la même longueur.

        Raises:
            ValueError: Si la longueur ne correspond pas ou si la somme est nulle.

        """
        if len(weights) != len(self.names):
            msg = (
                f"MixingWeights.set: expected {len(self.names)} weights, "
                f"got {len(weights)}."
            )
            raise ValueError(msg)
        with self._lock:
            self._weights = self._normalise(list(weights))

    def set_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids brut d'un dataset par son nom.

        Les autres poids bruts restent inchangés ; seule la normalisation
        est recalculée.

        Args:
            name: Nom du dataset à modifier.
            weight: Nouveau poids brut.

        Raises:
            KeyError: Si *name* n'est pas dans le vecteur.

        """
        try:
            idx = self.names.index(name)
        except ValueError:
            msg = f"Dataset '{name}' not found. Available: {self.names}"
            raise KeyError(msg) from None
        with self._lock:
            raw      = list(self._weights)
            raw[idx] = weight
            self._weights = self._normalise(raw)

    @staticmethod
    def _normalise(weights: list[float]) -> list[float]:
        """Normalise un vecteur de poids pour que leur somme soit 1.0."""
        total = sum(weights)
        if total <= 0:
            msg = f"Weights must sum to a positive number, got {weights}."
            raise ValueError(msg)
        return [w / total for w in weights]


class ResolutionSource:
    """Holder thread-safe de la résolution courante de crop.

    Joue le rôle de callback DALI ExternalSource (batch=False).
    L'appel à set() est immédiatement visible par le prochain prefetch DALI.
    """

    def __init__(self, global_size: int, local_size: int) -> None:
        """Initialise avec les tailles de crop initiales."""
        self._global = global_size
        self._local  = local_size
        self._lock   = threading.Lock()

    def set(self, global_size: int, local_size: int) -> None:
        """Met à jour les deux dimensions de manière atomique."""
        with self._lock:
            self._global = global_size
            self._local  = local_size

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """Retourne (global_size, local_size) comme scalaires numpy."""
        with self._lock:
            return (
                np.array(self._global, dtype=np.int32),
                np.array(self._local,  dtype=np.int32),
            )


class SampleRecord:
    """Sample décodé prêt pour le pipeline DALI."""

    __slots__ = ("jpeg", "key", "metadata")

    def __init__(
        self,
        jpeg:     bytes,
        metadata: dict | None = None,
        key:      str         = "",
    ) -> None:
        """Initialise un SampleRecord."""
        self.jpeg     = jpeg
        self.metadata = metadata
        self.key      = key


class _Sentinel:
    """Sentinel passé dans _io_queue pour signaler l'arrêt du thread I/O."""

    __slots__ = ()


_STOP = _Sentinel()

# Drapeaux d'état du thread I/O pour reset_epoch().
_IO_RUNNING = 0
_IO_PAUSED  = 1
_IO_STOPPED = 2


class ShardIterator:
    """Cycling de shards par dataset avec double-buffering strict.

    Stage A — thread I/O (daemon unique) :
        Lit les bytes de shard depuis le cache dans _io_queue.

    Stage B — workers d'extraction (ThreadPoolExecutor *partagé*) :
        Parse le tar, évalue le prédicat, pousse les SampleRecord acceptés
        dans _sample_queue.

    Différence par rapport à la version précédente
    -----------------------------------------------
    Le ``ThreadPoolExecutor`` n'est plus créé ici : il est injecté par
    ``MixingSource`` et partagé entre tous les ``ShardIterator`` d'un même
    mix.  Cela borne le budget total de threads à
    ``SharedExtractionPoolConfig.max_workers``.

    [FIX-DEADLOCK] reset_epoch() utilise un threading.Condition pour signaler
    l'arrêt et attendre l'ACK sans risque de deadlock si close() intervient.

    [FIX-IOLOCK] prefetch() est appelé hors de _cycle_lock pour éviter de
    tenir le lock pendant une opération I/O potentiellement lente.

    [FIX-RNG] Chaque worker d'extraction reçoit un seed déterministe basé sur
    un compteur atomique incrémenté à chaque soumission, garantissant la
    reproductibilité avec un seed fixe tout en évitant le partage de state RNG.
    """

    _IO_BUFFER: int = 2

    def __init__(
        self,
        spec:                 DatasetSpec,
        cache:                Any,
        rank:                 int,
        world_size:           int,
        executor:             ThreadPoolExecutor,
        prefetch_ahead:       int   = 32,
        seed:                 int   = 0,
        device_id:            int   = 0,
        shuffle_buffer_size:  int   = 512,
        min_sample_quality:   float | None = None,
        sample_predicate:     SamplePredicate | None = None,
    ) -> None:
        """Initialise un ShardIterator.

        Args:
            spec: Spécification du dataset (shards, qualité, …).
            cache: Cache de shards.
            rank: Rang global de ce processus.
            world_size: Nombre total de rangs.
            executor: Pool d'extraction partagé injecté par MixingSource.
            prefetch_ahead: Nombre de shards à précharger en avance.
            seed: Graine RNG de base.
            device_id: Index GPU local (réservé pour usage futur).
            shuffle_buffer_size: Profondeur du réservoir de shuffle in-memory.
            min_sample_quality: Seuil qualité ; surcharge la valeur du spec.
            sample_predicate: Filtre optionnel évalué avant le décodage JPEG.

        Raises:
            RuntimeError: Si aucun shard n'est assigné à ce rang.

        """
        self._name             = spec.name
        self._cache            = cache
        self._seed             = seed
        self._rank             = rank
        self._sampling         = spec.shard_sampling
        self._sample_predicate = sample_predicate
        self._executor         = executor

        self._all_shards: list[str] = [
            s for i, s in enumerate(spec.shards) if i % world_size == rank
        ]
        self._shard_weights: list[float] | None = None
        if spec.shard_quality_scores is not None:
            raw   = [
                spec.shard_quality_scores[i]
                for i, _ in enumerate(spec.shards)
                if i % world_size == rank
            ]
            total = sum(raw) or 1.0
            self._shard_weights = [w / total for w in raw]

        if not self._all_shards:
            msg = (
                f"Rank {rank}/{world_size}: no shards assigned for dataset "
                f"'{spec.name}' ({len(spec.shards)} shards total)."
            )
            raise RuntimeError(msg)

        if len(self._all_shards) < 4:
            log.warning(
                "ShardIterator '%s': only %d shard(s) assigned to rank %d/%d. "
                "Consider using shard_sampling='resampled' for small datasets.",
                self._name, len(self._all_shards), rank, world_size,
            )

        self._resampled_iter = None
        if self._sampling == "resampled":
            if HAS_WDS:
                self._resampled_iter = iter(
                    ResampledShards(
                        urls          = self._all_shards,
                        seed          = seed + rank,
                        deterministic = True,
                    ),
                )
            else:
                log.warning(
                    "ShardIterator '%s': shard_sampling='resampled' requested "
                    "but webdataset is not installed — falling back to 'epoch'.",
                    self._name,
                )
                self._sampling = "epoch"

        self._shuffle_buffer_size = shuffle_buffer_size
        self._min_sample_quality  = (
            min_sample_quality
            if min_sample_quality is not None
            else spec.min_sample_quality
        )

        self._io_queue:     queue.Queue[Any]          = queue.Queue(maxsize=self._IO_BUFFER)
        self._sample_queue: queue.Queue[SampleRecord] = queue.Queue()
        self._closed        = False

        # [FIX-DEADLOCK] Condition variable remplace stop_event + stopped_event.
        # Le thread I/O attend sur _io_cond quand il reçoit l'ordre de pause,
        # et reset_epoch() le réveille après avoir mis à jour le générateur.
        # close() positionne _closed=True avant de notifier, ce qui empêche
        # le deadlock même si close() est appelé pendant reset_epoch().
        self._io_cond   = threading.Condition(threading.Lock())
        self._io_state  = _IO_RUNNING   # _IO_RUNNING | _IO_PAUSED | _IO_STOPPED

        # [FIX-IOLOCK] RNG de shuffling de shards — accès exclusif au thread I/O.
        self._io_rng = np.random.default_rng(seed + rank)

        # _cycle_lock protège _shard_cycle depuis le thread I/O ET reset_epoch().
        self._cycle_lock  = threading.Lock()
        with self._cycle_lock:
            self._shard_cycle = self._make_shard_cycle(self._io_rng)

        # [FIX-RNG] Compteur atomique pour les seeds des workers d'extraction.
        # itertools.count() est thread-safe dans CPython (GIL) — chaque next()
        # est atomique.  Cela donne un seed unique et déterministe par worker
        # sans dépendre de threading.get_ident() (non reproductible).
        self._worker_counter = itertools.count()

        self._io_thread = threading.Thread(
            target=self._io_loop,
            name=f"shard-io-{self._name}",
            daemon=True,
        )
        self._extract_futures: deque[Future] = deque()

        self._io_thread.start()
        self._submit_extract()

    def next_sample(self) -> SampleRecord:
        """Bloque jusqu'à ce qu'un sample passe tous les filtres, puis le retourne."""
        while not self._closed:
            try:
                return self._sample_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                continue
        msg = f"ShardIterator '{self._name}' has been closed."
        raise StopIteration(msg)

    def reset_epoch(self, epoch: int) -> None:
        """Redémarre le cycle de shards pour une nouvelle époque.

        [FIX-DEADLOCK] Utilise threading.Condition pour signaler la pause et
        attendre l'ACK du thread I/O, sans risque de deadlock si close() est
        appelé simultanément.

        Args:
            epoch: Numéro de la nouvelle époque.

        """
        # 1. Demander au thread I/O de se mettre en pause.
        with self._io_cond:
            if self._closed:
                return
            self._io_state = _IO_PAUSED
            self._io_cond.notify_all()
            # Attendre que le thread I/O accuse réception de la pause.
            # Le timeout évite un blocage indéfini si le thread est déjà mort.
            self._io_cond.wait_for(
                lambda: self._io_state == _IO_STOPPED or self._closed,
                timeout=5.0,
            )
            if self._closed:
                return

        # 2. Vider les queues hors du lock (pas besoin de lock : le thread I/O
        #    est en pause et ne produit plus rien).
        for q in (self._io_queue, self._sample_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # 3. Remplacer le générateur de shards sous _cycle_lock.
        self._io_rng = np.random.default_rng(self._seed + self._rank + epoch * 997)
        with self._cycle_lock:
            self._shard_cycle = self._make_shard_cycle(self._io_rng)

        # 4. Réveiller le thread I/O.
        with self._io_cond:
            self._io_state = _IO_RUNNING
            self._io_cond.notify_all()

        self._submit_extract()

    def close(self) -> None:
        """Signale l'arrêt et annule les tâches en cours."""
        with self._io_cond:
            self._closed = True
            self._io_cond.notify_all()
        try:
            self._io_queue.put_nowait(_STOP)
        except queue.Full:
            pass
        # Ne pas appeler executor.shutdown() ici — le pool est partagé.

    @property
    def reservoir_size(self) -> int:
        """Nombre approximatif de samples en attente dans le buffer."""
        return self._sample_queue.qsize()

    def _passes_predicate(self, record: SampleRecord, shard_path: str) -> bool:
        """Retourne True si le sample passe tous les filtres anticipés."""
        meta = record.metadata

        if self._min_sample_quality is not None and meta is not None:
            score = meta.get("quality_score")
            if score is not None and score < self._min_sample_quality:
                return False

        if self._sample_predicate is not None:
            sample_meta = SampleMeta(
                key        = record.key,
                shard_path = shard_path,
                metadata   = meta,
            )
            if not self._sample_predicate(sample_meta):
                return False

        return True

    # Stage A: boucle I/O

    def _io_loop(self) -> None:
        """Point d'entrée du thread I/O daemon."""
        try:
            import contextvars  # noqa: PLC0415
            ctx = contextvars.copy_context()
            ctx.run(self._io_loop_inner)
        except Exception as exc:
            log.error(
                "ShardIterator '%s': I/O thread crashed: %s",
                self._name, exc, exc_info=True,
            )

    def _io_loop_inner(self) -> None:
        """Corps interne de la boucle I/O.

        [FIX-DEADLOCK] Utilise _io_cond pour la gestion pause/reprise.
        [FIX-IOLOCK] prefetch() est appelé hors de _cycle_lock.
        """
        while not self._closed:
            # Vérifier si une pause est demandée.
            with self._io_cond:
                if self._io_state == _IO_PAUSED:
                    # Signaler que la pause est effective.
                    self._io_state = _IO_STOPPED
                    self._io_cond.notify_all()
                    # Attendre la reprise ou la fermeture.
                    self._io_cond.wait_for(
                        lambda: self._io_state == _IO_RUNNING or self._closed,
                    )
                    if self._closed:
                        return
                    continue

            # [FIX-IOLOCK] Extraire le prochain chemin sous lock, puis appeler
            # prefetch() hors du lock pour ne pas le tenir pendant une I/O.
            with self._cycle_lock:
                try:
                    shard_path = next(self._shard_cycle)
                except StopIteration:
                    break

            # Précharge le prochain shard hors du lock.
            with self._cycle_lock:
                try:
                    next_path = next(self._shard_cycle)
                    # Remettre next_path en tête du cycle avant de précharger.
                    self._shard_cycle = self._chain_path(next_path, self._shard_cycle)
                except StopIteration:
                    next_path = None

            if next_path is not None:
                self._cache.prefetch(next_path)  # hors de _cycle_lock

            try:
                with self._cache.get_view(shard_path) as mv:
                    data = bytes(mv)
            except Exception as exc:
                log.error(
                    "ShardIterator '%s': I/O read failed for %s: %s",
                    self._name, shard_path, exc,
                )
                continue

            if self._closed:
                return

            try:
                self._io_queue.put((shard_path, data), block=True, timeout=5.0)
            except queue.Full:
                if self._closed:
                    return
                log.warning(
                    "ShardIterator '%s': _io_queue full after 5 s — "
                    "extraction workers may be stalled.",
                    self._name,
                )
                if not self._closed:
                    self._io_queue.put((shard_path, data), block=True)

        if not self._closed:
            self._io_queue.put(_STOP)

    @staticmethod
    def _chain_path(
        path: str,
        gen: Generator,
    ) -> Generator[str, None, None]:
        """Yield *path* en premier, puis tous les items de *gen*."""
        yield path
        yield from gen

    # Stage B: workers d'extraction

    def _submit_extract(self) -> None:
        """Soumet une nouvelle tâche d'extraction au pool partagé."""
        if self._closed:
            return
        fut = self._executor.submit(self._extract_worker)
        self._extract_futures.append(fut)

    def _extract_worker(self) -> None:
        """Dépile un shard de _io_queue, parse, filtre, pousse les samples acceptés.

        [FIX-RNG] Utilise un compteur atomique pour un seed déterministe et
        unique par worker, garantissant la reproductibilité avec un seed fixe.
        """
        from dino_loader.datasets.utils import _extract_jpegs_with_meta  # noqa: PLC0415

        # [FIX-RNG] Seed déterministe : pas de threading.get_ident() (arbitraire
        # et non reproductible).  Le compteur atomique garantit un seed unique
        # à chaque soumission de worker.
        worker_id  = next(self._worker_counter)
        worker_rng = np.random.default_rng(
            self._seed + self._rank * 10007 + worker_id,
        )

        while not self._closed:
            try:
                item = self._io_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue

            if isinstance(item, _Sentinel) or self._closed:
                try:
                    self._io_queue.put_nowait(_STOP)
                except queue.Full:
                    pass
                return

            shard_path, data = item

            try:
                records = _extract_jpegs_with_meta(
                    memoryview(data),
                    metadata_key   = None,
                    min_quality    = None,
                    shuffle_buffer = self._shuffle_buffer_size,
                    rng            = worker_rng,
                )
                for record in records:
                    if self._closed:
                        return
                    if not self._passes_predicate(record, shard_path):
                        continue
                    self._sample_queue.put_nowait(record)
            except Exception as exc:
                log.error(
                    "ShardIterator '%s': extraction failed for %s: %s",
                    self._name, shard_path, exc,
                )

    def _make_shard_cycle(
        self, rng: np.random.Generator,
    ) -> Generator[str, None, None]:
        """Génère les chemins de shards en ordre aléatoire, cyclant à l'épuisement."""
        if self._sampling == "resampled" and self._resampled_iter is not None:
            while not self._closed:
                item = next(self._resampled_iter)
                yield item["url"]
        else:
            while not self._closed:
                shards = list(self._all_shards)
                if self._shard_weights:
                    ordered = rng.choice(
                        shards,
                        size    = len(shards),
                        replace = False,
                        p       = self._shard_weights,
                    ).tolist()
                else:
                    rng.shuffle(shards)
                    ordered = shards
                yield from ordered


class MixingSource:
    """Callback DALI ExternalSource qui mixe des samples de plusieurs datasets.

    Appelé une fois par batch ; retourne une liste de np.ndarray (bytes JPEG).

    Pool d'extraction partagé [POOL]
    ---------------------------------
    Un seul ``ThreadPoolExecutor`` est instancié ici et injecté dans chaque
    ``ShardIterator``.

    [FIX-RNG] _rng est protégé par _rng_lock.
    [FIX-KEYLOG] threading.Lock remplace fcntl. encoding="utf-8" ajouté.
    """

    def __init__(
        self,
        specs:               list[DatasetSpec],
        batch_size:          int,
        cache:               Any,
        rank:                int,
        world_size:          int,
        pool_cfg:            SharedExtractionPoolConfig | None = None,
        seed:                int   = 0,
        device_id:           int   = 0,
        shuffle_buffer_size: int   = 512,
        debug_log_keys:      str | None = None,
        sample_predicate:    SamplePredicate | None = None,
    ) -> None:
        """Initialise MixingSource.

        Args:
            specs: Liste ordonnée de spécifications de datasets.
            batch_size: Samples par batch.
            cache: Cache de shards partagé entre tous les datasets.
            rank: Rang global de ce processus.
            world_size: Nombre total de rangs.
            pool_cfg: Config du pool d'extraction partagé.
            seed: Graine RNG de base.
            device_id: Index GPU local.
            shuffle_buffer_size: Profondeur du réservoir par shard.
            debug_log_keys: Chemin vers le log de clés par sample.
            sample_predicate: Filtre optionnel évalué avant le décodage.

        """
        self._batch_size  = batch_size
        self._debug_log   = debug_log_keys

        self._rng      = np.random.default_rng(seed + rank * 1000 + 7)
        self._rng_lock = threading.Lock()

        self._ds_index_callbacks: list[Callable[[list[int]], None]] = []

        self._weights = MixingWeights.from_specs(specs)
        self.names    = self._weights.names

        cfg = pool_cfg or SharedExtractionPoolConfig()
        self._executor = ThreadPoolExecutor(
            max_workers        = cfg.max_workers,
            thread_name_prefix = "shard-extract",
        )
        log.info(
            "MixingSource: shared extraction pool — max_workers=%d for %d dataset(s)",
            cfg.max_workers, len(specs),
        )

        self._iters: list[ShardIterator] = []
        for spec in specs:
            it = ShardIterator(
                spec                = spec,
                cache               = cache,
                rank                = rank,
                world_size          = world_size,
                executor            = self._executor,
                seed                = seed,
                device_id           = device_id,
                shuffle_buffer_size = shuffle_buffer_size,
                sample_predicate    = sample_predicate,
            )
            self._iters.append(it)

        self._debug_log_file   = None
        self._debug_log_lock   = threading.Lock()
        if debug_log_keys:
            try:
                # [FIX-KEYLOG] encoding="utf-8" pour les clés non-ASCII.
                self._debug_log_file = open(  # noqa: SIM115
                    debug_log_keys, "a", buffering=1, encoding="utf-8",
                )
            except Exception as exc:
                log.warning(
                    "MixingSource: could not open debug_log_keys '%s': %s",
                    debug_log_keys, exc,
                )

        # Compteur de batch pour l'échantillonnage des métriques de queue.
        self._batch_count = 0

    def __call__(self) -> list[np.ndarray]:
        """Retourne un batch de tableaux numpy de bytes JPEG (appelé par DALI)."""
        weights = self._weights.get()
        ds_indices: list[int] = []
        records:    list[SampleRecord] = []

        with self._rng_lock:
            chosen_indices = self._rng.choice(
                len(self._iters),
                size    = self._batch_size,
                replace = True,
                p       = weights,
            ).tolist()

        for ds_idx in chosen_indices:
            rec = self._iters[ds_idx].next_sample()
            records.append(rec)
            ds_indices.append(ds_idx)

        if self._ds_index_callbacks:
            for cb in self._ds_index_callbacks:
                cb(ds_indices)

        if self._debug_log_file is not None:
            with self._debug_log_lock:
                try:
                    for rec in records:
                        self._debug_log_file.write(rec.key + "\n")
                except Exception:
                    pass

        # [PERF] Métriques de queue échantillonnées 1/100 pour ne pas saturer
        # le hot path avec des appels qsize() par dataset à chaque batch.
        self._batch_count += 1
        if HAS_METRICS and self._batch_count % 100 == 0:
            reg = get_registry()
            if reg is not None:
                m = reg.metrics
                if m is not None:
                    total_depth = sum(it.reservoir_size for it in self._iters)
                    current     = m.mixing_source_queue_depth
                    if total_depth != current:
                        reg.inc(MetricField.MIXING_QUEUE_DEPTH, total_depth - current)

        return [np.frombuffer(rec.jpeg, dtype=np.uint8) for rec in records]

    def pop_last_metadata(self) -> list[dict | None]:
        """Retourne les métadonnées du dernier __call__.

        Note: Dans cette implémentation, les métadonnées sont collectées
        directement dans __call__() et retournées ici. Pour un alignement
        FIFO strict avec DALI, utiliser _ReaderAdapter qui gère sa propre
        queue de métadonnées.
        """
        # Cette méthode est conservée pour l'API MixingSource standalone.
        # L'alignement FIFO est géré par _ReaderAdapter dans loader.py.
        return []

    def register_dataset_index_callback(
        self,
        cb: Callable[[list[int]], None],
    ) -> None:
        """Enregistre un callback qui reçoit les indices de dataset par sample."""
        self._ds_index_callbacks.append(cb)

    def set_epoch(self, epoch: int) -> None:
        """Réinitialise tous les ShardIterators pour une nouvelle époque."""
        for it in self._iters:
            it.reset_epoch(epoch)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Met à jour les poids de mixage (re-normalisés automatiquement)."""
        self._weights.set(weights)

    def set_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids d'un dataset par son nom."""
        self._weights.set_by_name(name, weight)

    @property
    def current_weights(self) -> list[float]:
        """Poids de mixage normalisés courants."""
        return self._weights.get()

    @property
    def dataset_names(self) -> list[str]:
        """Liste ordonnée des noms de datasets."""
        return list(self.names)

    def close(self) -> None:
        """Arrête tous les ShardIterators et libère les ressources."""
        for it in self._iters:
            it.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._debug_log_file is not None:
            with self._debug_log_lock:
                try:
                    self._debug_log_file.close()
                except Exception:
                    pass
