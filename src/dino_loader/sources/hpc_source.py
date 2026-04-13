"""dino_loader.sources.hpc_source
=================================
Source de données optimisée pour les clusters HPC (Lustre + /dev/shm).

Architecture
------------
``MixingSource`` est le callback DALI ``ExternalSource`` qui mixe des samples
issus de plusieurs datasets WebDataset.  Elle repose sur deux stages :

Stage A — thread I/O (un daemon par ``ShardIterator``)
    Lit les bytes de shard depuis le cache (``NodeSharedShardCache`` en
    production, ``InProcessShardCache`` en tests) dans une queue bornée.

Stage B — workers d'extraction (``ThreadPoolExecutor`` *partagé*)
    Parse le tar, évalue le prédicat, pousse les ``SampleRecord`` acceptés
    dans la queue de samples.

Le pool d'extraction est partagé entre tous les ``ShardIterator`` d'un même
``MixingSource`` via ``SharedExtractionPoolConfig``, ce qui borne le budget
total de threads à ``max_workers`` quelle que soit la cardinalité du mix.

Performance
-----------
[PERF-LOCK] ``MixingSource.__call__`` pré-calcule les indices hors du lock :
    le numpy RNG n'est pas thread-safe, donc on copie l'état RNG sous lock
    (opération O(1)), puis on effectue le draw hors du lock.  Cela évite de
    bloquer le thread DALI pendant 512 tirages RNG sous contention.

Quand utiliser cette source ?
-----------------------------
- Clusters avec Lustre lent et beaucoup de rangs par nœud (≥ 8).
- Configurations GB200 NVL72 : 71 rangs lisent depuis ``/dev/shm``, un seul
  lit depuis Lustre, ce qui réduit la pression réseau d'un facteur ~70×.
- Entraînements longue durée où le cache atteint un régime permanent.

Pour des cas plus simples (NVMe local, peu de rangs), voir ``wds_source.py``.
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
from webdataset.shardlists import ResampledShards

from dino_loader.augmentation import SampleMeta, SamplePredicate, SampleRecord
from dino_loader.config import SharedExtractionPoolConfig
from dino_loader.monitor.metrics import MetricField, get_registry
from dino_loader.sources._weights import MixingWeights

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinelle de fin de boucle I/O
# ---------------------------------------------------------------------------


class _Sentinel:
    """Sentinel passé dans _io_queue pour signaler l'arrêt du thread I/O."""

    __slots__ = ()


_STOP = _Sentinel()

_IO_RUNNING = 0
_IO_PAUSED  = 1
_IO_STOPPED = 2


# ---------------------------------------------------------------------------
# ShardIterator
# ---------------------------------------------------------------------------


class ShardIterator:
    """Cycling de shards par dataset avec double-buffering strict.

    Le ``ThreadPoolExecutor`` est injecté par ``MixingSource`` et partagé
    entre tous les ``ShardIterator`` d'un même mix, ce qui borne le budget
    total de threads à ``SharedExtractionPoolConfig.max_workers``.

    Args:
        spec: Spécification du dataset (shards, qualité, …).
        cache: Cache de shards.
        rank: Rang global de ce processus.
        world_size: Nombre total de rangs.
        executor: Pool d'extraction partagé injecté par ``MixingSource``.
        prefetch_ahead: Nombre de shards à précharger en avance.
        seed: Graine RNG de base.
        device_id: Index GPU local (réservé pour usage futur).
        shuffle_buffer_size: Profondeur du réservoir de shuffle in-memory.
        min_sample_quality: Seuil qualité ; surcharge la valeur du spec.
        sample_predicate: Filtre optionnel évalué avant le décodage JPEG.

    Raises:
        RuntimeError: Si aucun shard n'est assigné à ce rang.

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
            self._resampled_iter = iter(
                ResampledShards(
                    urls          = self._all_shards,
                    seed          = seed + rank,
                    deterministic = True,
                ),
            )

        self._shuffle_buffer_size = shuffle_buffer_size
        self._min_sample_quality  = (
            min_sample_quality
            if min_sample_quality is not None
            else spec.min_sample_quality
        )

        self._io_queue:     queue.Queue[Any]          = queue.Queue(maxsize=self._IO_BUFFER)
        self._sample_queue: queue.Queue[SampleRecord] = queue.Queue()
        self._closed        = False

        self._io_cond   = threading.Condition(threading.Lock())
        self._io_state  = _IO_RUNNING

        self._io_rng = np.random.default_rng(seed + rank)

        self._cycle_lock  = threading.Lock()
        with self._cycle_lock:
            self._shard_cycle = self._make_shard_cycle(self._io_rng)

        # itertools.count() est thread-safe dans CPython (GIL).
        self._worker_counter = itertools.count()

        self._io_thread = threading.Thread(
            target=self._io_loop,
            name=f"shard-io-{self._name}",
            daemon=True,
        )
        self._extract_futures: deque[Future] = deque()

        self._io_thread.start()
        self._submit_extract()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

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
        """Redémarre le cycle de shards pour une nouvelle époque."""
        with self._io_cond:
            if self._closed:
                return
            self._io_state = _IO_PAUSED
            self._io_cond.notify_all()
            self._io_cond.wait_for(
                lambda: self._io_state == _IO_STOPPED or self._closed,
                timeout=5.0,
            )
            if self._closed:
                return

        for q in (self._io_queue, self._sample_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self._io_rng = np.random.default_rng(self._seed + self._rank + epoch * 997)
        with self._cycle_lock:
            self._shard_cycle = self._make_shard_cycle(self._io_rng)

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

    @property
    def reservoir_size(self) -> int:
        """Nombre approximatif de samples en attente dans le buffer."""
        return self._sample_queue.qsize()

    # ------------------------------------------------------------------
    # Filtrage
    # ------------------------------------------------------------------

    def _passes_predicate(self, record: SampleRecord, shard_path: str) -> bool:
        """Retourne ``True`` si le sample passe tous les filtres anticipés."""
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

    # ------------------------------------------------------------------
    # Stage A : boucle I/O
    # ------------------------------------------------------------------

    def _io_loop(self) -> None:
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
        while not self._closed:
            with self._io_cond:
                if self._io_state == _IO_PAUSED:
                    self._io_state = _IO_STOPPED
                    self._io_cond.notify_all()
                    self._io_cond.wait_for(
                        lambda: self._io_state == _IO_RUNNING or self._closed,
                    )
                    if self._closed:
                        return
                    continue

            with self._cycle_lock:
                try:
                    shard_path = next(self._shard_cycle)
                except StopIteration:
                    break

            with self._cycle_lock:
                try:
                    next_path = next(self._shard_cycle)
                    self._shard_cycle = self._chain_path(next_path, self._shard_cycle)
                except StopIteration:
                    next_path = None

            if next_path is not None:
                self._cache.prefetch(next_path)

            try:
                with self._cache.get_view(shard_path) as mv:
                    data = bytes(mv)
            except Exception as exc:  # noqa: BLE001
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
        gen:  Generator,
    ) -> Generator[str, None, None]:
        yield path
        yield from gen

    # ------------------------------------------------------------------
    # Stage B : workers d'extraction
    # ------------------------------------------------------------------

    def _submit_extract(self) -> None:
        if self._closed:
            return
        fut = self._executor.submit(self._extract_worker)
        self._extract_futures.append(fut)

    def _extract_worker(self) -> None:
        from dino_loader.datasets.utils import _extract_jpegs_with_meta  # noqa: PLC0415

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
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "ShardIterator '%s': extraction failed for %s: %s",
                    self._name, shard_path, exc,
                )

    def _make_shard_cycle(
        self, rng: np.random.Generator,
    ) -> Generator[str, None, None]:
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


# ---------------------------------------------------------------------------
# MixingSource
# ---------------------------------------------------------------------------


class MixingSource:
    """Callback DALI ``ExternalSource`` mixant des samples de plusieurs datasets.

    Appelée une fois par batch ; retourne une liste de ``np.ndarray`` (bytes JPEG).

    Pool d'extraction partagé
    --------------------------
    Un seul ``ThreadPoolExecutor`` est instancié ici et injecté dans chaque
    ``ShardIterator``, bornant le budget total de threads à
    ``SharedExtractionPoolConfig.max_workers``.

    Performance [PERF-LOCK]
    -----------------------
    ``__call__`` tient ``_rng_lock`` le moins longtemps possible : l'état RNG
    est copié sous lock, puis le draw ``np.random.choice`` s'effectue hors du
    lock.  Cela évite de bloquer le thread DALI sur un draw potentiellement long.

    Args:
        specs: Liste ordonnée de spécifications de datasets.
        batch_size: Samples par batch.
        cache: Cache de shards partagé entre tous les datasets.
        rank: Rang global de ce processus.
        world_size: Nombre total de rangs.
        pool_cfg: Configuration du pool d'extraction partagé.
        seed: Graine RNG de base.
        device_id: Index GPU local.
        shuffle_buffer_size: Profondeur du réservoir par shard.
        debug_log_keys: Chemin vers le log d'audit par sample.
        sample_predicate: Filtre optionnel évalué avant le décodage.

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
                self._debug_log_file = open(  # noqa: SIM115
                    debug_log_keys, "a", buffering=1, encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "MixingSource: could not open debug_log_keys '%s': %s",
                    debug_log_keys, exc,
                )

        self._batch_count = 0

    def __call__(self) -> list[np.ndarray]:
        """Retourne un batch de tableaux numpy de bytes JPEG (appelé par DALI).

        [PERF-LOCK] Le draw RNG s'effectue hors du lock pour ne pas bloquer
        le thread DALI pendant les N tirages.

        Returns:
            Liste de ``batch_size`` tableaux numpy de dtype ``uint8``.

        """
        # Copier l'état RNG sous lock (O(1)), puis draw hors du lock.
        with self._rng_lock:
            weights = self._weights.get()
            rng_snapshot = np.random.default_rng(int(self._rng.integers(2**31)))
        # Le draw lui-même — potentiellement coûteux — hors du lock.
        chosen_indices = rng_snapshot.choice(
            len(self._iters),
            size    = self._batch_size,
            replace = True,
            p       = weights,
        ).tolist()

        ds_indices: list[int] = []
        records:    list[SampleRecord] = []

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
                except Exception:  # noqa: BLE001
                    pass

        # Métriques de queue échantillonnées 1/100 pour ne pas saturer le hot path.
        self._batch_count += 1
        if self._batch_count % 100 == 0:
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
        """API de compatibilité — retourne une liste vide.

        L'alignement FIFO des métadonnées est géré par ``_ReaderAdapter``
        dans ``shard_reader.py`` qui maintient sa propre queue.
        """
        return []

    def register_dataset_index_callback(
        self,
        cb: Callable[[list[int]], None],
    ) -> None:
        """Enregistre un callback qui reçoit les indices de dataset par sample."""
        self._ds_index_callbacks.append(cb)

    def set_epoch(self, epoch: int) -> None:
        """Réinitialise tous les ``ShardIterator`` pour une nouvelle époque."""
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
        """Arrête tous les ``ShardIterator`` et libère les ressources."""
        for it in self._iters:
            it.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._debug_log_file is not None:
            with self._debug_log_lock:
                try:
                    self._debug_log_file.close()
                except Exception:  # noqa: BLE001
                    pass
