"""dino_loader.dali_node
======================
``_DALINode`` — torchdata ``BaseNode`` qui pilote un itérateur backend
(DALI ou CPU) et assemble des ``Batch`` prêts pour la boucle d'entraînement.

Responsabilité unique
---------------------
Ce nœud est la frontière entre ``torchdata.nodes`` et le pipeline
d'augmentation.  Il ne contient aucune logique d'augmentation, de filtrage
ou de composition de pipeline.

Séparation des responsabilités
--------------------------------
::

    BackendProtocol.build_pipeline_iterator()   → produit un itérateur
    _DALINode                                   → consomme l'itérateur, assemble Batch
    pipeline_graph.NodePipeline                 → compose des transforms sur les Batch

Thread safety
-------------
``reset_iter()`` et ``next()`` peuvent être appelés depuis des threads
différents.  ``_iter_lock`` protège l'accès à ``_iter`` contre la race
condition entre ``set_epoch()`` (thread principal) et ``next()`` (thread
torchdata).
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from torchdata.nodes import BaseNode

from dino_loader.memory import Batch

log = logging.getLogger(__name__)


class _DALINode(BaseNode):  # type: ignore[misc]
    """Pilote un itérateur DALI/CPU et émet des ``Batch`` fully assembled.

    Args:
        dali_iter_factory: ``() -> iterator`` appelé à chaque ``reset()``.
            Produit un itérateur compatible ``DALIGenericIterator``.
        pop_metadata_fn: ``() -> list[dict | None]`` — récupère les
            métadonnées du batch courant depuis l'adaptateur source.
        build_batch_fn: ``(views, metadata) -> Batch`` — assemble le batch.
        output_map: Noms des vues produites par l'itérateur.
        stall_timeout_s: Secondes avant levée si aucun batch. 0 = désactivé.
        rank: Rang global pour les messages d'erreur.

    """

    def __init__(
        self,
        dali_iter_factory: Callable[[], Any],
        pop_metadata_fn:   Callable[[], list[dict | None]],
        build_batch_fn:    Callable[[list[Any], list[dict | None]], Batch],
        output_map:        list[str],
        stall_timeout_s:   float = 600.0,
        rank:              int   = 0,
    ) -> None:
        super().__init__()
        self._iter_factory  = dali_iter_factory
        self._pop_metadata  = pop_metadata_fn
        self._build_batch   = build_batch_fn
        self._output_map    = output_map
        self._stall_timeout = stall_timeout_s
        self._rank          = rank
        self._iter: Any     = None
        self._num_yielded   = 0
        self._iter_lock     = threading.Lock()

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """(Re-)initialise l'itérateur pour une nouvelle époque."""
        super().reset(initial_state)
        with self._iter_lock:
            self._iter        = self._iter_factory()
            self._num_yielded = 0

    def reset_iter(self) -> None:
        """Invalide l'itérateur courant — appelé par ``set_epoch()`` (thread-safe).

        La prochaine invocation de ``reset()`` recréera l'itérateur via la
        factory, prenant en compte la nouvelle configuration d'époque.
        """
        with self._iter_lock:
            self._iter = None

    def next(self) -> Batch:
        """Retourne le prochain ``Batch`` assemblé.

        Raises:
            StopIteration: En fin d'époque.
            RuntimeError: Si aucun batch n'est produit et stall_timeout_s > 0.
            AssertionError: Si ``reset()`` n'a pas été appelé.

        """
        with self._iter_lock:
            current_iter = self._iter

        if current_iter is None:
            msg = "reset() must be called before next()"
            raise AssertionError(msg)

        try:
            dali_out = next(current_iter)
        except StopIteration:
            if self._num_yielded == 0 and self._stall_timeout > 0:
                if os.environ.get("DINO_DISABLE_EMPTY_CHECK"):
                    log.warning(
                        "_DALINode rank %d: aucun batch produit mais "
                        "DINO_DISABLE_EMPTY_CHECK est actif.",
                        self._rank,
                    )
                else:
                    msg = (
                        f"_DALINode (rank {self._rank}): aucun batch produit. "
                        "Causes possibles : shards corrompus, /dev/shm plein, "
                        "sample_predicate a rejeté tous les samples, démarrage MDS lent. "
                        "Désactiver : DINO_DISABLE_EMPTY_CHECK=1 ou stall_timeout_s=0."
                    )
                    raise RuntimeError(msg) from None
            raise

        t0       = time.perf_counter()
        views    = [dali_out[0][name] for name in self._output_map]
        metadata = self._pop_metadata()
        batch    = self._build_batch(views, metadata)
        elapsed  = int((time.perf_counter() - t0) * 1000)

        self._num_yielded += 1
        _update_metrics(elapsed)

        return batch

    def get_state(self) -> dict[str, Any]:
        """Retourne l'état persistable de ce nœud."""
        return {"_num_yielded": self._num_yielded}


def _update_metrics(elapsed_ms: int) -> None:
    """Incrémente les métriques loader via le registry global."""
    try:
        from dino_loader.monitor.metrics import get_registry  # noqa: PLC0415
        reg = get_registry()
        if reg is not None:
            reg.inc("loader_batches_yielded", 1)
            reg.inc("pipeline_yield_time_ms", elapsed_ms)
            reg.heartbeat()
    except Exception:  # noqa: BLE001
        pass
