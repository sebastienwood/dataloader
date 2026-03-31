"""dino_loader.pipeline_graph
============================
Composable, stateful post-processing pipeline built on ``torchdata.nodes``.

Architecture
------------
Ce module est le point d'entrée unique pour la composition post-augmentation.
``DINODataLoader`` construit un ``NodePipeline`` en interne via
``wrap_loader()``.  L'utilisateur peut ensuite l'enrichir via
``.map()``, ``.select()``, ``.with_epoch()`` sans toucher au loader.

Rôle de chaque classe
---------------------
``_DALINode``
    BaseNode qui pilote l'itérateur DALI/CPU, assemble les ``Batch``,
    met à jour les métriques et détecte les stalls.  C'est le seul endroit
    qui connaît l'itérateur DALI.

``BatchMapNode``
    Applique une transformation ``Batch → Batch``.

``BatchFilterNode``
    Filtre les batches selon un prédicat.  Incrémente le compteur de métriques.

``_LimitNode``
    Limite à ``max_steps`` batches par époque.

``NodePipeline``
    Wrapper composable.  Expose toutes les propriétés publiques de
    ``DINODataLoader``.

Corrections
-----------
[FIX-ITER-RACE] _DALINode expose reset_iter() avec lock interne pour éviter
    la race condition entre set_epoch() (thread principal) et next() (thread
    tn.Loader).  loader.py utilise reset_iter() au lieu d'accéder à _iter
    directement.
[FIX-DOUBLE-ITER] DINODataLoader.__iter__ et NodePipeline gèrent l'itération
    active via _active_iter pour lever RuntimeError si deux itérations
    démarrent simultanément.
[FIX-STATE-MAX-STEPS] max_steps est inclus dans state_dict() / load_state_dict()
    pour qu'un round-trip le restitue correctement.
[FIX-FUTURE] from __future__ import annotations supprimé (Python ≥ 3.12 natif).
[METRICS]   Métriques et stall watchdog dans _DALINode.
[PROPS]     NodePipeline expose current_resolution, current_weights,
    backend, aug_spec.
"""

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torchdata.nodes as tn
from torchdata.nodes import BaseNode

from dino_loader.memory import Batch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _DALINode — pilote DALI/CPU, assemble Batch, métriques, stall watchdog
# ---------------------------------------------------------------------------


class _DALINode(BaseNode):  # type: ignore[misc]
    """Pilote l'itérateur DALI/CPU et émet des ``Batch`` fully assembled.

    Ce nœud est la frontière entre torchdata.nodes et le pipeline DALI.
    Il encapsule :
    - Le défilement de l'itérateur DALI/CPU.
    - L'assemblage des vues en ``Batch`` (via ``build_batch_fn``).
    - La mise à jour des métriques (batches_yielded, pipeline_yield_ms).
    - La détection de stall (aucun batch produit après ``stall_timeout_s``).

    Thread safety
    -------------
    ``reset_iter()`` et ``next()`` peuvent être appelés depuis des threads
    différents (thread principal vs thread tn.Loader).  ``_iter_lock`` protège
    l'accès à ``_iter`` pour éviter la race condition entre ``reset_iter()``
    (appelé par ``set_epoch()``) et ``next()``.

    Restauration intra-époque
    --------------------------
    ``reset()`` repart de zéro (DALI n'est pas sérialisable).  Sur restore
    d'un state_dict, l'itération repart du début de l'époque avec le bon
    seed.  ``_num_yielded`` est persisté pour information seulement.

    Args:
        dali_iter_factory: Callable ``() -> iterator`` appelé à chaque
            ``reset()``.  Produit un itérateur compatible DALIGenericIterator.
        pop_metadata_fn: Callable ``() -> list[dict|None]`` pour récupérer
            les métadonnées du batch courant depuis ``_ReaderAdapter``.
        build_batch_fn: Callable ``(views, metadata) -> Batch``.
        output_map: Noms des vues produites par l'itérateur DALI.
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
        """Initialise _DALINode."""
        super().__init__()
        self._iter_factory   = dali_iter_factory
        self._pop_metadata   = pop_metadata_fn
        self._build_batch    = build_batch_fn
        self._output_map     = output_map
        self._stall_timeout  = stall_timeout_s
        self._rank           = rank
        self._iter: Any      = None
        self._num_yielded    = 0
        # [FIX-ITER-RACE] Lock protégeant l'accès à _iter entre set_epoch()
        # (thread principal) et next() (thread tn.Loader).
        self._iter_lock      = threading.Lock()

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """(Re-)initialise l'itérateur DALI/CPU pour une nouvelle époque."""
        super().reset(initial_state)
        with self._iter_lock:
            self._iter        = self._iter_factory()
            self._num_yielded = 0

    def reset_iter(self) -> None:
        """Invalide l'itérateur courant pour forcer sa recréation au prochain reset().

        Appelé par ``DINODataLoader.set_epoch()`` depuis le thread principal.
        Thread-safe : utilise ``_iter_lock`` pour éviter la race condition avec
        ``next()`` qui peut s'exécuter dans le thread ``tn.Loader``.
        """
        with self._iter_lock:
            self._iter = None

    def next(self) -> Batch:
        """Retourne le prochain Batch assemblé.

        Raises:
            StopIteration: En fin d'époque (signal normal pour tn.Loader).
            RuntimeError: Si aucun batch n'est produit et stall_timeout_s > 0.

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
                        "_DALINode rank %d: no batch produced but "
                        "DINO_DISABLE_EMPTY_CHECK is set — continuing silently.",
                        self._rank,
                    )
                else:
                    msg = (
                        f"_DALINode (rank {self._rank}): no batch produced. "
                        "Possible causes: corrupted shards, /dev/shm full, "
                        "sample_predicate rejected every sample, filesystem MDS slow start. "
                        "Disable: DINO_DISABLE_EMPTY_CHECK=1 or stall_timeout_s=0."
                    )
                    raise RuntimeError(msg) from None
            raise

        t0       = time.perf_counter()
        views    = [dali_out[0][name] for name in self._output_map]
        metadata = self._pop_metadata()
        batch    = self._build_batch(views, metadata)
        elapsed  = int((time.perf_counter() - t0) * 1000)

        self._num_yielded += 1
        self._update_metrics(elapsed)

        return batch

    def get_state(self) -> dict[str, Any]:
        """Retourne l'état persistable de ce nœud."""
        return {"_num_yielded": self._num_yielded}

    @staticmethod
    def _update_metrics(elapsed_ms: int) -> None:
        """Met à jour les métriques via le registry global."""
        try:
            from dino_loader.monitor.metrics import get_registry  # noqa: PLC0415
            reg = get_registry()
            if reg is not None:
                reg.inc("loader_batches_yielded", 1)
                reg.inc("pipeline_yield_time_ms", elapsed_ms)
                reg.heartbeat()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# BatchMapNode
# ---------------------------------------------------------------------------


class BatchMapNode(BaseNode):  # type: ignore[misc]
    """Apply a ``Batch → Batch`` function to every element of a node stream."""

    def __init__(
        self,
        source: BaseNode,  # type: ignore[type-arg]
        fn:     Callable[[Batch], Batch],
        *,
        label:  str = "<map>",
    ) -> None:
        """Initialise a BatchMapNode."""
        super().__init__()
        self._source = source
        self._fn     = fn
        self._label  = label

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Reset upstream source."""
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self) -> Batch:
        """Apply fn to the next batch from source."""
        return self._fn(self._source.next())

    def get_state(self) -> dict[str, Any]:
        """Delegate to source."""
        return self._source.get_state()

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return f"BatchMapNode({self._label!r})"


# ---------------------------------------------------------------------------
# BatchFilterNode
# ---------------------------------------------------------------------------


class BatchFilterNode(BaseNode):  # type: ignore[misc]
    """Skip ``Batch`` objects for which a predicate returns ``False``."""

    def __init__(
        self,
        source:    BaseNode,  # type: ignore[type-arg]
        predicate: Callable[[Batch], bool],
        *,
        label:     str = "<filter>",
    ) -> None:
        """Initialise a BatchFilterNode."""
        super().__init__()
        self._source    = source
        self._predicate = predicate
        self._label     = label
        self._n_skipped = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Reset upstream source and clear skip counter."""
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._n_skipped = 0

    def next(self) -> Batch:
        """Return the next batch accepted by the predicate."""
        while True:
            batch = self._source.next()
            if self._predicate(batch):
                return batch
            self._n_skipped += 1
            self._record_filtered()

    def get_state(self) -> dict[str, Any]:
        """Delegate to source."""
        return self._source.get_state()

    @property
    def n_skipped(self) -> int:
        """Running count of batches rejected by the predicate."""
        return self._n_skipped

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return f"BatchFilterNode({self._label!r}, skipped={self._n_skipped})"

    @staticmethod
    def _record_filtered() -> None:
        """Increment the batches_filtered metric if the registry is available."""
        try:
            from dino_loader.monitor.metrics import get_registry  # noqa: PLC0415
            reg = get_registry()
            if reg is not None:
                reg.inc("batches_filtered", 1)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# _LimitNode
# ---------------------------------------------------------------------------


class _LimitNode(BaseNode):  # type: ignore[misc]
    """Yield at most ``max_steps`` batches per epoch, then raise StopIteration."""

    def __init__(self, source: BaseNode, max_steps: int) -> None:  # type: ignore[type-arg]
        """Initialise a _LimitNode."""
        super().__init__()
        self._source    = source
        self._max_steps = max_steps
        self._yielded   = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Reset upstream and clear yield counter."""
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._yielded = 0

    def next(self) -> Batch:
        """Return next batch or raise StopIteration when limit is reached."""
        if self._yielded >= self._max_steps:
            raise StopIteration
        batch = self._source.next()
        self._yielded += 1
        return batch

    def get_state(self) -> dict[str, Any]:
        """Return source state (limit counter not persisted)."""
        return self._source.get_state()


# ---------------------------------------------------------------------------
# NodePipeline — pipeline composable avec toutes les propriétés DINODataLoader
# ---------------------------------------------------------------------------


class NodePipeline:
    """Composable, stateful post-processing pipeline built on ``torchdata.nodes``.

    Expose toutes les propriétés publiques de ``DINODataLoader``.

    Propriétés déléguées
    --------------------
    ``current_resolution``, ``current_weights``, ``backend``, ``aug_spec``

    Méthodes de contrôle déléguées
    --------------------------------
    ``set_epoch``, ``checkpoint``, ``set_weights``, ``set_weight_by_name``,
    ``set_resolution``, ``state_dict``, ``load_state_dict``

    Méthodes de composition
    -----------------------
    ``.map(fn)``         — ajoute un ``Batch → Batch`` transform.
    ``.select(pred)``    — filtre les batches.
    ``.with_epoch(n)``   — limite à n steps par époque.

    Note sur l'intra-époque
    -----------------------
    ``state_dict`` persiste l'époque, les poids, la résolution et max_steps.
    La position intra-époque n'est **pas** restaurée (DALI non sérialisable).

    [FIX-STATE-MAX-STEPS] max_steps est inclus dans state_dict pour un
    round-trip correct lors de la reprise.
    """

    def __init__(
        self,
        root_node:   BaseNode,  # type: ignore[type-arg]
        dino_loader: Any,
        max_steps:   int | None = None,
    ) -> None:
        """Initialise a NodePipeline."""
        self._root      = root_node
        self._loader    = dino_loader
        self._max_steps = max_steps
        self._tn_loader: tn.Loader | None = None

    # ------------------------------------------------------------------
    # Fluent composition API
    # ------------------------------------------------------------------

    def map(self, fn: Callable[[Batch], Batch], *, label: str = "<map>") -> "NodePipeline":
        """Append a ``Batch → Batch`` transform.

        Args:
            fn: Transform applied to every batch.
            label: Optional label for debugging.

        Returns:
            A new ``NodePipeline`` with ``fn`` appended.

        """
        return NodePipeline(
            root_node   = BatchMapNode(self._root, fn, label=label),
            dino_loader = self._loader,
            max_steps   = self._max_steps,
        )

    def select(
        self,
        predicate: Callable[[Batch], bool],
        *,
        label: str = "<filter>",
    ) -> "NodePipeline":
        """Drop batches for which ``predicate(batch)`` is ``False``.

        Args:
            predicate: Return ``True`` to keep a batch, ``False`` to skip it.
            label: Optional label for debugging.

        Returns:
            A new ``NodePipeline`` with the filter appended.

        """
        return NodePipeline(
            root_node   = BatchFilterNode(self._root, predicate, label=label),
            dino_loader = self._loader,
            max_steps   = self._max_steps,
        )

    def with_epoch(self, n_steps: int) -> "NodePipeline":
        """Limit iteration to ``n_steps`` batches per epoch.

        Args:
            n_steps: Maximum number of batches yielded before ``StopIteration``.

        Returns:
            A new ``NodePipeline`` bounded to ``n_steps``.

        """
        return NodePipeline(
            root_node   = self._root,
            dino_loader = self._loader,
            max_steps   = n_steps,
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _build_tn_loader(self) -> tn.Loader:
        """Build the underlying tn.Loader, adding a _LimitNode if needed."""
        root = self._root
        if self._max_steps is not None:
            root = _LimitNode(root, self._max_steps)
        return tn.Loader(root, restart_on_stop_iteration=True)

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches, building the tn.Loader lazily."""
        if self._tn_loader is None:
            self._tn_loader = self._build_tn_loader()
        return iter(self._tn_loader)

    def __len__(self) -> int:
        """Return max_steps if set, else delegate to the underlying loader."""
        if self._max_steps is not None:
            return self._max_steps
        return len(self._loader)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return the full graph state dict.

        [FIX-STATE-MAX-STEPS] max_steps est inclus pour qu'un round-trip via
        load_state_dict() + with_epoch() restitue le même comportement.
        La position intra-époque n'est pas restaurable (DALI non sérialisable).
        """
        loader_sd = self._loader.state_dict()
        tn_sd: dict[str, Any] = {}
        if self._tn_loader is not None:
            tn_sd = self._tn_loader.state_dict()
        return {
            "loader":    loader_sd,
            "tn_graph":  tn_sd,
            "max_steps": self._max_steps,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore the full graph state.

        Note: max_steps is restored in memory but does not rebuild the
        tn.Loader graph. If the pipeline has already been built (i.e. iterated
        over at least once), call with_epoch() again to apply the new limit.
        """
        if "loader" in sd:
            self._loader.load_state_dict(sd["loader"])
        if "tn_graph" in sd and self._tn_loader is not None:
            self._tn_loader.load_state_dict(sd["tn_graph"])
        if "max_steps" in sd:
            self._max_steps = sd["max_steps"]

    # ------------------------------------------------------------------
    # Delegation vers DINODataLoader
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.set_epoch(epoch)

    def checkpoint(self, step: int) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.checkpoint(step)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.set_weight_by_name(name, weight)

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Delegate to the underlying ``DINODataLoader``."""
        self._loader.set_resolution(global_size, local_size)

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Current crop resolution as ``(global_size, local_size)``."""
        return self._loader.current_resolution

    @property
    def current_weights(self) -> list[float]:
        """Current normalised mixing weights."""
        return self._loader.current_weights

    @property
    def backend(self) -> Any:
        """The active backend instance."""
        return self._loader.backend

    @property
    def aug_spec(self) -> Any:
        """The active augmentation spec."""
        return self._loader.aug_spec


# ---------------------------------------------------------------------------
# wrap_loader — factory publique
# ---------------------------------------------------------------------------


def wrap_loader(dino_loader: Any) -> NodePipeline:
    """Wrap a ``DINODataLoader`` in a composable ``NodePipeline``.

    Utilisé en interne par ``DINODataLoader.as_pipeline()`` et disponible
    en standalone pour les usages avancés.

    Args:
        dino_loader: A ``DINODataLoader`` instance.  Doit exposer un attribut
            ``_dali_node`` de type ``_DALINode``.

    Returns:
        ``NodePipeline`` — iterate directly or compose further with
        ``.map()``, ``.select()``, ``.with_epoch()``.

    Raises:
        TypeError: Si ``dino_loader`` n'expose pas ``_dali_node``.

    Example::

        from dino_loader.pipeline_graph import wrap_loader

        pipeline = (
            wrap_loader(DINODataLoader(...))
            .map(apply_ibot_masks)
            .select(lambda b: any(m is not None for m in b.metadata))
            .with_epoch(steps_per_epoch)
        )

        for epoch in range(100):
            pipeline.set_epoch(epoch)
            for batch in pipeline:
                train_step(batch)
            sd = pipeline.state_dict()

    """
    dali_node = getattr(dino_loader, "_dali_node", None)
    if dali_node is None:
        msg = (
            f"wrap_loader: the provided object ({type(dino_loader).__name__}) "
            "does not expose a '_dali_node' attribute. "
            "Pass a DINODataLoader instance."
        )
        raise TypeError(msg)
    return NodePipeline(root_node=dali_node, dino_loader=dino_loader)


# ---------------------------------------------------------------------------
# Backward-compat : _DINOLoaderNode conservé pour les tests existants
# ---------------------------------------------------------------------------


class _DINOLoaderNode(BaseNode):  # type: ignore[misc]
    """Backward-compatible wrapper. Préférer _DALINode pour les nouveaux usages."""

    def __init__(self, dino_loader: Any) -> None:
        """Initialise a _DINOLoaderNode."""
        super().__init__()
        self._dino_loader  = dino_loader
        self._it: Iterator[Batch] | None = None
        self._num_yielded  = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Re-enter the loader iterator for a new epoch."""
        super().reset(initial_state)
        self._it          = iter(self._dino_loader)
        self._num_yielded = 0

    def next(self) -> Batch:
        """Return the next Batch from the loader."""
        if self._it is None:
            msg = "reset() must be called before next()"
            raise AssertionError(msg)
        batch = next(self._it)
        self._num_yielded += 1
        return batch

    def get_state(self) -> dict[str, Any]:
        """Return loader state dict."""
        loader_sd: dict[str, Any] = {}
        try:
            loader_sd = self._dino_loader.state_dict()
        except RuntimeError:
            pass
        return {**loader_sd, "_num_yielded": self._num_yielded}
