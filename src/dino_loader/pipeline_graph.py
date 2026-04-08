"""dino_loader.pipeline_graph
============================
Pipeline de post-traitement composable et stateful, construit sur
``torchdata.nodes``.

Responsabilités de ce module
-----------------------------
Ce module est le point d'entrée unique pour **tout ce qui opère sur des
``Batch``** après la sortie du pipeline d'augmentation (DALI ou CPU) :

``_DALINode``
    Pilote l'itérateur DALI/CPU, assemble les ``Batch``, met à jour les
    métriques et détecte les stalls.  Seul endroit qui connaît l'itérateur.

``MetadataNode``
    Sépare les flux JPEG et métadonnées issus d'un ``ShardReaderNode``.
    Bufferise les métadonnées pour une récupération décorrélée.

``MaskMapNode``
    Attache des masques de patches iBOT (``MaskingGenerator``) aux ``Batch``.
    Opère sur des indices de patches ViT, pas sur des pixels — ne peut donc
    pas être fusionné dans le graphe DALI.

``BatchMapNode``
    Applique une transformation ``Batch → Batch`` arbitraire.

``BatchFilterNode``
    Filtre les batches selon un prédicat booléen.

``_LimitNode``
    Borne l'itération à ``max_steps`` batches par époque.

``NodePipeline``
    Wrapper composable exposant toutes les propriétés publiques de
    ``DINODataLoader``.  Point d'entrée pour l'utilisateur final.

``wrap_loader``
    Factory publique : bridge ``DINODataLoader`` → ``NodePipeline``.

Séparation des responsabilités
--------------------------------
Ce module n'importe PAS depuis ``shard_reader`` (dépendance sens unique) :

    loader.py → shard_reader.py → sources/
    loader.py → pipeline_graph.py
    pipeline_graph.py → memory.py (Batch)

Corrections intégrées
---------------------
[FIX-ITER-RACE]    _DALINode.reset_iter() protégé par lock interne.
[FIX-DOUBLE-ITER]  DINODataLoader.__iter__ lève RuntimeError si déjà actif.
[FIX-STATE-MAX-STEPS] max_steps inclus dans state_dict / load_state_dict.
[METRICS]          Métriques et stall watchdog dans _DALINode.
[PROPS]            NodePipeline expose current_resolution, current_weights,
                   backend, aug_spec.
"""

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch
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

    Args:
        dali_iter_factory: Callable ``() -> iterator`` appelé à chaque
            ``reset()``.  Produit un itérateur compatible DALIGenericIterator.
        pop_metadata_fn:   Callable ``() -> list[dict|None]`` pour récupérer
            les métadonnées du batch courant.
        build_batch_fn:    Callable ``(views, metadata) -> Batch``.
        output_map:        Noms des vues produites par l'itérateur DALI.
        stall_timeout_s:   Secondes avant levée si aucun batch. 0 = désactivé.
        rank:              Rang global pour les messages d'erreur.

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
        """(Re-)initialise l'itérateur DALI/CPU pour une nouvelle époque."""
        super().reset(initial_state)
        with self._iter_lock:
            self._iter        = self._iter_factory()
            self._num_yielded = 0

    def reset_iter(self) -> None:
        """Invalide l'itérateur courant pour forcer sa recréation au prochain reset().

        Appelé par ``DINODataLoader.set_epoch()`` depuis le thread principal.
        Thread-safe via ``_iter_lock``.
        """
        with self._iter_lock:
            self._iter = None

    def next(self) -> Batch:
        """Retourne le prochain Batch assemblé.

        Raises:
            StopIteration: En fin d'époque.
            RuntimeError:  Si aucun batch n'est produit et stall_timeout_s > 0.
            AssertionError: Si reset() n'a pas été appelé.

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
                        "DINO_DISABLE_EMPTY_CHECK est actif — continuation silencieuse.",
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
# MetadataNode
# ---------------------------------------------------------------------------


class MetadataNode(BaseNode):  # type: ignore[misc]
    """Sépare le flux JPEG et métadonnées issu d'un ``ShardReaderNode``.

    ``ShardReaderNode.next()`` retourne un ``ReaderBatch`` :
    ``(list[np.ndarray], list[dict | None])``.  ``MetadataNode`` bufferise
    les métadonnées pour permettre leur récupération décorrélée (après la
    passe DALI) via ``pop_last_metadata()``.

    Args:
        source: Nœud amont produisant des ``ReaderBatch``.

    """

    def __init__(self, source: BaseNode) -> None:  # type: ignore[type-arg]
        """Initialise un MetadataNode."""
        super().__init__()
        self._source: BaseNode                           = source  # type: ignore[type-arg]
        self._last_meta: list[dict[str, Any] | None]    = []

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Remet à zéro la source amont et vide le buffer de métadonnées."""
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._last_meta = []

    def next(self) -> tuple[list[np.ndarray], list[dict[str, Any] | None]]:
        """Retourne le prochain batch en bufferisant les métadonnées."""
        jpegs, meta     = self._source.next()
        self._last_meta = meta
        return jpegs, meta

    def get_state(self) -> dict[str, Any]:
        """Délègue l'état à la source amont."""
        return self._source.get_state()

    def pop_last_metadata(self) -> list[dict[str, Any] | None]:
        """Retourne les métadonnées du dernier ``next()`` et vide le buffer."""
        meta, self._last_meta = self._last_meta, []
        return meta


# ---------------------------------------------------------------------------
# MaskMapNode
# ---------------------------------------------------------------------------


class MaskMapNode(BaseNode):  # type: ignore[misc]
    """Attache des masques de patches iBOT à chaque ``Batch``.

    ``MaskingGenerator`` opère sur des indices de patches ViT (grille bool
    de forme ``grid × grid`` où ``grid = img_size // patch_size``), et non
    sur des pixels.  Le graphe DALI ne peut traiter que des tenseurs image
    denses : ce nœud doit donc rester hors du graphe DALI, sur CPU.

    Le surcoût CPU est d'environ 0,3 ms par batch pour une grille 37×37 —
    négligeable face aux ~40 ms du décodage DALI.

    Args:
        source:              Nœud amont produisant des ``Batch``.
        mask_generator:      Instance de ``MaskingGenerator``.
        num_masking_patches: Nombre cible de patches masqués (transmis au
                             générateur si fourni).

    """

    def __init__(
        self,
        source:              BaseNode,  # type: ignore[type-arg]
        mask_generator:      Any,
        num_masking_patches: int | None = None,
    ) -> None:
        """Initialise un MaskMapNode."""
        super().__init__()
        self._source = source
        self._gen    = mask_generator
        self._n_mask = num_masking_patches

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Remet à zéro la source amont."""
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self) -> Batch:
        """Retourne le prochain ``Batch`` avec ``Batch.masks`` renseigné."""
        batch: Batch = self._source.next()
        if not batch.global_crops:
            msg = (
                "MaskMapNode.next: batch.global_crops est vide. "
                "MaskMapNode requiert au moins un crop global pour dériver "
                "la dimension batch des masques."
            )
            raise ValueError(msg)

        mask  = self._gen(flat=True)
        masks = torch.from_numpy(mask).unsqueeze(0)
        b     = batch.global_crops[0].shape[0]
        masks = masks.expand(b, -1)

        batch.masks = masks
        return batch

    def get_state(self) -> dict[str, Any]:
        """Délègue l'état à la source amont."""
        return self._source.get_state()

    @staticmethod
    def as_transform(
        mask_generator:      Any,
        num_masking_patches: int | None = None,
    ) -> Callable[[Batch], Batch]:
        """Retourne un callable ``Batch → Batch`` pour utilisation avec ``.map()``.

        Args:
            mask_generator:      Instance de ``MaskingGenerator``.
            num_masking_patches: Passé au générateur (non utilisé actuellement,
                                 réservé pour la validation future).

        Returns:
            Callable applicable via ``NodePipeline.map()``.

        """
        def _apply(batch: Batch) -> Batch:
            if not batch.global_crops:
                msg = (
                    "MaskMapNode.as_transform: batch.global_crops est vide. "
                    "Le masquage requiert au moins un crop global."
                )
                raise ValueError(msg)

            mask  = mask_generator(flat=True)
            masks = torch.from_numpy(mask).unsqueeze(0)
            b     = batch.global_crops[0].shape[0]
            masks = masks.expand(b, -1)

            return Batch(
                global_crops = batch.global_crops,
                local_crops  = batch.local_crops,
                metadata     = batch.metadata,
                masks        = masks,
            )

        return _apply


# ---------------------------------------------------------------------------
# BatchMapNode
# ---------------------------------------------------------------------------


class BatchMapNode(BaseNode):  # type: ignore[misc]
    """Applique une transformation ``Batch → Batch`` à chaque élément du flux."""

    def __init__(
        self,
        source: BaseNode,  # type: ignore[type-arg]
        fn:     Callable[[Batch], Batch],
        *,
        label:  str = "<map>",
    ) -> None:
        """Initialise un BatchMapNode."""
        super().__init__()
        self._source = source
        self._fn     = fn
        self._label  = label

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Remet à zéro la source amont."""
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self) -> Batch:
        """Applique fn au prochain batch de la source."""
        return self._fn(self._source.next())

    def get_state(self) -> dict[str, Any]:
        """Délègue l'état à la source amont."""
        return self._source.get_state()

    def __repr__(self) -> str:
        """Représentation compacte."""
        return f"BatchMapNode({self._label!r})"


# ---------------------------------------------------------------------------
# BatchFilterNode
# ---------------------------------------------------------------------------


class BatchFilterNode(BaseNode):  # type: ignore[misc]
    """Ignore les ``Batch`` pour lesquels un prédicat retourne ``False``."""

    def __init__(
        self,
        source:    BaseNode,  # type: ignore[type-arg]
        predicate: Callable[[Batch], bool],
        *,
        label:     str = "<filter>",
    ) -> None:
        """Initialise un BatchFilterNode."""
        super().__init__()
        self._source    = source
        self._predicate = predicate
        self._label     = label
        self._n_skipped = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Remet à zéro la source et le compteur de batches ignorés."""
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._n_skipped = 0

    def next(self) -> Batch:
        """Retourne le prochain batch accepté par le prédicat."""
        while True:
            batch = self._source.next()
            if self._predicate(batch):
                return batch
            self._n_skipped += 1
            self._record_filtered()

    def get_state(self) -> dict[str, Any]:
        """Délègue l'état à la source amont."""
        return self._source.get_state()

    @property
    def n_skipped(self) -> int:
        """Nombre cumulé de batches rejetés par le prédicat."""
        return self._n_skipped

    def __repr__(self) -> str:
        """Représentation compacte."""
        return f"BatchFilterNode({self._label!r}, skipped={self._n_skipped})"

    @staticmethod
    def _record_filtered() -> None:
        """Incrémente le compteur batches_filtered si le registry est disponible."""
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
    """Émet au plus ``max_steps`` batches par époque, puis lève StopIteration."""

    def __init__(self, source: BaseNode, max_steps: int) -> None:  # type: ignore[type-arg]
        """Initialise un _LimitNode."""
        super().__init__()
        self._source    = source
        self._max_steps = max_steps
        self._yielded   = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        """Remet à zéro la source et le compteur de batches émis."""
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._yielded = 0

    def next(self) -> Batch:
        """Retourne le prochain batch ou lève StopIteration si la limite est atteinte."""
        if self._yielded >= self._max_steps:
            raise StopIteration
        batch = self._source.next()
        self._yielded += 1
        return batch

    def get_state(self) -> dict[str, Any]:
        """Retourne l'état de la source (compteur non persisté)."""
        return self._source.get_state()


# ---------------------------------------------------------------------------
# NodePipeline — pipeline composable exposant toutes les propriétés DINODataLoader
# ---------------------------------------------------------------------------


class NodePipeline:
    """Pipeline de post-traitement composable et stateful.

    Expose toutes les propriétés publiques de ``DINODataLoader`` par délégation,
    ce qui permet de l'utiliser comme drop-in replacement dans la boucle
    d'entraînement.

    Propriétés déléguées
    --------------------
    ``current_resolution``, ``current_weights``, ``backend``, ``aug_spec``

    Méthodes de contrôle déléguées
    --------------------------------
    ``set_epoch``, ``checkpoint``, ``set_weights``, ``set_weight_by_name``,
    ``set_resolution``, ``state_dict``, ``load_state_dict``

    Méthodes de composition (retournent un nouveau NodePipeline)
    -------------------------------------------------------------
    ``.map(fn)``         — ajoute un transform ``Batch → Batch``.
    ``.select(pred)``    — filtre les batches.
    ``.with_epoch(n)``   — limite à n batches par époque.

    Note sur la position intra-époque
    ----------------------------------
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
        """Initialise un NodePipeline."""
        self._root      = root_node
        self._loader    = dino_loader
        self._max_steps = max_steps
        self._tn_loader: tn.Loader | None = None

    # ------------------------------------------------------------------
    # API de composition fluide
    # ------------------------------------------------------------------

    def map(self, fn: Callable[[Batch], Batch], *, label: str = "<map>") -> "NodePipeline":
        """Ajoute un transform ``Batch → Batch``.

        Args:
            fn:    Transform appliqué à chaque batch.
            label: Label optionnel pour le débogage.

        Returns:
            Nouveau ``NodePipeline`` avec ``fn`` ajouté.

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
        """Ignore les batches pour lesquels ``predicate(batch)`` est ``False``.

        Args:
            predicate: Retourne ``True`` pour garder un batch.
            label:     Label optionnel pour le débogage.

        Returns:
            Nouveau ``NodePipeline`` avec le filtre ajouté.

        """
        return NodePipeline(
            root_node   = BatchFilterNode(self._root, predicate, label=label),
            dino_loader = self._loader,
            max_steps   = self._max_steps,
        )

    def with_epoch(self, n_steps: int) -> "NodePipeline":
        """Limite l'itération à ``n_steps`` batches par époque.

        Args:
            n_steps: Nombre maximum de batches avant ``StopIteration``.

        Returns:
            Nouveau ``NodePipeline`` borné à ``n_steps``.

        """
        return NodePipeline(
            root_node   = self._root,
            dino_loader = self._loader,
            max_steps   = n_steps,
        )

    # ------------------------------------------------------------------
    # Itération
    # ------------------------------------------------------------------

    def _build_tn_loader(self) -> tn.Loader:
        """Construit le ``tn.Loader`` sous-jacent, en ajoutant un _LimitNode si besoin."""
        root = self._root
        if self._max_steps is not None:
            root = _LimitNode(root, self._max_steps)
        return tn.Loader(root, restart_on_stop_iteration=True)

    def __iter__(self) -> Iterator[Batch]:
        """Itère sur les batches, en construisant le tn.Loader paresseusement."""
        if self._tn_loader is None:
            self._tn_loader = self._build_tn_loader()
        return iter(self._tn_loader)

    def __len__(self) -> int:
        """Retourne max_steps si défini, sinon délègue au loader sous-jacent."""
        if self._max_steps is not None:
            return self._max_steps
        return len(self._loader)

    # ------------------------------------------------------------------
    # Gestion d'état
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Retourne le state dict complet du graphe.

        [FIX-STATE-MAX-STEPS] max_steps est inclus pour qu'un round-trip via
        ``load_state_dict()`` + ``with_epoch()`` restitue le même comportement.
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
        """Restaure le graphe depuis un state dict.

        Note : max_steps est restauré en mémoire mais ne reconstruit pas le
        ``tn.Loader``.  Appeler ``with_epoch()`` à nouveau si le pipeline a
        déjà itéré.
        """
        if "loader" in sd:
            self._loader.load_state_dict(sd["loader"])
        if "tn_graph" in sd and self._tn_loader is not None:
            self._tn_loader.load_state_dict(sd["tn_graph"])
        if "max_steps" in sd:
            self._max_steps = sd["max_steps"]

    # ------------------------------------------------------------------
    # Délégation vers DINODataLoader
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Délègue à ``DINODataLoader``."""
        self._loader.set_epoch(epoch)

    def checkpoint(self, step: int) -> None:
        """Délègue à ``DINODataLoader``."""
        self._loader.checkpoint(step)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Délègue à ``DINODataLoader``."""
        self._loader.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Délègue à ``DINODataLoader``."""
        self._loader.set_weight_by_name(name, weight)

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Délègue à ``DINODataLoader``."""
        self._loader.set_resolution(global_size, local_size)

    @property
    def current_resolution(self) -> tuple[int, int]:
        """Résolution de crop courante sous la forme ``(global_size, local_size)``."""
        return self._loader.current_resolution

    @property
    def current_weights(self) -> list[float]:
        """Poids de mixage normalisés courants."""
        return self._loader.current_weights

    @property
    def backend(self) -> Any:
        """Instance du backend actif."""
        return self._loader.backend

    @property
    def aug_spec(self) -> Any:
        """Spec d'augmentation active."""
        return self._loader.aug_spec


# ---------------------------------------------------------------------------
# wrap_loader — factory publique
# ---------------------------------------------------------------------------


def wrap_loader(dino_loader: Any) -> NodePipeline:
    """Enveloppe un ``DINODataLoader`` dans un ``NodePipeline`` composable.

    C'est le point d'entrée recommandé pour construire un pipeline de
    post-traitement.  Utilisé en interne par ``DINODataLoader.as_pipeline()``.

    Args:
        dino_loader: Instance de ``DINODataLoader``.  Doit exposer un attribut
            ``_dali_node`` de type ``_DALINode``.

    Returns:
        ``NodePipeline`` — itérable directement ou composable via
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
            f"wrap_loader: l'objet fourni ({type(dino_loader).__name__}) "
            "n'expose pas d'attribut '_dali_node'. "
            "Passer une instance de DINODataLoader."
        )
        raise TypeError(msg)
    return NodePipeline(root_node=dali_node, dino_loader=dino_loader)