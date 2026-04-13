"""dino_loader.pipeline_graph
============================
Pipeline de post-traitement composable et stateful, construit sur
``torchdata.nodes``.

Responsabilité unique
---------------------
Ce module gère uniquement les transformations opérant sur des ``Batch``
après la sortie du pipeline d'augmentation :

``MetadataNode``
    Sépare les flux JPEG et métadonnées issus d'un ``ShardReaderNode``.

``MaskMapNode``
    Attache des masques de patches iBOT aux ``Batch``.

``BatchMapNode``
    Applique une transformation ``Batch → Batch`` arbitraire.

``BatchFilterNode``
    Filtre les batches selon un prédicat booléen.

``_LimitNode``
    Borne l'itération à ``max_steps`` batches par époque.

``NodePipeline``
    Wrapper composable exposant les propriétés publiques de ``DINODataLoader``.

``wrap_loader``
    Factory publique : bridge ``DINODataLoader`` → ``NodePipeline``.
    Accepte tout objet itérable : si ``_dali_node`` est présent il est utilisé
    comme nœud source natif, sinon ``tn.IterableWrapper`` enveloppe l'itérable.
    Cela permet aux tests d'utiliser des faux loaders légers sans instancier
    le stack DALI complet.

Note sur ``_DALINode``
----------------------
``_DALINode`` vit dans ``dino_loader.dali_node``.  Importer directement
depuis ce module pour un accès explicite.

Séparation des responsabilités
--------------------------------
::

    dali_node.py         → pilote l'itérateur backend, assemble Batch
    pipeline_graph.py    → compose des transforms post-augmentation sur Batch
    shard_reader.py      → stages I/O + mixing (pas d'import de pipeline_graph)

Corrections intégrées
---------------------
[FIX-STATE-MAX-STEPS] max_steps inclus dans state_dict / load_state_dict.
[PROPS]               NodePipeline expose current_resolution, current_weights,
                      backend, aug_spec.
[FIX-WRAP-LOADER]     wrap_loader accepte tout itérable : _dali_node est
                      optionnel.  Les tests peuvent passer des faux loaders
                      sans instancier le stack DALI.
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch
import torchdata.nodes as tn
from torchdata.nodes import BaseNode

from dino_loader.memory import Batch

log = logging.getLogger(__name__)

__all__ = [
    "BatchFilterNode",
    "BatchMapNode",
    "MaskMapNode",
    "MetadataNode",
    "NodePipeline",
    "wrap_loader",
]


# ---------------------------------------------------------------------------
# MetadataNode
# ---------------------------------------------------------------------------


class MetadataNode(BaseNode):  # type: ignore[misc]
    """Sépare le flux JPEG et métadonnées issu d'un ``ShardReaderNode``.

    Bufferise les métadonnées pour permettre leur récupération décorrélée
    (après la passe DALI) via ``pop_last_metadata()``.

    Args:
        source: Nœud amont produisant des ``(list[np.ndarray], list[dict|None])``.

    """

    def __init__(self, source: BaseNode) -> None:  # type: ignore[type-arg]
        super().__init__()
        self._source: BaseNode                        = source  # type: ignore[type-arg]
        self._last_meta: list[dict[str, Any] | None] = []

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._last_meta = []

    def next(self) -> tuple[list[np.ndarray], list[dict[str, Any] | None]]:
        jpegs, meta     = self._source.next()
        self._last_meta = meta
        return jpegs, meta

    def get_state(self) -> dict[str, Any]:
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

    Opère sur des indices de patches ViT, pas sur des pixels — ne peut pas
    être fusionné dans le graphe DALI.  Surcoût CPU ~0,3 ms pour grille 37×37.

    Args:
        source:              Nœud amont produisant des ``Batch``.
        mask_generator:      Instance de ``MaskingGenerator``.
        num_masking_patches: Nombre cible de patches masqués.

    """

    def __init__(
        self,
        source:              BaseNode,  # type: ignore[type-arg]
        mask_generator:      Any,
        num_masking_patches: int | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._gen    = mask_generator
        self._n_mask = num_masking_patches

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self) -> Batch:
        batch: Batch = self._source.next()
        if not batch.global_crops:
            msg = (
                "MaskMapNode.next: batch.global_crops est vide. "
                "MaskMapNode requiert au moins un crop global."
            )
            raise ValueError(msg)

        mask  = self._gen(flat=True)
        masks = torch.from_numpy(mask).unsqueeze(0)
        b     = batch.global_crops[0].shape[0]
        masks = masks.expand(b, -1)
        batch.masks = masks
        return batch

    def get_state(self) -> dict[str, Any]:
        return self._source.get_state()

    @staticmethod
    def as_transform(
        mask_generator:      Any,
        num_masking_patches: int | None = None,
    ) -> Callable[[Batch], Batch]:
        """Retourne un callable ``Batch → Batch`` pour utilisation avec ``.map()``.

        Args:
            mask_generator:      Instance de ``MaskingGenerator``.
            num_masking_patches: Réservé pour validation future.

        """
        def _apply(batch: Batch) -> Batch:
            if not batch.global_crops:
                msg = "MaskMapNode.as_transform: batch.global_crops est vide."
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
    """Applique une transformation ``Batch → Batch`` à chaque élément."""

    def __init__(
        self,
        source: BaseNode,  # type: ignore[type-arg]
        fn:     Callable[[Batch], Batch],
        *,
        label:  str = "<map>",
    ) -> None:
        super().__init__()
        self._source = source
        self._fn     = fn
        self._label  = label

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)

    def next(self) -> Batch:
        return self._fn(self._source.next())

    def get_state(self) -> dict[str, Any]:
        return self._source.get_state()

    def __repr__(self) -> str:
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
        super().__init__()
        self._source    = source
        self._predicate = predicate
        self._label     = label
        self._n_skipped = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._n_skipped = 0

    def next(self) -> Batch:
        while True:
            batch = self._source.next()
            if self._predicate(batch):
                return batch
            self._n_skipped += 1
            self._record_filtered()

    def get_state(self) -> dict[str, Any]:
        return self._source.get_state()

    @property
    def n_skipped(self) -> int:
        """Nombre cumulé de batches rejetés."""
        return self._n_skipped

    def __repr__(self) -> str:
        return f"BatchFilterNode({self._label!r}, skipped={self._n_skipped})"

    @staticmethod
    def _record_filtered() -> None:
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
    """Émet au plus ``max_steps`` batches par époque."""

    def __init__(self, source: BaseNode, max_steps: int) -> None:  # type: ignore[type-arg]
        super().__init__()
        self._source    = source
        self._max_steps = max_steps
        self._yielded   = 0

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        self._source.reset(initial_state)
        self._yielded = 0

    def next(self) -> Batch:
        if self._yielded >= self._max_steps:
            raise StopIteration
        batch = self._source.next()
        self._yielded += 1
        return batch

    def get_state(self) -> dict[str, Any]:
        return self._source.get_state()


# ---------------------------------------------------------------------------
# NodePipeline
# ---------------------------------------------------------------------------


class NodePipeline:
    """Pipeline de post-traitement composable et stateful.

    Expose toutes les propriétés publiques de ``DINODataLoader`` par délégation,
    utilisable comme drop-in replacement dans la boucle d'entraînement.

    Méthodes de composition (retournent un nouveau NodePipeline)
    -------------------------------------------------------------
    ``.map(fn)``       — ajoute un transform ``Batch → Batch``.
    ``.select(pred)``  — filtre les batches.
    ``.with_epoch(n)`` — limite à n batches par époque.

    Note sur la position intra-époque
    ----------------------------------
    ``state_dict`` persiste l'époque, les poids, la résolution et max_steps.
    La position intra-époque n'est pas restaurée (DALI non sérialisable).

    [FIX-STATE-MAX-STEPS] max_steps est inclus dans state_dict pour un
    round-trip correct lors de la reprise.
    """

    def __init__(
        self,
        root_node:   BaseNode,  # type: ignore[type-arg]
        dino_loader: Any,
        max_steps:   int | None = None,
    ) -> None:
        self._root      = root_node
        self._loader    = dino_loader
        self._max_steps = max_steps
        self._tn_loader: tn.Loader | None = None

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def map(self, fn: Callable[[Batch], Batch], *, label: str = "<map>") -> "NodePipeline":
        """Ajoute un transform ``Batch → Batch``."""
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
        """Ignore les batches pour lesquels ``predicate(batch)`` est False."""
        return NodePipeline(
            root_node   = BatchFilterNode(self._root, predicate, label=label),
            dino_loader = self._loader,
            max_steps   = self._max_steps,
        )

    def with_epoch(self, n_steps: int) -> "NodePipeline":
        """Limite l'itération à ``n_steps`` batches par époque."""
        return NodePipeline(
            root_node   = self._root,
            dino_loader = self._loader,
            max_steps   = n_steps,
        )

    # ------------------------------------------------------------------
    # Itération
    # ------------------------------------------------------------------

    def _build_tn_loader(self) -> tn.Loader:
        root = self._root
        if self._max_steps is not None:
            root = _LimitNode(root, self._max_steps)
        return tn.Loader(root, restart_on_stop_iteration=True)

    def __iter__(self) -> Iterator[Batch]:
        if self._tn_loader is None:
            self._tn_loader = self._build_tn_loader()
        return iter(self._tn_loader)

    def __len__(self) -> int:
        if self._max_steps is not None:
            return self._max_steps
        return len(self._loader)

    # ------------------------------------------------------------------
    # État
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Retourne le state dict complet du graphe.

        [FIX-STATE-MAX-STEPS] max_steps est inclus pour un round-trip correct.
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
        """Restaure le graphe depuis un state dict."""
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
        self._loader.set_epoch(epoch)

    def checkpoint(self, step: int) -> None:
        self._loader.checkpoint(step)

    def set_weights(self, weights: Sequence[float]) -> None:
        self._loader.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._loader.set_weight_by_name(name, weight)

    def set_resolution(self, global_size: int, local_size: int) -> None:
        self._loader.set_resolution(global_size, local_size)

    @property
    def current_resolution(self) -> tuple[int, int]:
        return self._loader.current_resolution

    @property
    def current_weights(self) -> list[float]:
        return self._loader.current_weights

    @property
    def backend(self) -> Any:
        return self._loader.backend

    @property
    def aug_spec(self) -> Any:
        return self._loader.aug_spec


# ---------------------------------------------------------------------------
# wrap_loader — factory publique
# ---------------------------------------------------------------------------


def wrap_loader(dino_loader: Any) -> NodePipeline:
    """Enveloppe un ``DINODataLoader`` dans un ``NodePipeline`` composable.

    Point d'entrée recommandé pour construire un pipeline de post-traitement.
    Utilisé en interne par ``DINODataLoader.as_pipeline()``.

    ``_dali_node`` est utilisé comme source native quand il est présent
    (production).  En son absence, ``tn.IterableWrapper`` enveloppe
    ``dino_loader`` directement, ce qui permet aux tests d'utiliser des
    faux loaders légers sans instancier le stack DALI complet.

    Args:
        dino_loader: Instance de ``DINODataLoader`` ou tout itérable de ``Batch``.

    Example::

        pipeline = (
            wrap_loader(DINODataLoader(...))
            .map(apply_ibot_masks)
            .select(lambda b: any(m is not None for m in b.metadata))
            .with_epoch(steps_per_epoch)
        )

    """
    dali_node = getattr(dino_loader, "_dali_node", None)
    if dali_node is not None:
        root: BaseNode = dali_node  # type: ignore[type-arg]
    else:
        # Fallback for test fakes and any iterable loader.
        root = tn.IterableWrapper(iter(dino_loader))

    return NodePipeline(root_node=root, dino_loader=dino_loader)