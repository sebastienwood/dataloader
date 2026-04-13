"""dino_loader.augmentation
========================
Augmentation pipeline abstraction for dino_loader.

Architecture
------------
``AugmentationSpec`` est le pur objet de configuration décrivant la stratégie
d'augmentation.  ``AugmentationPipeline`` est l'objet runtime construit à
partir de celui-ci.

La séparation nette permet des stratégies prédéfinies (``DinoV2AugSpec``,
``EvalAugSpec``, ``LeJEPAAugSpec``) et des stratégies utilisateur
(``UserAugSpec``) sans modifier le loader ou les backends.

Types partagés sources/augmentation
-------------------------------------
``SampleRecord``, ``SampleMeta`` et ``SamplePredicate`` sont définis ici car
ils constituent le contrat entre le stage de lecture (sources) et le stage de
filtrage/augmentation.  Les sources produisent des ``SampleRecord`` ; les
prédicats les évaluent via ``SampleMeta`` avant tout décodage JPEG.

Responsabilité étendue des specs
----------------------------------
Chaque spec connaît comment :

- Nommer ses vues (``output_map``).
- Déclarer ses dimensions initiales de crop (``initial_global_size``,
  ``initial_local_size``).
- Séparer une liste plate de vues en ``(global_crops, local_crops)``
  (``split_views``).
- Déclarer si elle supporte le masquage iBOT (``supports_masking``).

Cela concentre le dispatch par sous-type dans les specs elles-mêmes,
éliminant les chaînes ``isinstance`` répétées dans ``loader.py``.

Normalisation
-------------
Toutes les sous-classes exposent ``norm_stats`` en échelle [0, 1].  Les
consommateurs (``pipeline.py``, ``cpu.py``, ``dynamic_pipeline.py``) appellent
``norm_stats.to_dali_scale()`` — aucune multiplication ad-hoc par 255.

Corrections
-----------
[FIX-ISINSTANCE] Ajout de ``supports_masking`` sur ``AugmentationSpec`` pour
    éliminer le ``isinstance(aug_spec, DinoV2AugSpec)`` dans ``loader.py``.
    Seul ``DinoV2AugSpec`` retourne ``True`` (le masquage iBOT ne s'applique
    qu'aux crops multi-vues DINOv2).  Les autres specs retournent ``False``.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dino_loader.config import DINOAugConfig, NormStats

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types partagés : contrat source → filtrage → augmentation
# ---------------------------------------------------------------------------


class SampleRecord:
    """Sample décodé prêt pour le pipeline d'augmentation.

    Produit par les sources (``hpc_source``, ``wds_source``) après extraction
    depuis un shard WebDataset.  Consommé par les ``ShardIterator`` et filtré
    via ``SamplePredicate`` avant d'entrer dans le pipeline DALI.

    Attributes:
        jpeg: Bytes JPEG bruts.
        metadata: Dict JSON sidecar, ou ``None`` si absent.
        key: Clé WebDataset (e.g. ``"sample_000042"``).

    """

    __slots__ = ("jpeg", "key", "metadata")

    def __init__(
        self,
        jpeg:     bytes,
        metadata: dict | None = None,
        key:      str         = "",
    ) -> None:
        self.jpeg     = jpeg
        self.metadata = metadata
        self.key      = key


@dataclass(frozen=True)
class SampleMeta:
    """Descripteur léger disponible avant le décodage JPEG.

    Passé aux ``SamplePredicate`` pour que le filtrage se fasse sans décodage.

    Attributes:
        key:        Clé WebDataset (e.g. ``"000042"``).
        shard_path: Chemin absolu vers le fichier ``.tar``.
        metadata:   Dict JSON sidecar, ou ``None`` si absent.

    """

    key:        str
    shard_path: str
    metadata:   dict[str, Any] | None


@runtime_checkable
class SamplePredicate(Protocol):
    """Callable protocol pour le filtrage anticipé des samples.

    Retourner ``True`` pour garder le sample, ``False`` pour le rejeter avant
    le décodage JPEG.  Doit être thread-safe (appelé depuis les workers
    d'extraction).

    Example::

        def quality_filter(meta: SampleMeta) -> bool:
            if meta.metadata is None:
                return True
            return meta.metadata.get("quality_score", 1.0) >= 0.5

    """

    def __call__(self, meta: SampleMeta) -> bool: ...


# ---------------------------------------------------------------------------
# Runtime pipeline protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AugmentationPipeline(Protocol):
    """Pipeline d'augmentation runtime consommé par les backends."""

    @property
    def output_map(self) -> list[str]: ...

    def __iter__(self) -> "AugmentationPipeline": ...

    def __next__(self) -> dict[str, Any]: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# AugmentationSpec — base
# ---------------------------------------------------------------------------


class AugmentationSpec(ABC):
    """Classe de base pour les specs de stratégie d'augmentation.

    Une spec est un pur objet de configuration sans état runtime.  La factory
    ``BackendProtocol.build_pipeline`` la transforme en pipeline live.

    Chaque sous-classe doit implémenter :
    - ``output_map`` — noms ordonnés des vues produites.
    - ``norm_stats`` — statistiques de normalisation en [0, 1].
    - ``initial_global_size`` — taille initiale du crop global.
    - ``initial_local_size`` — taille initiale du crop local.
    - ``split_views`` — sépare une liste plate de vues en (global, local).

    Et peut surcharger :
    - ``supports_masking`` — True si le masquage iBOT est applicable.
      Par défaut ``False`` ; seul ``DinoV2AugSpec`` retourne ``True``.

    """

    @property
    @abstractmethod
    def output_map(self) -> list[str]: ...

    @property
    @abstractmethod
    def norm_stats(self) -> NormStats:
        """Statistiques de normalisation en [0, 1]."""
        ...

    @property
    @abstractmethod
    def initial_global_size(self) -> int:
        """Taille initiale du crop global en pixels."""
        ...

    @property
    @abstractmethod
    def initial_local_size(self) -> int:
        """Taille initiale du crop local en pixels."""
        ...

    @abstractmethod
    def split_views(
        self,
        views: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Sépare une liste plate de vues en ``(global_crops, local_crops)``.

        Args:
            views: Liste de tenseurs issus du pipeline d'augmentation, dans
                l'ordre défini par ``output_map``.

        Returns:
            Tuple ``(global_crops, local_crops)``.

        """
        ...

    @property
    def supports_masking(self) -> bool:
        """True si cette spec supporte le masquage de patches iBOT.

        [FIX-ISINSTANCE] Propriété polymorphe qui évite le isinstance dans
        loader.py.  Seul DinoV2AugSpec retourne True — le masquage iBOT ne
        fait sens qu'avec des crops multi-vues au format DINOv2.

        Retourne False par défaut pour toutes les autres specs.
        """
        return False

    @property
    def n_views(self) -> int:
        """Nombre total de vues de sortie."""
        return len(self.output_map)

    @property
    def uses_dali(self) -> bool:
        """True si cette spec peut être fusionnée dans un graphe DALI."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_views={self.n_views})"


# ---------------------------------------------------------------------------
# Specs concrètes
# ---------------------------------------------------------------------------


@dataclass
class DinoV2AugSpec(AugmentationSpec):
    """Augmentation multi-crop style DINOv2.

    Spec par défaut quand ``DINODataLoader`` est construit sans ``aug_spec``
    explicite.  Enveloppe ``DINOAugConfig`` pour la rétrocompatibilité.

    Attributes:
        aug_cfg:            Configuration complète DINOv2.
        fuse_normalization: Fusionner mean/std par-dataset dans le graphe DALI.
        fp8_output:         Émettre des tenseurs FP8 directement depuis DALI.

    """

    aug_cfg:            DINOAugConfig = field(default_factory=DINOAugConfig)
    fuse_normalization: bool          = True
    fp8_output:         bool          = False

    @property
    def output_map(self) -> list[str]:
        return [f"view_{i}" for i in range(self.aug_cfg.n_views)]

    @property
    def norm_stats(self) -> NormStats:
        return self.aug_cfg.norm_stats

    @property
    def initial_global_size(self) -> int:
        return self.aug_cfg.global_crop_size

    @property
    def initial_local_size(self) -> int:
        return self.aug_cfg.local_crop_size

    @property
    def supports_masking(self) -> bool:
        """DinoV2AugSpec supporte le masquage iBOT (patch masking)."""
        return True

    def split_views(self, views: list[Any]) -> tuple[list[Any], list[Any]]:
        n = self.aug_cfg.n_global_crops
        return views[:n], views[n:]


@dataclass
class EvalAugSpec(AugmentationSpec):
    """Augmentation d'évaluation : resize + centre-crop, sans stochastique.

    Attributes:
        crop_size:     Résolution de sortie en pixels.
        mean:          Moyenne de normalisation par canal en [0, 1].
        std:           Écart-type de normalisation par canal en [0, 1].
        interpolation: Mode de redimensionnement (``"bicubic"`` ou ``"bilinear"``).

    """

    crop_size:     int                        = 224
    mean:          tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:           tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str                        = "bicubic"

    @property
    def output_map(self) -> list[str]:
        return ["view_0"]

    @property
    def norm_stats(self) -> NormStats:
        return NormStats(mean=self.mean, std=self.std)

    @property
    def initial_global_size(self) -> int:
        return self.crop_size

    @property
    def initial_local_size(self) -> int:
        return self.crop_size

    def split_views(self, views: list[Any]) -> tuple[list[Any], list[Any]]:
        return views, []

    def __post_init__(self) -> None:
        valid = {"bicubic", "bilinear"}
        if self.interpolation not in valid:
            msg = (
                f"EvalAugSpec.interpolation must be one of {valid}, "
                f"got {self.interpolation!r}."
            )
            raise ValueError(msg)


@dataclass
class LeJEPAAugSpec(AugmentationSpec):
    """Augmentation LeJEPA : crop contexte + crops cibles.

    Produit ``1 + n_target_views`` vues par sample :
    - ``context`` : grand crop (entrée encodeur).
    - ``target_N`` : petits crops (cibles du prédicteur).

    Attributes:
        context_crop_size: Résolution du crop contexte.
        target_crop_size:  Résolution des crops cibles.
        n_target_views:    Nombre de crops cibles indépendants par sample.
        context_scale:     Plage de scale RandomResizedCrop pour le contexte.
        target_scale:      Plage de scale pour les cibles.
        mean:              Moyenne de normalisation par canal en [0, 1].
        std:               Écart-type par canal en [0, 1].

    """

    context_crop_size: int                        = 224
    target_crop_size:  int                        = 96
    n_target_views:    int                        = 4
    context_scale:     tuple[float, float]        = (0.85, 1.0)
    target_scale:      tuple[float, float]        = (0.15, 0.30)
    mean:              tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:               tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def output_map(self) -> list[str]:
        return ["context", *[f"target_{i}" for i in range(self.n_target_views)]]

    @property
    def norm_stats(self) -> NormStats:
        return NormStats(mean=self.mean, std=self.std)

    @property
    def initial_global_size(self) -> int:
        return self.context_crop_size

    @property
    def initial_local_size(self) -> int:
        return self.target_crop_size

    def split_views(self, views: list[Any]) -> tuple[list[Any], list[Any]]:
        return [views[0]], views[1:]

    def __post_init__(self) -> None:
        if self.n_target_views < 1:
            msg = f"LeJEPAAugSpec.n_target_views must be ≥ 1, got {self.n_target_views}."
            raise ValueError(msg)
        if not (0.0 < self.context_scale[0] <= self.context_scale[1] <= 1.0):
            msg = f"LeJEPAAugSpec.context_scale must be in (0, 1], got {self.context_scale}."
            raise ValueError(msg)
        if not (0.0 < self.target_scale[0] <= self.target_scale[1] <= 1.0):
            msg = f"LeJEPAAugSpec.target_scale must be in (0, 1], got {self.target_scale}."
            raise ValueError(msg)


# Type alias pour les fonctions d'augmentation utilisateur.
UserAugFn = Callable[
    ["torch.Tensor"],           # noqa: F821
    "dict[str, torch.Tensor]",  # noqa: F821
]


@dataclass
class UserAugSpec(AugmentationSpec):
    """Fonction d'augmentation utilisateur appliquée aux tenseurs GPU décodés.

    Le décodage JPEG est toujours effectué par le pipeline DALI nvjpeg.  La
    fonction utilisateur reçoit des tenseurs float16 déjà décodés sur GPU.

    Attributes:
        aug_fn:        ``(Tensor[B,C,H,W]) → dict[str, Tensor[B,C,H,W]]``.
        _output_map:    Noms des vues retournées par ``aug_fn``.
        decode_size:   Résolution à laquelle DALI décode avant d'appeler
                       ``aug_fn``.  Doit être la plus grande taille produite.
        mean:          Moyenne de normalisation en [0, 1] appliquée avant.
        std:           Écart-type en [0, 1].
        warn_not_dali: Émettre un warning de performance (défaut True).

    """

    aug_fn:        UserAugFn
    _output_map:    list[str]
    decode_size:   int                         = 256
    mean:          tuple[float, float, float]  = (0.485, 0.456, 0.406)
    std:           tuple[float, float, float]  = (0.229, 0.224, 0.225)
    warn_not_dali: bool                        = True

    @property
    def uses_dali(self) -> bool:
        return False

    @property
    def output_map(self) -> list[str]:
        return self._output_map

    @property
    def n_views(self) -> int:
        return len(self.output_map)

    @property
    def norm_stats(self) -> NormStats:
        return NormStats(mean=self.mean, std=self.std)

    @property
    def initial_global_size(self) -> int:
        return self.decode_size

    @property
    def initial_local_size(self) -> int:
        return self.decode_size

    def split_views(self, views: list[Any]) -> tuple[list[Any], list[Any]]:
        mid = max(1, len(views) // 2)
        return views[:mid], views[mid:]

    def __post_init__(self) -> None:
        if not callable(self.aug_fn):
            msg = "UserAugSpec.aug_fn must be callable."
            raise TypeError(msg)
        if not self.output_map:
            msg = "UserAugSpec.output_map must be a non-empty list of view names."
            raise ValueError(msg)
        if self.decode_size < 1:
            msg = f"UserAugSpec.decode_size must be ≥ 1, got {self.decode_size}."
            raise ValueError(msg)

        if self.warn_not_dali:
            warnings.warn(
                "UserAugSpec: aug_fn runs outside the DALI computation graph. "
                "JPEG decoding still uses the nvjpeg hardware pipeline, but "
                "augmentation ops cannot be fused with decode. "
                "Expect ~10–20% throughput reduction vs. a native DALI pipeline. "
                "Suppress with warn_not_dali=False once acknowledged.",
                UserWarning,
                stacklevel=3,
            )