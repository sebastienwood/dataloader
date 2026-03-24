"""
dino_loader.augmentation
========================
Augmentation pipeline abstraction for dino_loader.

Architecture
------------
``AugmentationSpec`` is the configuration object describing the augmentation
strategy. ``AugmentationPipeline`` is the runtime object built from it.

The clean separation enables preset strategies (``DinoV2AugSpec``,
``EvalAugSpec``, ``LeJEPAAugSpec``) and user-defined strategies
(``UserAugSpec``) without modifying the loader or backend.

Early filtering
---------------
``SamplePredicate`` is called by ``ShardIterator`` before a sample enters
the DALI pipeline, eliminating GPU decode cost for rejected samples.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from dino_loader.config import DINOAugConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleMeta:
    """Lightweight sample descriptor available before JPEG decoding.

    Passed to SamplePredicate callables so filtering can happen at zero
    image-decode cost.

    Attributes:
        key: WebDataset sample key (e.g. "000042").
        shard_path: Absolute path to the .tar shard file.
        metadata: Parsed JSON sidecar dict, or None if absent.
    """

    key:        str
    shard_path: str
    metadata:   dict[str, Any] | None


@runtime_checkable
class SamplePredicate(Protocol):
    """Callable protocol for early sample filtering.

    Return True to keep the sample, False to discard it before JPEG decode.
    Must be thread-safe (called from extraction worker threads).

    Example — keep only samples with quality_score ≥ 0.5::

        def quality_filter(meta: SampleMeta) -> bool:
            if meta.metadata is None:
                return True
            return meta.metadata.get("quality_score", 1.0) >= 0.5
    """

    def __call__(self, meta: SampleMeta) -> bool: ...


@runtime_checkable
class AugmentationPipeline(Protocol):
    """Runtime augmentation pipeline consumed by loader backends."""

    @property
    def output_map(self) -> list[str]: ...

    def __iter__(self) -> "AugmentationPipeline": ...

    def __next__(self) -> dict[str, Any]: ...

    def reset(self) -> None: ...


class AugmentationSpec(ABC):
    """Base class for augmentation strategy specifications.

    A spec is a pure configuration object with no runtime state. The backend's
    build_aug_pipeline() factory turns it into a live AugmentationPipeline.
    """

    @property
    @abstractmethod
    def output_map(self) -> list[str]: ...

    @property
    def n_views(self) -> int:
        return len(self.output_map)

    @property
    def uses_dali(self) -> bool:
        """True if this spec can be fully fused into a DALI pipeline graph."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_views={self.n_views})"


@dataclass
class DinoV2AugSpec(AugmentationSpec):
    """DINOv2-style multi-crop augmentation.

    Default spec used when DINODataLoader is constructed without an explicit
    aug_spec argument. Wraps the existing DINOAugConfig for backward
    compatibility.

    Attributes:
        aug_cfg: Full DINOv2 augmentation configuration.
        fuse_normalization: Fuse per-dataset mean/std into the DALI graph.
        fp8_output: Emit FP8-cast tensors directly from the DALI graph.
    """

    aug_cfg:            DINOAugConfig = field(default_factory=DINOAugConfig)
    fuse_normalization: bool          = True
    fp8_output:         bool          = False

    @property
    def output_map(self) -> list[str]:
        return [f"view_{i}" for i in range(self.aug_cfg.n_views)]


@dataclass
class EvalAugSpec(AugmentationSpec):
    """Evaluation augmentation: resize-then-centre-crop, no stochastic ops.

    Suitable for val/test loops and fine-tuning phases.

    Attributes:
        crop_size: Output spatial resolution in pixels.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
        interpolation: Resize interpolation mode ("bicubic" or "bilinear").
    """

    crop_size:     int                        = 224
    mean:          tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:           tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str                        = "bicubic"

    @property
    def output_map(self) -> list[str]:
        return ["view_0"]

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
    """LeJEPA-style augmentation: context + target patch views.

    Produces two views per sample:
    - ``context``: large crop (encoder input).
    - ``target_N``: smaller crops (predictor targets).

    Patch masking required by JEPA is applied after the pipeline on CPU using
    MaskingGenerator — identical to DINOv2 iBOT masks (DALI cannot express
    patch-index operations).

    Attributes:
        context_crop_size: Context view spatial resolution.
        target_crop_size: Target view spatial resolution.
        n_target_views: Number of independent target crops per sample.
        context_scale: RandomResizedCrop scale range for the context view.
        target_scale: RandomResizedCrop scale range for target views.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
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


# Type alias for user-provided augmentation functions.
UserAugFn = Callable[
    ["torch.Tensor"],          # noqa: F821
    "dict[str, torch.Tensor]", # noqa: F821
]


@dataclass
class UserAugSpec(AugmentationSpec):
    """User-provided augmentation function applied to decoded GPU tensors.

    JPEG decoding is always performed by DALI's hardware nvjpeg pipeline.
    The user function receives already-decoded float16 tensors on GPU
    (shape [B, C, H, W]) and never sees raw bytes.

    Attributes:
        aug_fn: Callable (Tensor[B,C,H,W]) → dict[str, Tensor[B,C,H,W]].
            Input tensor is decoded, normalised to [0,1] float16.
            Keys must match output_map.
        output_map: Names of views returned by aug_fn.
        decode_size: Resolution at which DALI decodes before calling aug_fn.
            Should be the largest crop size your function may produce.
        mean: Per-channel normalisation mean applied before calling aug_fn.
        std: Per-channel normalisation std.
        warn_not_dali: Emit a non-DALI performance warning (default True).
    """

    aug_fn:       UserAugFn
    output_map:   list[str]
    decode_size:  int                         = 256
    mean:         tuple[float, float, float]  = (0.485, 0.456, 0.406)
    std:          tuple[float, float, float]  = (0.229, 0.224, 0.225)
    warn_not_dali: bool                       = True

    @property
    def uses_dali(self) -> bool:
        return False

    @property
    def n_views(self) -> int:
        return len(self.output_map)

    def __post_init__(self) -> None:
        if not callable(self.aug_fn):
            raise TypeError("UserAugSpec.aug_fn must be callable.")
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
