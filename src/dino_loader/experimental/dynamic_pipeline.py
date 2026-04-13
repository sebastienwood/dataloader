"""dino_loader.experimental.dynamic_pipeline
============================================
DALI v2 **dynamic mode** augmentation pipeline for DINOv2-style multi-crop.

Background
----------
The production pipeline in ``dino_loader/pipeline.py`` uses DALI's static
graph API (``@pipeline_def`` + ``fn.*`` operators).  This requires:

- ``ExternalSource`` callbacks for runtime-variable state (resolution,
  per-dataset mean/std).
- Four separate ``@pipeline_def`` functions for the four ``AugmentationSpec``
  subtypes.
- A compile step at construction time that inlines all topology decisions.

DALI's experimental *dynamic mode* (``nvidia.dali.experimental.dynamic``)
lets you write the augmentation as **imperative Python** that DALI
JIT-compiles per batch.  This eliminates every ``ExternalSource`` callback
and allows natural Python branching for spec dispatch.

Key differences from the static pipeline
-----------------------------------------
- ``NormSource`` and ``ResolutionSource`` are gone; values are read directly
  from the Python closure at call time.
- Resolution changes take effect on the **next call** without any inter-thread
  signalling.
- Per-sample normalisation is handled via a per-sample index lookup inside the
  batch function, matching the ``NormSource`` behaviour of the static pipeline.
- The four ``@pipeline_def`` dispatch paths collapse into four plain Python
  branches in a single function.

Augmentation randomness — important design note
------------------------------------------------
All stochastic parameters (crop scale, colour jitter, blur sigma, …) are
driven by ``ndd.random.*`` operators, **not** by Python ``random`` or
``numpy.random``.  This is critical: Python RNG calls inside a dynamic batch
function return a *single scalar* shared by every sample in the batch, making
all samples augmented identically.  ``ndd.random.*`` produces one independent
draw per sample in the batch, preserving the per-instance diversity required
by contrastive self-supervised learning.

Per-sample normalisation
------------------------
Unlike the naive approach of using the first sample's dataset index for the
whole batch, the batch function performs a proper per-sample lookup: for each
sample ``i`` it selects ``norm_table[ds_indices[i]]`` and stacks them into
batch mean/std tensors that ``ndd.crop_mirror_normalize`` consumes natively.
This matches the ``NormSource`` semantics of the static pipeline.

Limitations
-----------
- Requires DALI ≥ 1.40 with ``nvidia.dali.experimental.dynamic`` available.
- Kernel fusion guarantees (``normalize → cast → transpose`` in one kernel)
  may differ from the static graph; measure with Nsight before committing.
- The API is marked ``experimental`` by NVIDIA and may change.
- ``dali_fp8_output`` is not yet supported in this path.

See ``scripts/benchmark.py`` for a head-to-head throughput comparison.

Public API
----------
::

    from dino_loader.experimental.dynamic_pipeline import (
        DynamicDINOPipeline,
        build_dynamic_pipeline,
    )

    pipeline = build_dynamic_pipeline(
        aug_spec       = DinoV2AugSpec(aug_cfg=DINOAugConfig()),
        batch_size     = 512,
        device_id      = 0,
        source         = mixing_source,
        specs          = dataset_specs,
    )

    # Use exactly like the static pipeline:
    for dali_out in pipeline:
        views = [dali_out[0][name] for name in aug_spec.output_map]
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from dino_datasets import DatasetSpec

from dino_loader.augmentation import (
    AugmentationSpec,
    DinoV2AugSpec,
    EvalAugSpec,
    LeJEPAAugSpec,
    UserAugSpec,
)
from dino_loader.config import DINOAugConfig, NormStats

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import nvidia.dali.experimental.dynamic as ndd
    from nvidia.dali import types

    _HAS_DYNAMIC_DALI = True
except ImportError:
    _HAS_DYNAMIC_DALI = False
    ndd   = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


def _require_dynamic_dali() -> None:
    """Raise ImportError with a helpful message if dynamic DALI is unavailable."""
    if not _HAS_DYNAMIC_DALI:
        msg = (
            "nvidia.dali.experimental.dynamic is required for the dynamic pipeline.\n"
            "Install DALI ≥ 1.40: pip install nvidia-dali-cuda120\n"
            "Note: dynamic mode is experimental and may not be available in all builds."
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Per-dataset normalisation lookup table
# ---------------------------------------------------------------------------


def _build_norm_table(
    aug_cfg: DINOAugConfig,
    specs:   list[DatasetSpec],
) -> list[NormStats]:
    """Build a lookup table of per-dataset normalisation statistics.

    Falls back to the global ``aug_cfg`` norm stats for datasets that have
    no per-dataset override.

    Args:
        aug_cfg: Global augmentation config (provides fallback mean/std).
        specs: Dataset specifications (may carry per-dataset mean/std).

    Returns:
        List of ``NormStats`` aligned with ``specs`` (index ``i`` →
        ``specs[i]``).  Stats are stored in [0, 1] scale.

    """
    global_stats = aug_cfg.norm_stats
    return [
        NormStats.from_config(mean=spec.mean, std=spec.std, fallback=global_stats)
        for spec in specs
    ]


# ---------------------------------------------------------------------------
# Resolution holder
# ---------------------------------------------------------------------------


class _ResolutionHolder:
    """Thread-safe holder for the current crop resolution.

    Unlike ``ResolutionSource`` (which is a DALI ``ExternalSource``
    callback), this is a plain Python object read directly inside the
    dynamic batch function — no inter-thread DALI signalling required.

    Args:
        global_size: Initial global crop size in pixels.
        local_size: Initial local crop size in pixels.

    """

    def __init__(self, global_size: int, local_size: int) -> None:
        """Initialise the resolution holder."""
        self._global = global_size
        self._local  = local_size
        self._lock   = threading.Lock()

    def set(self, global_size: int, local_size: int) -> None:
        """Update both dimensions atomically.

        Args:
            global_size: New global crop size in pixels.
            local_size: New local crop size in pixels.

        """
        with self._lock:
            self._global = global_size
            self._local  = local_size

    @property
    def global_size(self) -> int:
        """Current global crop size (thread-safe)."""
        with self._lock:
            return self._global

    @property
    def local_size(self) -> int:
        """Current local crop size (thread-safe)."""
        with self._lock:
            return self._local


# ---------------------------------------------------------------------------
# Dynamic augmentation functions
# ---------------------------------------------------------------------------


def _make_dinov2_aug_fn(
    aug_cfg:    DINOAugConfig,
    resolution: _ResolutionHolder,
    norm_table: list[NormStats],
) -> Any:
    """Return a dynamic-mode batch function implementing DINOv2 multi-crop.

    All stochastic parameters are driven by ``ndd.random.*`` operators to
    produce **per-sample** independent draws within each batch.  Using Python
    ``random`` or ``numpy.random`` here would produce a single scalar shared
    by every sample, making all samples in the batch augmented identically —
    which would destroy the contrastive diversity required by DINOv3.

    Per-sample normalisation uses a full per-sample index lookup (not just the
    first sample's index), matching the ``NormSource`` semantics of the static
    pipeline.

    Args:
        aug_cfg: Augmentation configuration.
        resolution: Thread-safe resolution holder (updated by ``set_resolution``).
        norm_table: Per-dataset normalisation statistics in [0, 1] scale.

    Returns:
        Callable compatible with the dynamic pipeline iteration protocol.

    """
    _require_dynamic_dali()

    n_global = aug_cfg.n_global_crops
    n_local  = aug_cfg.n_local_crops

    # Pre-convert norm table to [0, 255] scale once at construction time
    # rather than on every batch call.
    norm_table_255: list[tuple[list[float], list[float]]] = [
        stats.to_dali_scale() for stats in norm_table
    ]
    # Fallback for batches with no dataset index information.
    fallback_mean_255, fallback_std_255 = aug_cfg.norm_stats.to_dali_scale()

    def _aug_fn(jpegs: ndd.Batch, ds_indices: list[int]) -> dict[str, ndd.Batch]:
        global_size = resolution.global_size
        local_size  = resolution.local_size

        # Decode once; subsequent crops share the decoded buffer.
        decoded = ndd.decoders.image(jpegs, device="gpu", output_type=types.RGB)

        # Build per-sample mean/std arrays for batch normalisation.
        # Each sample gets the stats of its own dataset — not a shared scalar.
        # This matches the NormSource semantics from the static pipeline.
        if ds_indices:
            batch_means = np.stack([
                np.array(norm_table_255[min(idx, len(norm_table_255) - 1)][0],
                         dtype=np.float32)
                for idx in ds_indices
            ])  # shape (B, 3)
            batch_stds = np.stack([
                np.array(norm_table_255[min(idx, len(norm_table_255) - 1)][1],
                         dtype=np.float32)
                for idx in ds_indices
            ])  # shape (B, 3)
        else:
            # No index information: use global fallback for all samples.
            b = len(jpegs)
            batch_means = np.tile(
                np.array(fallback_mean_255, dtype=np.float32), (b, 1),
            )
            batch_stds = np.tile(
                np.array(fallback_std_255,  dtype=np.float32), (b, 1),
            )

        views: dict[str, ndd.Batch] = {}
        view_idx = 0

        for i in range(n_global):
            is_first  = i == 0
            blur_prob = aug_cfg.blur_prob_global1 if is_first else aug_cfg.blur_prob_global2
            sol_prob  = aug_cfg.solarize_prob if (i == 1) else 0.0

            # ndd.random.* produces one independent draw per sample in the batch.
            crop = ndd.random_resized_crop(
                decoded,
                size                = global_size,
                random_area         = aug_cfg.global_crops_scale,
                random_aspect_ratio = (3 / 4, 4 / 3),
                device              = "gpu",
            )

            # Per-sample colour jitter — ndd.random.uniform draws B independent values.
            crop = ndd.color_twist(
                crop,
                brightness = ndd.random.uniform(range=(0.6, 1.4)),
                contrast   = ndd.random.uniform(range=(0.6, 1.4)),
                saturation = ndd.random.uniform(range=(0.6, 1.4)),
                hue        = ndd.random.uniform(range=(-0.1, 0.1)),
            )
            crop = ndd.flip(crop, horizontal=ndd.random.coin_flip(), vertical=0)

            # Gaussian blur — applied only to samples where coin_flip fires.
            blur_mask = ndd.random.coin_flip(probability=blur_prob)
            blurred   = ndd.gaussian_blur(
                crop,
                sigma=ndd.random.uniform(range=(aug_cfg.blur_sigma_min,
                                                aug_cfg.blur_sigma_max)),
            )
            crop = blur_mask * blurred + (1 - blur_mask) * crop

            # Solarisation — second global crop only.
            if sol_prob > 0.0:
                sol_mask = ndd.random.coin_flip(probability=sol_prob)
                solarised = ndd.solarize(crop, threshold=128)
                crop = sol_mask * solarised + (1 - sol_mask) * crop

            crop = ndd.crop_mirror_normalize(
                crop,
                dtype         = types.FLOAT16,
                output_layout = "CHW",
                mean          = batch_means,
                std           = batch_stds,
            )
            views[f"view_{view_idx}"] = crop
            view_idx += 1

        for _ in range(n_local):
            crop = ndd.random_resized_crop(
                decoded,
                size                = local_size,
                random_area         = aug_cfg.local_crops_scale,
                random_aspect_ratio = (3 / 4, 4 / 3),
                device              = "gpu",
            )
            crop = ndd.color_twist(
                crop,
                brightness = ndd.random.uniform(range=(0.6, 1.4)),
                contrast   = ndd.random.uniform(range=(0.6, 1.4)),
                saturation = ndd.random.uniform(range=(0.6, 1.4)),
                hue        = ndd.random.uniform(range=(-0.1, 0.1)),
            )
            crop = ndd.flip(crop, horizontal=ndd.random.coin_flip(), vertical=0)

            blur_mask = ndd.random.coin_flip(probability=aug_cfg.blur_prob_local)
            blurred   = ndd.gaussian_blur(
                crop,
                sigma=ndd.random.uniform(range=(aug_cfg.blur_sigma_min,
                                                aug_cfg.blur_sigma_max)),
            )
            crop = blur_mask * blurred + (1 - blur_mask) * crop

            crop = ndd.crop_mirror_normalize(
                crop,
                dtype         = types.FLOAT16,
                output_layout = "CHW",
                mean          = batch_means,
                std           = batch_stds,
            )
            views[f"view_{view_idx}"] = crop
            view_idx += 1

        return views

    return _aug_fn


def _make_eval_aug_fn(aug_spec: EvalAugSpec) -> Any:
    """Return a dynamic-mode batch function for evaluation (resize + centre-crop).

    Args:
        aug_spec: Evaluation augmentation specification.

    Returns:
        Callable compatible with the dynamic pipeline iteration protocol.

    """
    _require_dynamic_dali()

    crop_size      = aug_spec.crop_size
    resize_size    = int(crop_size * 256 / 224)
    mean_255, std_255 = aug_spec.norm_stats.to_dali_scale()

    def _aug_fn(jpegs: ndd.Batch, ds_indices: list[int]) -> dict[str, ndd.Batch]:  # noqa: ARG001
        decoded = ndd.decoders.image(jpegs, device="gpu", output_type=types.RGB)
        resized = ndd.resize(decoded, resize_shorter=resize_size, device="gpu")
        cropped = ndd.crop_mirror_normalize(
            resized,
            crop_h        = crop_size,
            crop_w        = crop_size,
            crop_pos_x    = 0.5,
            crop_pos_y    = 0.5,
            dtype         = types.FLOAT16,
            output_layout = "CHW",
            mean          = mean_255,
            std           = std_255,
        )
        return {"view_0": cropped}

    return _aug_fn


def _make_lejpa_aug_fn(aug_spec: LeJEPAAugSpec) -> Any:
    """Return a dynamic-mode batch function for LeJEPA (context + target crops).

    All stochastic parameters for the context crop use ``ndd.random.*`` to
    produce per-sample diversity, consistent with the DINOv2 aug function.

    Args:
        aug_spec: LeJEPA augmentation specification.

    Returns:
        Callable compatible with the dynamic pipeline iteration protocol.

    """
    _require_dynamic_dali()

    mean_255, std_255 = aug_spec.norm_stats.to_dali_scale()

    def _aug_fn(jpegs: ndd.Batch, ds_indices: list[int]) -> dict[str, ndd.Batch]:  # noqa: ARG001
        decoded = ndd.decoders.image(jpegs, device="gpu", output_type=types.RGB)

        context = ndd.random_resized_crop(
            decoded,
            size        = aug_spec.context_crop_size,
            random_area = aug_spec.context_scale,
            device      = "gpu",
        )
        # Per-sample colour jitter via ndd.random.uniform.
        context = ndd.color_twist(
            context,
            brightness = ndd.random.uniform(range=(0.6, 1.4)),
            contrast   = ndd.random.uniform(range=(0.6, 1.4)),
            saturation = ndd.random.uniform(range=(0.6, 1.4)),
            hue        = ndd.random.uniform(range=(-0.1, 0.1)),
        )
        context = ndd.flip(context, horizontal=ndd.random.coin_flip(), vertical=0)
        context = ndd.crop_mirror_normalize(
            context,
            dtype         = types.FLOAT16,
            output_layout = "CHW",
            mean          = mean_255,
            std           = std_255,
        )

        views: dict[str, ndd.Batch] = {"context": context}

        # Target crops — no colour jitter (preserve reconstruction signal).
        for i in range(aug_spec.n_target_views):
            target = ndd.random_resized_crop(
                decoded,
                size        = aug_spec.target_crop_size,
                random_area = aug_spec.target_scale,
                device      = "gpu",
            )
            target = ndd.crop_mirror_normalize(
                target,
                dtype         = types.FLOAT16,
                output_layout = "CHW",
                mean          = mean_255,
                std           = std_255,
            )
            views[f"target_{i}"] = target

        return views

    return _aug_fn


# ---------------------------------------------------------------------------
# DynamicDINOPipeline
# ---------------------------------------------------------------------------


class DynamicDINOPipeline:
    """DALI dynamic-mode replacement for the static ``@pipeline_def`` pipeline.

    Wraps a dynamic-mode batch function and exposes the same iteration
    interface as the static ``DALIGenericIterator``:
    ``for dali_out in pipeline: views = [dali_out[0][name] for name in output_map]``.

    Args:
        aug_fn: Dynamic-mode batch function returning ``{view_name: ndd.Batch}``.
        source: ``MixingSource``-compatible callable returning JPEG byte arrays.
        output_map: Ordered list of view names (must match ``aug_fn`` output keys).
        batch_size: Samples per batch.
        device_id: GPU index.
        resolution: Thread-safe resolution holder (for ``DinoV2AugSpec`` only).
        ds_index_fn: Optional callable returning the current batch's per-sample
            dataset indices.  When provided, enables per-sample normalisation
            in the DINOv2 aug function.

    """

    def __init__(
        self,
        aug_fn:     Any,
        source:     Any,
        output_map: list[str],
        batch_size:  int,
        device_id:   int,
        resolution:  _ResolutionHolder | None = None,
        ds_index_fn: Any | None = None,
    ) -> None:
        """Initialise a DynamicDINOPipeline."""
        _require_dynamic_dali()
        self._aug_fn     = aug_fn
        self._source     = source
        self._output_map = output_map
        self._batch_size = batch_size
        self._device_id  = device_id
        self._resolution = resolution
        self._ds_index_fn = ds_index_fn

    def __iter__(self) -> DynamicDINOPipeline:
        """Return self."""
        return self

    def __next__(self) -> list[dict[str, Any]]:
        """Return one batch in DALIGenericIterator-compatible format.

        If a ``ds_index_fn`` was registered (via
        ``MixingSource.register_dataset_index_callback``), per-sample dataset
        indices are forwarded to the aug function to enable per-sample
        normalisation.  Otherwise the aug function falls back to global stats.
        """
        jpegs = self._source()

        # Per-sample dataset indices — populated if MixingSource has a
        # registered callback (set up by build_dynamic_pipeline).
        ds_indices: list[int] = (
            self._ds_index_fn() if self._ds_index_fn is not None else []
        )

        jpegs_batch = ndd.from_numpy(
            [np.frombuffer(j, dtype=np.uint8) for j in jpegs],
            device=f"gpu:{self._device_id}",
        )
        views = self._aug_fn(jpegs_batch, ds_indices)
        return [views]

    def reset(self) -> None:
        """No-op — stateless between epochs (matches static pipeline behaviour)."""

    @property
    def output_map(self) -> list[str]:
        """Ordered view names produced by this pipeline."""
        return list(self._output_map)

    def set_resolution(self, global_size: int, local_size: int) -> None:
        """Update crop resolution for the next batch (thread-safe).

        Args:
            global_size: New global crop size in pixels.
            local_size: New local crop size in pixels.

        """
        if self._resolution is not None:
            self._resolution.set(global_size, local_size)
        else:
            log.warning(
                "DynamicDINOPipeline.set_resolution called but no "
                "_ResolutionHolder is configured (non-DinoV2 aug spec).",
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_dynamic_pipeline(
    aug_spec:   AugmentationSpec,
    batch_size: int,
    device_id:  int,
    source:     Any,
    specs:      list[DatasetSpec] | None = None,
    seed:       int = 42,
) -> DynamicDINOPipeline:
    """Build a ``DynamicDINOPipeline`` dispatched on ``aug_spec`` type.

    This is a drop-in replacement for the ``build_pipeline`` call in
    ``DALIBackend.build_pipeline``.  The returned object supports the same
    iteration interface as ``DALIGenericIterator``.

    Per-sample dataset indices are wired automatically when ``source`` is a
    ``MixingSource`` instance: a lightweight callback captures the indices
    from the last ``__call__`` and forwards them to the aug function.  This
    enables per-sample normalisation matching the static pipeline's
    ``NormSource``.

    Args:
        aug_spec: Augmentation specification — determines which dynamic batch
            function is constructed.
        batch_size: Samples per batch.
        device_id: GPU index.
        source: ``MixingSource``-compatible callable.
        specs: Dataset specifications for per-dataset normalisation (required
            for ``DinoV2AugSpec``; ignored otherwise).
        seed: RNG seed (forwarded to DALI's internal seed; Python/NumPy RNG
            is intentionally NOT seeded here to avoid global state mutation).

    Returns:
        ``DynamicDINOPipeline`` ready for iteration.

    Raises:
        ImportError: If DALI dynamic mode is unavailable.
        TypeError: If ``aug_spec`` is not a recognised type.

    """
    _require_dynamic_dali()

    resolution:  _ResolutionHolder | None = None
    ds_index_fn: Any | None = None

    # Wire up per-sample dataset index capture if the source supports it.
    # MixingSource.register_dataset_index_callback stores the last batch's
    # indices; we read them back in __next__ via ds_index_fn().
    _last_ds_indices: list[list[int]] = [[]]  # mutable container for closure

    def _capture_indices(indices: list[int]) -> None:
        _last_ds_indices[0] = indices

    def _read_indices() -> list[int]:
        return _last_ds_indices[0]

    # Register only if the source exposes the callback API (MixingSource does).
    if hasattr(source, "register_dataset_index_callback"):
        source.register_dataset_index_callback(_capture_indices)
        ds_index_fn = _read_indices

    match aug_spec:
        case DinoV2AugSpec():
            effective_specs = specs or []
            norm_table      = _build_norm_table(aug_spec.aug_cfg, effective_specs)
            resolution      = _ResolutionHolder(
                aug_spec.aug_cfg.global_crop_size,
                aug_spec.aug_cfg.local_crop_size,
            )
            aug_fn     = _make_dinov2_aug_fn(aug_spec.aug_cfg, resolution, norm_table)
            output_map = aug_spec.output_map

        case EvalAugSpec():
            aug_fn     = _make_eval_aug_fn(aug_spec)
            output_map = aug_spec.output_map

        case LeJEPAAugSpec():
            aug_fn     = _make_lejpa_aug_fn(aug_spec)
            output_map = aug_spec.output_map

        case UserAugSpec():
            # UserAugSpec already carries a custom Python aug function.
            # In dynamic mode this is trivial — we decode + normalise with DALI
            # and hand the result to the user function.
            mean_255, std_255 = aug_spec.norm_stats.to_dali_scale()

            def aug_fn(jpegs: Any, ds_indices: list[int]) -> dict[str, Any]:  # noqa: ARG001
                decoded = ndd.decoders.image(jpegs, device="gpu", output_type=types.RGB)
                resized = ndd.resize(
                    decoded,
                    resize_shorter=aug_spec.decode_size,
                    device="gpu",
                )
                normalised = ndd.crop_mirror_normalize(
                    resized,
                    dtype         = types.FLOAT16,
                    output_layout = "CHW",
                    mean          = mean_255,
                    std           = std_255,
                )
                torch_tensor = normalised.as_tensor()
                return aug_spec.aug_fn(torch_tensor)

            output_map = aug_spec.output_map

        case _:
            msg = (
                f"build_dynamic_pipeline: unsupported aug_spec type "
                f"{type(aug_spec).__name__}."
            )
            raise TypeError(msg)

    log.info(
        "DynamicDINOPipeline built: spec=%s batch=%d device=cuda:%d",
        type(aug_spec).__name__, batch_size, device_id,
    )

    return DynamicDINOPipeline(
        aug_fn      = aug_fn,
        source      = source,
        output_map  = output_map,
        batch_size  = batch_size,
        device_id   = device_id,
        resolution  = resolution,
        ds_index_fn = ds_index_fn,
    )
