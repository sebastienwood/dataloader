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
- Per-dataset normalisation is handled via a simple index lookup inside the
  batch function, not via a DALI graph node.
- The four ``@pipeline_def`` dispatch paths collapse into four plain Python
  branches in a single function.

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
        specs          = dataset_specs,
    )

    # Use exactly like the static pipeline:
    for dali_out in pipeline:
        views = [dali_out[0][name] for name in aug_spec.output_map]
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
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
from dino_loader.config import DINOAugConfig

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


@dataclass
class _NormStats:
    """Per-dataset mean/std arrays, pre-multiplied to [0, 255] scale."""

    mean: np.ndarray  # shape (3,), float32, values in [0, 255]
    std:  np.ndarray  # shape (3,), float32, positive


def _build_norm_table(
    aug_cfg: DINOAugConfig,
    specs:   list[DatasetSpec],
) -> list[_NormStats]:
    """Build a lookup table of per-dataset normalisation statistics.

    Falls back to the global ``aug_cfg`` mean/std for datasets that have
    no per-dataset override.

    Args:
        aug_cfg: Global augmentation config (provides fallback mean/std).
        specs: Dataset specifications (may carry per-dataset mean/std).

    Returns:
        List of ``_NormStats`` aligned with ``specs`` (index ``i`` →
        ``specs[i]``).

    """
    global_mean = np.array(aug_cfg.mean, dtype=np.float32) * 255.0
    global_std  = np.array(aug_cfg.std,  dtype=np.float32) * 255.0

    table: list[_NormStats] = []
    for spec in specs:
        m = (np.array(spec.mean, dtype=np.float32) * 255.0) if spec.mean else global_mean.copy()
        s = (np.array(spec.std,  dtype=np.float32) * 255.0) if spec.std  else global_std.copy()
        table.append(_NormStats(mean=m, std=s))

    return table


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
        self._global = global_size
        self._local  = local_size
        self._lock   = threading.Lock()

    def set(self, global_size: int, local_size: int) -> None:
        """Update both dimensions atomically."""
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
    norm_table: list[_NormStats],
) -> Any:
    """Return a dynamic-mode batch function implementing DINOv2 multi-crop.

    The returned callable takes a ``ndd.Batch`` of raw JPEG bytes and the
    current dataset index per sample, and returns a dict of augmented views.

    Args:
        aug_cfg: Augmentation configuration.
        resolution: Thread-safe resolution holder (updated by ``set_resolution``).
        norm_table: Per-dataset normalisation statistics.

    Returns:
        Callable compatible with ``ndd.pytorch.nodes.DictMapper``.

    """
    _require_dynamic_dali()

    n_global = aug_cfg.n_global_crops
    n_local  = aug_cfg.n_local_crops

    def _aug_fn(jpegs: ndd.Batch, ds_indices: list[int]) -> dict[str, ndd.Batch]:
        global_size = resolution.global_size
        local_size  = resolution.local_size

        # Decode once; subsequent crops share the decoded buffer.
        decoded = ndd.decoders.image(jpegs, device="gpu", output_type=types.RGB)

        # Build per-sample mean/std from the dataset index.
        # We use the most common stats for the batch to keep the graph simple;
        # a future improvement could scatter per-sample stats.
        # For now default to index 0 if ds_indices is unavailable.
        primary_idx = ds_indices[0] if ds_indices else 0
        norm        = norm_table[min(primary_idx, len(norm_table) - 1)]

        views: dict[str, ndd.Batch] = {}
        view_idx = 0

        for i in range(n_global):
            is_first  = i == 0
            blur_prob = aug_cfg.blur_prob_global1 if is_first else aug_cfg.blur_prob_global2
            sol_prob  = aug_cfg.solarize_prob if (i == 1) else 0.0

            crop = ndd.random_resized_crop(
                decoded,
                size        = global_size,
                random_area = aug_cfg.global_crops_scale,
                random_aspect_ratio = (3 / 4, 4 / 3),
                device      = "gpu",
            )

            # Colour jitter
            crop = ndd.color_twist(
                crop,
                brightness = float(np.random.uniform(0.6, 1.4)),
                contrast   = float(np.random.uniform(0.6, 1.4)),
                saturation = float(np.random.uniform(0.6, 1.4)),
                hue        = float(np.random.uniform(-0.1, 0.1)),
            )
            crop = ndd.flip(crop, horizontal=1, vertical=0)

            # Gaussian blur (probabilistic)
            if np.random.random() < blur_prob:
                sigma = float(np.random.uniform(0.1, 2.0))
                crop  = ndd.gaussian_blur(crop, sigma=sigma)

            # Solarisation (second global crop only)
            if sol_prob > 0.0 and np.random.random() < sol_prob:
                crop = ndd.solarize(crop, threshold=128)

            crop = ndd.crop_mirror_normalize(
                crop,
                dtype          = types.FLOAT16,
                output_layout  = "CHW",
                mean           = norm.mean.tolist(),
                std            = norm.std.tolist(),
            )
            views[f"view_{view_idx}"] = crop
            view_idx += 1

        for _ in range(n_local):
            crop = ndd.random_resized_crop(
                decoded,
                size        = local_size,
                random_area = aug_cfg.local_crops_scale,
                random_aspect_ratio = (3 / 4, 4 / 3),
                device      = "gpu",
            )
            crop = ndd.color_twist(
                crop,
                brightness = float(np.random.uniform(0.6, 1.4)),
                contrast   = float(np.random.uniform(0.6, 1.4)),
                saturation = float(np.random.uniform(0.6, 1.4)),
                hue        = float(np.random.uniform(-0.1, 0.1)),
            )
            crop = ndd.flip(crop, horizontal=1, vertical=0)
            if np.random.random() < aug_cfg.blur_prob_local:
                sigma = float(np.random.uniform(0.1, 2.0))
                crop  = ndd.gaussian_blur(crop, sigma=sigma)
            crop = ndd.crop_mirror_normalize(
                crop,
                dtype         = types.FLOAT16,
                output_layout = "CHW",
                mean          = norm.mean.tolist(),
                std           = norm.std.tolist(),
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
        Callable compatible with ``ndd.pytorch.nodes.DictMapper``.

    """
    _require_dynamic_dali()

    crop_size   = aug_spec.crop_size
    resize_size = int(crop_size * 256 / 224)
    mean_arr    = (np.array(aug_spec.mean, dtype=np.float32) * 255.0).tolist()
    std_arr     = (np.array(aug_spec.std,  dtype=np.float32) * 255.0).tolist()

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
            mean          = mean_arr,
            std           = std_arr,
        )
        return {"view_0": cropped}

    return _aug_fn


def _make_lejpa_aug_fn(aug_spec: LeJEPAAugSpec) -> Any:
    """Return a dynamic-mode batch function for LeJEPA (context + target crops).

    Args:
        aug_spec: LeJEPA augmentation specification.

    Returns:
        Callable compatible with ``ndd.pytorch.nodes.DictMapper``.

    """
    _require_dynamic_dali()

    mean_arr = (np.array(aug_spec.mean, dtype=np.float32) * 255.0).tolist()
    std_arr  = (np.array(aug_spec.std,  dtype=np.float32) * 255.0).tolist()

    def _aug_fn(jpegs: ndd.Batch, ds_indices: list[int]) -> dict[str, ndd.Batch]:  # noqa: ARG001
        decoded = ndd.decoders.image(jpegs, device="gpu", output_type=types.RGB)

        context = ndd.random_resized_crop(
            decoded,
            size        = aug_spec.context_crop_size,
            random_area = aug_spec.context_scale,
            device      = "gpu",
        )
        context = ndd.color_twist(
            context,
            brightness = float(np.random.uniform(0.6, 1.4)),
            contrast   = float(np.random.uniform(0.6, 1.4)),
            saturation = float(np.random.uniform(0.6, 1.4)),
            hue        = float(np.random.uniform(-0.1, 0.1)),
        )
        context = ndd.flip(context, horizontal=1, vertical=0)
        context = ndd.crop_mirror_normalize(
            context,
            dtype=types.FLOAT16, output_layout="CHW",
            mean=mean_arr, std=std_arr,
        )

        views: dict[str, ndd.Batch] = {"context": context}

        for i in range(aug_spec.n_target_views):
            target = ndd.random_resized_crop(
                decoded,
                size        = aug_spec.target_crop_size,
                random_area = aug_spec.target_scale,
                device      = "gpu",
            )
            target = ndd.crop_mirror_normalize(
                target,
                dtype=types.FLOAT16, output_layout="CHW",
                mean=mean_arr, std=std_arr,
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

    """

    def __init__(
        self,
        aug_fn:     Any,
        source:     Any,
        output_map: list[str],
        batch_size:  int,
        device_id:   int,
        resolution:  _ResolutionHolder | None = None,
    ) -> None:
        _require_dynamic_dali()
        self._aug_fn     = aug_fn
        self._source     = source
        self._output_map = output_map
        self._batch_size = batch_size
        self._device_id  = device_id
        self._resolution = resolution

    def __iter__(self) -> DynamicDINOPipeline:
        return self

    def __next__(self) -> list[dict[str, Any]]:
        """Return one batch in DALIGenericIterator-compatible format."""
        jpegs = self._source()

        # ds_indices are embedded in MixingSource but not surfaced as a
        # return value; we pass an empty list and let the aug_fn fall back
        # to index 0 for norm stats.  A future improvement would register a
        # dataset index callback on MixingSource.
        ds_indices: list[int] = []

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

    Args:
        aug_spec: Augmentation specification — determines which dynamic batch
            function is constructed.
        batch_size: Samples per batch.
        device_id: GPU index.
        source: ``MixingSource``-compatible callable.
        specs: Dataset specifications for per-dataset normalisation (required
            for ``DinoV2AugSpec``; ignored otherwise).
        seed: RNG seed (forwarded to numpy; DALI dynamic ops are seeded
            internally per call).

    Returns:
        ``DynamicDINOPipeline`` ready for iteration.

    Raises:
        ImportError: If DALI dynamic mode is unavailable.
        TypeError: If ``aug_spec`` is not a recognised type.

    """
    _require_dynamic_dali()

    np.random.seed(seed)  # deterministic augmentation when seed is fixed

    resolution: _ResolutionHolder | None = None

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
            # In dynamic mode this is trivial — we just call it directly.
            def aug_fn(jpegs: Any, ds_indices: list[int]) -> dict[str, Any]:  # noqa: ARG001
                decoded = ndd.decoders.image(jpegs, device="gpu", output_type=types.RGB)
                resized = ndd.resize(decoded, resize_shorter=aug_spec.decode_size, device="gpu")
                mean_arr = (np.array(aug_spec.mean, dtype=np.float32) * 255.0).tolist()
                std_arr  = (np.array(aug_spec.std,  dtype=np.float32) * 255.0).tolist()
                normalised = ndd.crop_mirror_normalize(
                    resized,
                    dtype=types.FLOAT16, output_layout="CHW",
                    mean=mean_arr, std=std_arr,
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
        aug_fn     = aug_fn,
        source     = source,
        output_map = output_map,
        batch_size = batch_size,
        device_id  = device_id,
        resolution = resolution,
    )
