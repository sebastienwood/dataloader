"""dino_loader.pipeline.

DALI augmentation pipeline builder supporting multiple augmentation strategies.

Changes in this version
-----------------------
[PL-AUG-1]  ``build_pipeline`` now dispatches on ``AugmentationSpec`` subtype.
            A single entry-point replaces the previous hardcoded DINOv2 graph,
            enabling ``EvalAugSpec``, ``LeJEPAAugSpec``, and ``UserAugSpec``
            without touching ``loader.py`` or the backend interface.

[PL-AUG-2]  ``UserAugSpec`` path: DALI handles decode + resize + normalise only;
            the user function is called as a post-decode hook on GPU tensors,
            *outside* the DALI graph but *inside* the CUDA stream so GPU-CPU
            synchronisation cost is minimised.

[PL-AUG-3]  ``EvalAugSpec`` path: deterministic resize + centre-crop, no
            stochastic ops, single view.  Suitable for val loops and
            fine-tuning.

[PL-AUG-4]  ``LeJEPAAugSpec`` path: one context crop (large scale) + N target
            crops (small scale).  No colour jitter on target crops to preserve
            the reconstruction signal.

[M2-FIX]    NormSource thread safety hardened (retained from previous version).

[PL-1..5]   All previous improvements retained.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from dino_loader.augmentation import (
    AugmentationSpec,
    DinoV2AugSpec,
    EvalAugSpec,
    LeJEPAAugSpec,
    UserAugSpec,
)

if TYPE_CHECKING:
    from dino_datasets import DatasetSpec

    from dino_loader.config import DINOAugConfig

log = logging.getLogger(__name__)

try:
    from nvidia.dali import fn, pipeline_def, types
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    log.error("nvidia-dali not installed — pipeline will not build")


# ══════════════════════════════════════════════════════════════════════════════
# [PL-3] NormSource — per-sample normalisation ExternalSource callback
# ══════════════════════════════════════════════════════════════════════════════

class NormSource:
    """DALI ExternalSource callback that emits per-sample (mean, std) tensors.

    Thread safety — [M2-FIX]:
        set_dataset_indices() is called from MixingSource's ExternalSource
        callback thread.  DALI calls __call__() from its own prefetch thread.
        A threading.Lock serialises access to self._indices.  set_dataset_indices()
        builds the new list fully before acquiring the lock, then swaps the
        reference atomically.  __call__() returns explicit numpy copies so the
        DALI pipeline cannot hold references into the live lookup table.
    """

    def __init__(
        self,
        aug_cfg:  DINOAugConfig,
        specs:    list[DatasetSpec],
    ) -> None:
        global_mean = np.array(aug_cfg.mean, dtype=np.float32)
        global_std  = np.array(aug_cfg.std,  dtype=np.float32)
        self._lookup: list[tuple[np.ndarray, np.ndarray]] = []
        for spec in specs:
            m = np.array(spec.mean, dtype=np.float32) if spec.mean else global_mean.copy()
            s = np.array(spec.std,  dtype=np.float32) if spec.std  else global_std.copy()
            self._lookup.append((m, s))

        self._indices: list[int] = [0]
        self._lock = threading.Lock()

    def set_dataset_indices(self, indices: list[int]) -> None:
        """Called by MixingSource before each batch.  [M2-FIX] Copy-on-write."""
        new_indices = list(indices)
        with self._lock:
            self._indices = new_indices

    def __call__(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (means, stds) — one (3,) FLOAT32 array per sample.  [M2-FIX]"""
        with self._lock:
            indices = self._indices

        means = [np.array(self._lookup[i][0], dtype=np.float32) for i in indices]
        stds  = [np.array(self._lookup[i][1], dtype=np.float32) for i in indices]
        return means, stds


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch entry-point
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    source:             Any,
    aug_spec:           AugmentationSpec,
    batch_size:         int,
    num_threads:        int,
    device_id:          int,
    resolution_src:     Any,
    hw_decoder_load:    float = 0.90,
    cpu_queue:          int   = 8,
    gpu_queue:          int   = 6,
    seed:               int   = 42,
    norm_source:        NormSource | None = None,
    fuse_normalization: bool  = True,
    dali_fp8_output:    bool  = False,
) -> Any:
    """Build and return a compiled DALI pipeline dispatched on ``aug_spec`` type.

    Args:
        source: MixingSource — DALI ExternalSource callback.
        aug_spec: Augmentation specification (``DinoV2AugSpec``, ``EvalAugSpec``,
            ``LeJEPAAugSpec``, or ``UserAugSpec``).
        batch_size: Samples per GPU per step.
        num_threads: DALI CPU worker threads.
        device_id: GPU index.
        resolution_src: ResolutionSource — dynamic resize (PL-1).
        hw_decoder_load: Fraction of JPEG decode sent to nvjpeg HW ASIC (0–1).
        cpu_queue: DALI CPU-side prefetch queue depth.
        gpu_queue: DALI GPU-side prefetch queue depth.
        seed: Pipeline RNG seed.
        norm_source: NormSource for per-dataset normalisation (PL-3).
        fuse_normalization: Fuse per-dataset norm into DALI kernel.
        dali_fp8_output: Emit FP8-cast tensors from the DALI graph (PL-5).

    Returns:
        A compiled DALI pipeline object.

    Raises:
        RuntimeError: If nvidia-dali is not installed.
        TypeError: If ``aug_spec`` is an unknown type.

    """
    if not HAS_DALI:
        raise RuntimeError(
            "nvidia-dali is not installed.  "
            "Install with: pip install nvidia-dali-cuda120",
        )

    common_kwargs: dict[str, Any] = dict(
        source          = source,
        batch_size      = batch_size,
        num_threads     = num_threads,
        device_id       = device_id,
        resolution_src  = resolution_src,
        hw_decoder_load = hw_decoder_load,
        cpu_queue       = cpu_queue,
        gpu_queue       = gpu_queue,
        seed            = seed,
    )

    match aug_spec:
        case DinoV2AugSpec():
            return _build_dinov2_pipeline(
                aug_cfg            = aug_spec.aug_cfg,
                norm_source        = norm_source,
                fuse_normalization = fuse_normalization,
                dali_fp8_output    = dali_fp8_output,
                **common_kwargs,
            )
        case EvalAugSpec():
            return _build_eval_pipeline(aug_spec=aug_spec, **common_kwargs)
        case LeJEPAAugSpec():
            return _build_lejpa_pipeline(aug_spec=aug_spec, **common_kwargs)
        case UserAugSpec():
            # For user-defined augmentation, we build a decode-only pipeline;
            # the user function is wired in by DALIBackend's iterator wrapper.
            return _build_decode_only_pipeline(aug_spec=aug_spec, **common_kwargs)
        case _:
            msg = (
                f"Unknown augmentation spec type: {type(aug_spec).__name__}.  "
                "Expected one of: DinoV2AugSpec, EvalAugSpec, LeJEPAAugSpec, UserAugSpec."
            )
            raise TypeError(msg)


# ══════════════════════════════════════════════════════════════════════════════
# DINOv2 multi-crop pipeline (retained, refactored to accept DinoV2AugSpec)
# ══════════════════════════════════════════════════════════════════════════════

def _build_dinov2_pipeline(
    source:             Any,
    aug_cfg:            DINOAugConfig,
    batch_size:         int,
    num_threads:        int,
    device_id:          int,
    resolution_src:     Any,
    hw_decoder_load:    float,
    cpu_queue:          int,
    gpu_queue:          int,
    seed:               int,
    norm_source:        NormSource | None,
    fuse_normalization: bool,
    dali_fp8_output:    bool,
) -> Any:
    """Build the original DINOv2 multi-crop DALI pipeline."""
    n_views = aug_cfg.n_views

    @pipeline_def(
        batch_size      = batch_size,
        num_threads     = num_threads,
        device_id       = device_id,
        prefetch_queue_depth = {"cpu_size": cpu_queue, "gpu_size": gpu_queue},
        seed            = seed,
        exec_async      = True,
        exec_pipelined  = True,
    )
    def _pipeline_fn():
        jpegs = fn.external_source(
            source      = source,
            num_outputs = 1,
            batch       = True,
            dtype       = types.UINT8,
            name        = "jpegs",
        )[0]

        global_size, local_size = fn.external_source(
            source      = resolution_src,
            num_outputs = 2,
            batch       = False,
            dtype       = types.INT32,
            name        = "resolution",
        )

        means = stds = None
        if fuse_normalization and norm_source is not None:
            means, stds = fn.external_source(
                source      = norm_source,
                num_outputs = 2,
                batch       = True,
                dtype       = types.FLOAT,
                name        = "norm_stats",
            )

        views = []
        for i in range(n_views):
            is_global  = i < aug_cfg.n_global_crops
            crop_size  = global_size if is_global else local_size
            scale      = aug_cfg.global_crops_scale if is_global else aug_cfg.local_crops_scale
            blur_prob  = (
                aug_cfg.blur_prob_global1 if i == 0
                else aug_cfg.blur_prob_global2 if is_global
                else aug_cfg.blur_prob_local
            )
            sol_prob   = aug_cfg.solarize_prob if (i == 1) else 0.0
            view = _augment_view_dinov2(
                jpegs           = jpegs,
                aug_cfg         = aug_cfg,
                crop_size       = crop_size,
                scale           = scale,
                blur_prob       = blur_prob,
                sol_prob        = sol_prob,
                seed            = seed + i,
                hw_decoder_load = hw_decoder_load,
                means           = means,
                stds            = stds,
                dali_fp8_output = dali_fp8_output,
                name            = f"view_{i}",
            )
            views.append(view)

        return tuple(views)

    pipe = _pipeline_fn()
    pipe.build()
    return pipe


def _augment_view_dinov2(
    jpegs,
    aug_cfg:         DINOAugConfig,
    crop_size,
    scale:           tuple[float, float],
    blur_prob:       float,
    sol_prob:        float,
    seed:            int,
    hw_decoder_load: float,
    means,
    stds,
    dali_fp8_output: bool,
    name:            str,
) -> Any:
    """Build one DINOv2 augmentation branch."""
    decoded = fn.decoders.image(
        jpegs,
        device             = "mixed",
        output_type        = types.RGB,
        hw_decoder_load    = hw_decoder_load,
    )

    if aug_cfg.preserve_aspect_ratio:
        resized = fn.random_resized_crop(
            decoded,
            size                = crop_size,
            random_area         = scale,
            random_aspect_ratio = (3 / 4, 4 / 3),
            device              = "gpu",
        )
    else:
        resized = fn.random_resized_crop(
            decoded,
            size        = crop_size,
            random_area = scale,
            device      = "gpu",
        )

    augmented = fn.color_twist(
        resized,
        brightness = fn.random.uniform(range=(0.6, 1.4), seed=seed),
        contrast   = fn.random.uniform(range=(0.6, 1.4), seed=seed + 1),
        saturation = fn.random.uniform(range=(0.6, 1.4), seed=seed + 2),
        hue        = fn.random.uniform(range=(-0.1, 0.1), seed=seed + 3),
    )
    augmented = fn.flip(augmented, horizontal=1, vertical=0)

    blurred = fn.gaussian_blur(
        augmented,
        sigma = fn.random.uniform(range=(0.1, 2.0), seed=seed + 4),
    )
    augmented = (
        fn.cast(fn.random.coin_flip(probability=blur_prob, seed=seed + 5), dtype=types.FLOAT)
        * blurred
        + fn.cast(
            1 - fn.random.coin_flip(probability=blur_prob, seed=seed + 5),
            dtype=types.FLOAT,
        )
        * augmented
    )

    if means is not None and stds is not None:
        normalised = fn.normalize(augmented, mean=means, stddev=stds, dtype=types.FLOAT16)
    else:
        mean_arr = np.array(aug_cfg.mean, dtype=np.float32) * 255.0
        std_arr  = np.array(aug_cfg.std,  dtype=np.float32) * 255.0
        normalised = fn.normalize(
            augmented,
            mean   = mean_arr.tolist(),
            stddev = std_arr.tolist(),
            dtype  = types.FLOAT16,
        )

    if dali_fp8_output:
        try:
            output = fn.cast(normalised, dtype=types.FLOAT8_E4M3)
        except AttributeError:
            log.warning(
                "DALI FLOAT8_E4M3 not available (requires DALI ≥ 1.36) — "
                "falling back to FLOAT16 output.",
            )
            output = normalised
    else:
        output = normalised

    return fn.transpose(output, perm=[2, 0, 1], name=name)


# ══════════════════════════════════════════════════════════════════════════════
# Eval pipeline — deterministic, single view
# ══════════════════════════════════════════════════════════════════════════════

def _build_eval_pipeline(
    source:         Any,
    aug_spec:       EvalAugSpec,
    batch_size:     int,
    num_threads:    int,
    device_id:      int,
    resolution_src: Any,  # ignored — eval uses fixed crop_size
    hw_decoder_load: float,
    cpu_queue:      int,
    gpu_queue:      int,
    seed:           int,
) -> Any:
    """Build the evaluation pipeline: resize-shorter-side + centre-crop."""
    crop_size = aug_spec.crop_size
    mean_arr  = (np.array(aug_spec.mean, dtype=np.float32) * 255.0).tolist()
    std_arr   = (np.array(aug_spec.std,  dtype=np.float32) * 255.0).tolist()
    interp    = types.INTERP_CUBIC if aug_spec.interpolation == "bicubic" else types.INTERP_LINEAR

    @pipeline_def(
        batch_size           = batch_size,
        num_threads          = num_threads,
        device_id            = device_id,
        prefetch_queue_depth = {"cpu_size": cpu_queue, "gpu_size": gpu_queue},
        seed                 = seed,
        exec_async           = True,
        exec_pipelined       = True,
    )
    def _pipeline_fn():
        jpegs = fn.external_source(
            source      = source,
            num_outputs = 1,
            batch       = True,
            dtype       = types.UINT8,
            name        = "jpegs",
        )[0]

        decoded = fn.decoders.image(
            jpegs,
            device          = "mixed",
            output_type     = types.RGB,
            hw_decoder_load = hw_decoder_load,
        )

        # Resize shorter side to crop_size * 256/224 ≈ 1.14× for a clean centre crop.
        resize_size = int(crop_size * 256 / 224)
        resized = fn.resize(
            decoded,
            resize_shorter    = resize_size,
            interp_type       = interp,
            device            = "gpu",
        )

        cropped = fn.crop(
            resized,
            crop_h = crop_size,
            crop_w = crop_size,
            crop_pos_x = 0.5,
            crop_pos_y = 0.5,
            device = "gpu",
        )

        normalised = fn.normalize(
            cropped,
            mean   = mean_arr,
            stddev = std_arr,
            dtype  = types.FLOAT16,
        )
        return fn.transpose(normalised, perm=[2, 0, 1], name="view_0")

    pipe = _pipeline_fn()
    pipe.build()
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# LeJEPA pipeline — context + N target crops
# ══════════════════════════════════════════════════════════════════════════════

def _build_lejpa_pipeline(
    source:          Any,
    aug_spec:        LeJEPAAugSpec,
    batch_size:      int,
    num_threads:     int,
    device_id:       int,
    resolution_src:  Any,  # ignored — LeJEPA uses fixed sizes
    hw_decoder_load: float,
    cpu_queue:       int,
    gpu_queue:       int,
    seed:            int,
) -> Any:
    """Build the LeJEPA pipeline: one context crop + N target crops."""
    mean_arr = (np.array(aug_spec.mean, dtype=np.float32) * 255.0).tolist()
    std_arr  = (np.array(aug_spec.std,  dtype=np.float32) * 255.0).tolist()

    @pipeline_def(
        batch_size           = batch_size,
        num_threads          = num_threads,
        device_id            = device_id,
        prefetch_queue_depth = {"cpu_size": cpu_queue, "gpu_size": gpu_queue},
        seed                 = seed,
        exec_async           = True,
        exec_pipelined       = True,
    )
    def _pipeline_fn():
        jpegs = fn.external_source(
            source      = source,
            num_outputs = 1,
            batch       = True,
            dtype       = types.UINT8,
            name        = "jpegs",
        )[0]

        decoded = fn.decoders.image(
            jpegs,
            device          = "mixed",
            output_type     = types.RGB,
            hw_decoder_load = hw_decoder_load,
        )

        outputs = []

        # Context: large crop with colour jitter (encoder input).
        context = fn.random_resized_crop(
            decoded,
            size        = aug_spec.context_crop_size,
            random_area = aug_spec.context_scale,
            device      = "gpu",
        )
        context = fn.color_twist(
            context,
            brightness = fn.random.uniform(range=(0.6, 1.4), seed=seed),
            contrast   = fn.random.uniform(range=(0.6, 1.4), seed=seed + 1),
            saturation = fn.random.uniform(range=(0.6, 1.4), seed=seed + 2),
            hue        = fn.random.uniform(range=(-0.1, 0.1), seed=seed + 3),
        )
        context = fn.flip(context, horizontal=1, vertical=0)
        context = fn.normalize(context, mean=mean_arr, stddev=std_arr, dtype=types.FLOAT16)
        context = fn.transpose(context, perm=[2, 0, 1], name="context")
        outputs.append(context)

        # Target crops: small crops, NO colour jitter (preserve reconstruction signal).
        for i in range(aug_spec.n_target_views):
            target = fn.random_resized_crop(
                decoded,
                size        = aug_spec.target_crop_size,
                random_area = aug_spec.target_scale,
                device      = "gpu",
                seed        = seed + 10 + i,
            )
            target = fn.normalize(target, mean=mean_arr, stddev=std_arr, dtype=types.FLOAT16)
            target = fn.transpose(target, perm=[2, 0, 1], name=f"target_{i}")
            outputs.append(target)

        return tuple(outputs)

    pipe = _pipeline_fn()
    pipe.build()
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# Decode-only pipeline for UserAugSpec
# ══════════════════════════════════════════════════════════════════════════════

def _build_decode_only_pipeline(
    source:          Any,
    aug_spec:        UserAugSpec,
    batch_size:      int,
    num_threads:     int,
    device_id:       int,
    resolution_src:  Any,  # ignored
    hw_decoder_load: float,
    cpu_queue:       int,
    gpu_queue:       int,
    seed:            int,
) -> Any:
    """Build a decode-only pipeline for user-defined augmentation.

    DALI decodes the JPEG, resizes to ``aug_spec.decode_size``, and normalises.
    The user function is applied outside the DALI graph by the iterator wrapper.
    """
    mean_arr = (np.array(aug_spec.mean, dtype=np.float32) * 255.0).tolist()
    std_arr  = (np.array(aug_spec.std,  dtype=np.float32) * 255.0).tolist()

    @pipeline_def(
        batch_size           = batch_size,
        num_threads          = num_threads,
        device_id            = device_id,
        prefetch_queue_depth = {"cpu_size": cpu_queue, "gpu_size": gpu_queue},
        seed                 = seed,
        exec_async           = True,
        exec_pipelined       = True,
    )
    def _pipeline_fn():
        jpegs = fn.external_source(
            source      = source,
            num_outputs = 1,
            batch       = True,
            dtype       = types.UINT8,
            name        = "jpegs",
        )[0]

        decoded = fn.decoders.image(
            jpegs,
            device          = "mixed",
            output_type     = types.RGB,
            hw_decoder_load = hw_decoder_load,
        )

        resized = fn.resize(
            decoded,
            resize_shorter = aug_spec.decode_size,
            device         = "gpu",
        )

        normalised = fn.normalize(
            resized,
            mean   = mean_arr,
            stddev = std_arr,
            dtype  = types.FLOAT16,
        )
        # Output named "decoded" — the iterator wrapper applies aug_fn on top.
        return fn.transpose(normalised, perm=[2, 0, 1], name="decoded")

    pipe = _pipeline_fn()
    pipe.build()
    return pipe
