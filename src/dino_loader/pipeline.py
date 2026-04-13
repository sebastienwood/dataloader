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

[NORM]      All normalisation stat conversions now go through
            ``NormStats.to_dali_scale()``, eliminating ad-hoc ``× 255``
            multiplications scattered across the codebase.

[FIX-FLIP]  ``_augment_view_dinov2`` previously forced every sample to be
            horizontally flipped (``fn.flip(..., horizontal=1)``), ignoring
            ``DINOAugConfig.flip_prob`` entirely and diverging from the CPU
            path.  The flip is now gated by a ``fn.random.coin_flip`` with the
            correct probability.

[FIX-SOL]   Solarization was computed in ``_build_dinov2_pipeline`` but never
            forwarded to ``_augment_view_dinov2`` — the parameter was missing
            from the function signature.  It is now correctly passed and
            applied, matching the documented DINOv2 recipe.

[FIX-DTYPE] ``_augment_view_dinov2`` (and every other pipeline function)
            previously hard-coded ``dtype=types.FLOAT16`` for normalisation,
            ignoring ``LoaderConfig.output_dtype`` / ``PipelineConfig.output_dtype``.
            The dtype is now resolved via ``PipelineConfig.dali_dtype_str`` and
            forwarded through each pipeline builder, so selecting ``"fp32"``
            actually produces FP32 tensors.

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
from dino_loader.config import NormStats, PipelineConfig

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


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def _dali_dtype(dali_dtype_str: str) -> Any:
    """Resolve a DALI type from its string name.

    Args:
        dali_dtype_str: One of ``"FLOAT16"`` or ``"FLOAT"`` (as produced by
            ``PipelineConfig.dali_dtype_str``).

    Returns:
        The corresponding ``nvidia.dali.types`` constant, or
        ``types.FLOAT16`` as a safe fallback.

    """
    if not HAS_DALI:
        return None
    return getattr(types, dali_dtype_str, types.FLOAT16)


# ══════════════════════════════════════════════════════════════════════════════
# [PL-3] NormSource — per-sample normalisation ExternalSource callback
# ══════════════════════════════════════════════════════════════════════════════

class NormSource:
    """DALI ExternalSource callback that emits per-sample (mean, std) tensors.

    All stats are stored in [0, 1] scale internally and converted to [0, 255]
    via ``NormStats.to_dali_scale()`` only when returned to the DALI graph.

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
        """Build the per-dataset normalisation lookup table.

        Args:
            aug_cfg: Global augmentation config providing fallback mean/std.
            specs: Dataset specifications (may carry per-dataset mean/std).

        """
        global_stats = aug_cfg.norm_stats  # NormStats in [0,1]

        self._lookup: list[NormStats] = []
        for spec in specs:
            self._lookup.append(
                NormStats.from_config(
                    mean     = spec.mean,
                    std      = spec.std,
                    fallback = global_stats,
                ),
            )

        self._indices: list[int] = [0]
        self._lock = threading.Lock()

    def set_dataset_indices(self, indices: list[int]) -> None:
        """Called by MixingSource before each batch.  [M2-FIX] Copy-on-write.

        Args:
            indices: Per-sample dataset index for the upcoming batch.

        """
        new_indices = list(indices)
        with self._lock:
            self._indices = new_indices

    def __call__(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (means, stds) in [0, 255] scale — one (3,) FLOAT32 array per sample.

        Returns explicit numpy copies so the DALI pipeline cannot hold
        references into the live lookup table.  [M2-FIX]
        """
        with self._lock:
            indices = self._indices

        means: list[np.ndarray] = []
        stds:  list[np.ndarray] = []

        for i in indices:
            stats = self._lookup[i]
            mean_255, std_255 = stats.to_dali_scale()
            means.append(np.array(mean_255, dtype=np.float32))
            stds.append(np.array(std_255,  dtype=np.float32))

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
    pipeline_cfg:       PipelineConfig | None = None,
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
        pipeline_cfg: If provided, overrides the individual dtype/queue/thread
            parameters.  Preferred over passing them individually.

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

    # Resolve effective dtype: PipelineConfig wins when supplied.
    effective_dali_dtype_str = (
        pipeline_cfg.dali_dtype_str if pipeline_cfg is not None else "FLOAT16"
    )
    if pipeline_cfg is not None:
        num_threads     = pipeline_cfg.num_threads
        device_id       = pipeline_cfg.device_id
        hw_decoder_load = pipeline_cfg.hw_decoder_load
        cpu_queue       = pipeline_cfg.cpu_queue
        gpu_queue       = pipeline_cfg.gpu_queue
        seed            = pipeline_cfg.seed
        fuse_normalization = pipeline_cfg.fuse_normalization
        dali_fp8_output    = pipeline_cfg.dali_fp8_output

    common_kwargs: dict[str, Any] = dict(
        source               = source,
        batch_size           = batch_size,
        num_threads          = num_threads,
        device_id            = device_id,
        resolution_src       = resolution_src,
        hw_decoder_load      = hw_decoder_load,
        cpu_queue            = cpu_queue,
        gpu_queue            = gpu_queue,
        seed                 = seed,
        dali_dtype_str       = effective_dali_dtype_str,
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
            return _build_decode_only_pipeline(aug_spec=aug_spec, **common_kwargs)
        case _:
            msg = (
                f"Unknown augmentation spec type: {type(aug_spec).__name__}.  "
                "Expected one of: DinoV2AugSpec, EvalAugSpec, LeJEPAAugSpec, UserAugSpec."
            )
            raise TypeError(msg)


# ══════════════════════════════════════════════════════════════════════════════
# DINOv2 multi-crop pipeline
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
    dali_dtype_str:     str = "FLOAT16",
) -> Any:
    """Build the original DINOv2 multi-crop DALI pipeline."""
    n_views = aug_cfg.n_views
    # Fallback stats in [0,255] for when per-sample fused normalisation is off.
    global_mean_255, global_std_255 = aug_cfg.norm_stats.to_dali_scale()
    out_dtype = _dali_dtype(dali_dtype_str)

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
            # [FIX-SOL] Solarization only on the second global crop (i == 1).
            # Previously sol_prob was computed here but never forwarded to the
            # augmentation helper — the parameter was absent from its signature.
            sol_prob   = aug_cfg.solarize_prob if (i == 1) else 0.0
            view = _augment_view_dinov2(
                jpegs            = jpegs,
                aug_cfg          = aug_cfg,
                crop_size        = crop_size,
                scale            = scale,
                blur_prob        = blur_prob,
                sol_prob         = sol_prob,
                seed             = seed + i,
                hw_decoder_load  = hw_decoder_load,
                means            = means,
                stds             = stds,
                global_mean_255  = global_mean_255,
                global_std_255   = global_std_255,
                out_dtype        = out_dtype,
                dali_fp8_output  = dali_fp8_output,
                name             = f"view_{i}",
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
    global_mean_255: list[float],
    global_std_255:  list[float],
    out_dtype:       Any,
    dali_fp8_output: bool,
    name:            str,
) -> Any:
    """Build one DINOv2 augmentation branch.

    Args:
        jpegs: DALI tensor of raw JPEG bytes.
        aug_cfg: Augmentation configuration.
        crop_size: DALI scalar for the crop size.
        scale: RandomResizedCrop area scale range.
        blur_prob: Gaussian blur probability for this view.
        sol_prob: Solarisation probability for this view.
        seed: RNG seed offset for this view.
        hw_decoder_load: Fraction of decode sent to nvjpeg HW.
        means: Per-sample mean tensors from NormSource, or None.
        stds: Per-sample std tensors from NormSource, or None.
        global_mean_255: Fallback mean in [0, 255].
        global_std_255: Fallback std in [0, 255].
        out_dtype: DALI dtype constant for normalisation output.
        dali_fp8_output: Whether to cast output to FP8.
        name: Output tensor name.

    Returns:
        A DALI graph node for this view.

    """
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

    # [FIX-FLIP] Previously `fn.flip(..., horizontal=1)` forced every sample to
    # be flipped regardless of DINOAugConfig.flip_prob, diverging from the CPU
    # path and the documented DINOv2 recipe.  The flip is now stochastic and
    # respects the configured probability.
    flip_coin = fn.random.coin_flip(probability=aug_cfg.flip_prob, seed=seed + 6)
    augmented = fn.flip(augmented, horizontal=flip_coin, vertical=0)

    blurred = fn.gaussian_blur(
        augmented,
        sigma = fn.random.uniform(range=(0.1, 2.0), seed=seed + 4),
    )
    blur_coin = fn.random.coin_flip(probability=blur_prob, seed=seed + 5)
    augmented = (
        fn.cast(blur_coin, dtype=types.FLOAT) * blurred
        + fn.cast(1 - blur_coin, dtype=types.FLOAT) * augmented
    )

    # [FIX-SOL] Solarization is now actually applied when sol_prob > 0.
    # Previously the sol_prob value was computed in the caller but the
    # parameter was absent from this function's signature, so the operator
    # was never inserted into the DALI graph.
    if sol_prob > 0.0:
        sol_coin  = fn.random.coin_flip(probability=sol_prob, seed=seed + 7)
        solarised = fn.experimental.solarize(augmented, threshold=128)
        augmented = (
            fn.cast(sol_coin, dtype=types.FLOAT) * solarised
            + fn.cast(1 - sol_coin, dtype=types.FLOAT) * augmented
        )

    # [FIX-DTYPE] Use out_dtype (resolved from PipelineConfig.dali_dtype_str)
    # instead of hard-coding types.FLOAT16.
    if means is not None and stds is not None:
        # Per-sample stats from NormSource (already in [0,255]).
        normalised = fn.normalize(augmented, mean=means, stddev=stds, dtype=out_dtype)
    else:
        # Global fallback stats (pre-converted to [0,255]).
        normalised = fn.normalize(
            augmented,
            mean   = global_mean_255,
            stddev = global_std_255,
            dtype  = out_dtype,
        )

    if dali_fp8_output:
        try:
            output = fn.cast(normalised, dtype=types.FLOAT8_E4M3)
        except AttributeError:
            log.warning(
                "DALI FLOAT8_E4M3 not available (requires DALI ≥ 1.36) — "
                "falling back to %s output.",
                out_dtype,
            )
            output = normalised
    else:
        output = normalised

    return fn.transpose(output, perm=[2, 0, 1], name=name)


# ══════════════════════════════════════════════════════════════════════════════
# Eval pipeline — deterministic, single view
# ══════════════════════════════════════════════════════════════════════════════

def _build_eval_pipeline(
    source:          Any,
    aug_spec:        EvalAugSpec,
    batch_size:      int,
    num_threads:     int,
    device_id:       int,
    resolution_src:  Any,  # ignored — eval uses fixed crop_size
    hw_decoder_load: float,
    cpu_queue:       int,
    gpu_queue:       int,
    seed:            int,
    dali_dtype_str:  str = "FLOAT16",
) -> Any:
    """Build the evaluation pipeline: resize-shorter-side + centre-crop."""
    crop_size          = aug_spec.crop_size
    mean_255, std_255  = aug_spec.norm_stats.to_dali_scale()
    interp             = types.INTERP_CUBIC if aug_spec.interpolation == "bicubic" else types.INTERP_LINEAR
    out_dtype          = _dali_dtype(dali_dtype_str)

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

        resize_size = int(crop_size * 256 / 224)
        resized = fn.resize(
            decoded,
            resize_shorter    = resize_size,
            interp_type       = interp,
            device            = "gpu",
        )

        cropped = fn.crop(
            resized,
            crop_h     = crop_size,
            crop_w     = crop_size,
            crop_pos_x = 0.5,
            crop_pos_y = 0.5,
            device     = "gpu",
        )

        # [FIX-DTYPE] Honour out_dtype.
        normalised = fn.normalize(
            cropped,
            mean   = mean_255,
            stddev = std_255,
            dtype  = out_dtype,
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
    dali_dtype_str:  str = "FLOAT16",
) -> Any:
    """Build the LeJEPA pipeline: one context crop + N target crops."""
    mean_255, std_255 = aug_spec.norm_stats.to_dali_scale()
    out_dtype         = _dali_dtype(dali_dtype_str)

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
        context = fn.flip(
            context,
            horizontal=fn.random.coin_flip(probability=0.5, seed=seed + 4),
            vertical=0,
        )
        # [FIX-DTYPE] Honour out_dtype.
        context = fn.normalize(context, mean=mean_255, stddev=std_255, dtype=out_dtype)
        context = fn.transpose(context, perm=[2, 0, 1], name="context")
        outputs.append(context)

        for i in range(aug_spec.n_target_views):
            target = fn.random_resized_crop(
                decoded,
                size        = aug_spec.target_crop_size,
                random_area = aug_spec.target_scale,
                device      = "gpu",
                seed        = seed + 10 + i,
            )
            # [FIX-DTYPE] Honour out_dtype.
            target = fn.normalize(target, mean=mean_255, stddev=std_255, dtype=out_dtype)
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
    dali_dtype_str:  str = "FLOAT16",
) -> Any:
    """Build a decode-only pipeline for user-defined augmentation.

    DALI decodes the JPEG, resizes to ``aug_spec.decode_size``, and normalises.
    The user function is applied outside the DALI graph by the iterator wrapper.
    """
    mean_255, std_255 = aug_spec.norm_stats.to_dali_scale()
    out_dtype         = _dali_dtype(dali_dtype_str)

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

        # [FIX-DTYPE] Honour out_dtype.
        normalised = fn.normalize(
            resized,
            mean   = mean_255,
            stddev = std_255,
            dtype  = out_dtype,
        )
        return fn.transpose(normalised, perm=[2, 0, 1], name="decoded")

    pipe = _pipeline_fn()
    pipe.build()
    return pipe
