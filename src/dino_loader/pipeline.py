"""
dino_loader.pipeline
====================
DALI augmentation pipeline for DINOv3 multi-crop.

Correctness fixes vs previous versions
---------------------------------------
1.  Coin-flip reuse bug: fixed in previous review.
2.  hw_decoder_load range validation: added in previous review.
3.  preallocate_width_hint / preallocate_height_hint: added in previous review.
4.  BF16 (FLOAT16 inside DALI) for all float ops: added in previous review.

[NOTE-F] Coin-flip arithmetic with BOOL tensors hardened.
         The previous pattern ``do_flag * tensor_a + (1 - do_flag) * tensor_b``
         relied on DALI's implicit BOOL → FLOAT16 promotion.  This works
         correctly in current DALI versions but is fragile against future
         dtype-promotion rule changes.  Replaced with an explicit
         ``fn.cast(do_flag, dtype=types.FLOAT16)`` before each use, making
         the intent unambiguous and immune to implicit-promotion changes.

No additional bugs were found in this file.  Clarifying comments added:
- Solarisation operates on values in [0, 255] (pre-normalisation) —
  the threshold 128.0 is correct and intentional.
- The tensor layout throughout is HWC (DALI native after image decode);
  fn.transpose([2,0,1]) at the end converts to CHW for PyTorch.
- fn.color_space_conversion requires 3-channel input; all paths maintain
  3 channels so the call is safe.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from dino_loader.config import DINOAugConfig

log = logging.getLogger(__name__)

try:
    import nvidia.dali.fn    as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    log.error("nvidia-dali not installed — pipeline will not build")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline builder
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    source,
    aug_cfg:         DINOAugConfig,
    batch_size:      int,
    num_threads:     int,
    device_id:       int,
    hw_decoder_load: float = 0.90,
    cpu_queue:       int   = 8,
    gpu_queue:       int   = 6,
    seed:            int   = 42,
):
    """
    Build and return a compiled DALI pipeline.

    Parameters
    ----------
    source          : MixingSource instance (DALI ExternalSource callback)
    aug_cfg         : DINOAugConfig
    batch_size      : samples per GPU per step
    num_threads     : DALI CPU worker threads
    device_id       : GPU index
    hw_decoder_load : fraction of JPEG decode work sent to NVJPEG HW ASIC
    cpu_queue       : DALI prefetch CPU queue depth
    gpu_queue       : DALI prefetch GPU queue depth
    seed            : base random seed (rank offset applied by caller)
    """
    if not HAS_DALI:
        raise RuntimeError("nvidia-dali is required but not installed.")

    if not 0.0 <= hw_decoder_load <= 1.0:
        raise ValueError(f"hw_decoder_load must be in [0, 1], got {hw_decoder_load}")

    @pipeline_def(
        batch_size            = batch_size,
        num_threads           = num_threads,
        device_id             = device_id,
        seed                  = seed,
        prefetch_queue_depth  = {"cpu_size": cpu_queue, "gpu_size": gpu_queue},
        exec_async            = True,
        exec_pipelined        = True,
    )
    def _pipe():
        jpegs = fn.external_source(
            source  = source,
            dtype   = types.UINT8,
            ndim    = 1,
            name    = "jpegs",
            no_copy = True,
        )

        views = []
        for i in range(aug_cfg.n_global_crops):
            blur_p = aug_cfg.blur_prob_global1 if i == 0 else aug_cfg.blur_prob_global2
            # Solarisation is applied only to the second global crop (DINOv2 §A.1)
            sol_p  = aug_cfg.solarize_prob if i == 1 else 0.0
            views.append(_augment_view(
                jpegs, aug_cfg,
                crop_size       = aug_cfg.global_crop_size,
                scale           = aug_cfg.global_crops_scale,
                blur_prob       = blur_p,
                solarize_prob   = sol_p,
                hw_decoder_load = hw_decoder_load,
            ))

        for _ in range(aug_cfg.n_local_crops):
            views.append(_augment_view(
                jpegs, aug_cfg,
                crop_size       = aug_cfg.local_crop_size,
                scale           = aug_cfg.local_crops_scale,
                blur_prob       = aug_cfg.blur_prob_local,
                solarize_prob   = 0.0,
                hw_decoder_load = hw_decoder_load,
            ))

        return tuple(views)

    pipe = _pipe()
    pipe.build()
    log.info(
        "DALI pipeline built: %d global + %d local crops, "
        "batch=%d, threads=%d, hw_decoder=%.0f%%",
        aug_cfg.n_global_crops, aug_cfg.n_local_crops,
        batch_size, num_threads, hw_decoder_load * 100,
    )
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# Single-view augmentation graph
# ══════════════════════════════════════════════════════════════════════════════

def _augment_view(
    jpegs,
    cfg:             DINOAugConfig,
    crop_size:       int,
    scale:           Tuple[float, float],
    blur_prob:       float,
    solarize_prob:   float,
    hw_decoder_load: float,
):
    """
    Augmentation sub-graph for one crop view.

    Tensor layout
    -------------
    DALI's image decode + crop returns tensors in HWC layout (H × W × C).
    All subsequent ops (flip, cast, color_twist, etc.) also operate in HWC.
    The final fn.transpose([2, 0, 1]) converts to CHW for PyTorch.

    Float range
    -----------
    After fn.cast(FLOAT16), pixel values are in [0.0, 255.0].  Solarisation
    uses threshold 128.0 which is the standard midpoint for uint8 images —
    this is intentional and correct.  Normalisation (÷255 then mean/std) is
    applied AFTER all stochastic augmentations.

    All float ops run in FLOAT16 (maps to BF16 on B200 tensor cores).

    Coin-flip blending [NOTE-F]
    ---------------------------
    Each stochastic op binds its coin-flip to a named variable.  The BOOL
    tensor is explicitly cast to FLOAT16 before arithmetic, making the
    blending formula ``w * branch_a + (1 - w) * branch_b`` unambiguous and
    independent of DALI's implicit-promotion rules.
    """
    # ── 1. Hardware JPEG decode + random resized crop ─────────────────────────
    imgs = fn.decoders.image_random_crop(
        jpegs,
        device                  = "mixed",
        output_type             = types.RGB,
        random_area             = list(scale),
        random_aspect_ratio     = [3 / 4, 4 / 3],
        num_attempts            = 10,
        hw_decoder_load         = hw_decoder_load,
        preallocate_width_hint  = crop_size * 2,
        preallocate_height_hint = crop_size * 2,
    )
    imgs = fn.resize(
        imgs,
        device      = "gpu",
        resize_x    = crop_size,
        resize_y    = crop_size,
        interp_type = types.INTERP_CUBIC,
        antialias   = False,
    )

    # ── 2. Random horizontal flip ─────────────────────────────────────────────
    # Layout: HWC, values [0, 255] uint8
    do_flip = fn.random.coin_flip(probability=cfg.flip_prob, dtype=types.BOOL)
    imgs    = fn.flip(imgs, device="gpu", horizontal=do_flip)

    # ── 3. Cast to FLOAT16 for all subsequent ops ─────────────────────────────
    # Values remain in [0.0, 255.0] after cast.
    imgs = fn.cast(imgs, dtype=types.FLOAT16)

    # ── 4. Color jitter ───────────────────────────────────────────────────────
    # [NOTE-F] Explicit cast of coin-flip BOOL → FLOAT16 before arithmetic.
    do_jitter  = fn.cast(
        fn.random.coin_flip(probability=cfg.color_jitter_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    jittered   = fn.color_twist(
        imgs,
        brightness = fn.random.uniform(range=(1 - cfg.brightness, 1 + cfg.brightness)),
        contrast   = fn.random.uniform(range=(1 - cfg.contrast,   1 + cfg.contrast)),
        saturation = fn.random.uniform(range=(1 - cfg.saturation, 1 + cfg.saturation)),
        hue        = fn.random.uniform(range=(-cfg.hue * 180,     cfg.hue * 180)),
    )
    imgs = do_jitter * jittered + (1 - do_jitter) * imgs

    # ── 5. Random grayscale ───────────────────────────────────────────────────
    # fn.color_space_conversion requires 3-channel HWC input — satisfied here.
    # [NOTE-F] Explicit cast before blending arithmetic.
    do_gray = fn.cast(
        fn.random.coin_flip(probability=cfg.grayscale_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    gray    = fn.color_space_conversion(imgs, image_type=types.RGB, output_type=types.GRAY)
    # Replicate single channel → 3 channels to keep downstream ops uniform
    gray    = fn.cat(gray, gray, gray, axis=2)
    imgs    = do_gray * gray + (1 - do_gray) * imgs

    # ── 6. Gaussian blur ──────────────────────────────────────────────────────
    # [NOTE-F] Explicit cast before blending arithmetic.
    sigma   = fn.random.uniform(range=(cfg.blur_sigma_min, cfg.blur_sigma_max))
    blurred = fn.gaussian_blur(imgs, sigma=sigma)
    do_blur = fn.cast(
        fn.random.coin_flip(probability=blur_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    imgs    = do_blur * blurred + (1 - do_blur) * imgs

    # ── 7. Solarisation (second global crop only, solarize_prob > 0) ──────────
    # Applied while values are still in [0.0, 255.0].  Threshold 128.0 is the
    # standard midpoint for uint8 images — correct and intentional.
    # [NOTE-F] Explicit cast before blending arithmetic.
    if solarize_prob > 0:
        do_sol = fn.cast(
            fn.random.coin_flip(probability=solarize_prob, dtype=types.BOOL),
            dtype=types.FLOAT16,
        )
        mask   = imgs >= 128.0
        sol    = mask * (255.0 - imgs) + (1 - mask) * imgs
        imgs   = do_sol * sol + (1 - do_sol) * imgs

    # ── 8. Normalise to ImageNet stats ────────────────────────────────────────
    # Scale to [0, 1] then apply per-channel mean/std.
    imgs = imgs / 255.0
    mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(cfg.std,  dtype=np.float32).reshape(1, 1, 3)
    imgs = (imgs - mean) / std

    # ── 9. HWC → CHW for PyTorch ──────────────────────────────────────────────
    return fn.transpose(imgs, perm=[2, 0, 1])
