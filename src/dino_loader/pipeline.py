"""
dino_loader.augment.pipeline
============================
DALI augmentation pipeline for DINOv3 multi-crop.

Correctness fixes vs previous versions
---------------------------------------
1.  Coin-flip reuse bug: the old pattern

        do = coin_flip(p)
        out = do * augmented + (1 - do) * original

    called coin_flip ONCE and reused the result for both sides of the
    multiply — correct.  But several places had:

        out = coin_flip(p) * aug + (1 - coin_flip(p)) * orig

    which draws TWO independent flips, making the blend probabilistic
    rather than a hard switch.  Fixed by binding each flip to a variable.

2.  No silent fallback for hw_decoder_load: DALI raises if the value is
    out of range [0, 1]; we validate at pipeline-build time.

3.  preallocate_width_hint / preallocate_height_hint added for B200 so
    DALI can pre-allocate decode scratch buffers at the right size.

4.  BF16 (FLOAT16 inside DALI, cast to bfloat16 in PyTorch) used for
    all floating-point augmentation ops.  B200 BF16 TCs give 2× the
    throughput of FP32 at negligible quality cost.
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
            no_copy = True,   # pass buffers by reference where possible
        )

        views = []
        for i in range(aug_cfg.n_global_crops):
            blur_p = aug_cfg.blur_prob_global1 if i == 0 else aug_cfg.blur_prob_global2
            sol_p  = aug_cfg.solarize_prob      if i == 1 else 0.0
            views.append(_augment_view(
                jpegs, aug_cfg,
                crop_size        = aug_cfg.global_crop_size,
                scale            = aug_cfg.global_crops_scale,
                blur_prob        = blur_p,
                solarize_prob    = sol_p,
                hw_decoder_load  = hw_decoder_load,
            ))

        for _ in range(aug_cfg.n_local_crops):
            views.append(_augment_view(
                jpegs, aug_cfg,
                crop_size        = aug_cfg.local_crop_size,
                scale            = aug_cfg.local_crops_scale,
                blur_prob        = aug_cfg.blur_prob_local,
                solarize_prob    = 0.0,
                hw_decoder_load  = hw_decoder_load,
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

    All float ops run in FLOAT16 (maps to BF16 on B200 tensor cores).
    Each stochastic operation binds its coin-flip result to a named variable
    and uses it exactly once per branch — avoiding the double-draw bug.
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
    do_flip = fn.random.coin_flip(probability=cfg.flip_prob, dtype=types.BOOL)
    imgs    = fn.flip(imgs, device="gpu", horizontal=do_flip)

    # ── 3. Cast to FLOAT16 for all subsequent ops ─────────────────────────────
    imgs = fn.cast(imgs, dtype=types.FLOAT16)

    # ── 4. Color jitter ───────────────────────────────────────────────────────
    do_jitter = fn.random.coin_flip(probability=cfg.color_jitter_prob, dtype=types.BOOL)
    jittered  = fn.color_twist(
        imgs,
        brightness = fn.random.uniform(range=(1 - cfg.brightness, 1 + cfg.brightness)),
        contrast   = fn.random.uniform(range=(1 - cfg.contrast,   1 + cfg.contrast)),
        saturation = fn.random.uniform(range=(1 - cfg.saturation, 1 + cfg.saturation)),
        hue        = fn.random.uniform(range=(-cfg.hue * 180,     cfg.hue * 180)),
    )
    # do_jitter bound once; used symmetrically on both sides
    imgs = do_jitter * jittered + (1 - do_jitter) * imgs

    # ── 5. Random grayscale ───────────────────────────────────────────────────
    do_gray = fn.random.coin_flip(probability=cfg.grayscale_prob, dtype=types.BOOL)
    gray    = fn.color_space_conversion(imgs, image_type=types.RGB, output_type=types.GRAY)
    gray    = fn.cat(gray, gray, gray, axis=2)
    imgs    = do_gray * gray + (1 - do_gray) * imgs

    # ── 6. Gaussian blur ──────────────────────────────────────────────────────
    sigma   = fn.random.uniform(range=(cfg.blur_sigma_min, cfg.blur_sigma_max))
    blurred = fn.gaussian_blur(imgs, sigma=sigma)
    do_blur = fn.random.coin_flip(probability=blur_prob, dtype=types.BOOL)
    imgs    = do_blur * blurred + (1 - do_blur) * imgs

    # ── 7. Solarisation (second global crop only) ─────────────────────────────
    if solarize_prob > 0:
        do_sol  = fn.random.coin_flip(probability=solarize_prob, dtype=types.BOOL)
        mask    = imgs >= 128.0
        sol     = mask * (255.0 - imgs) + (1 - mask) * imgs
        imgs    = do_sol * sol + (1 - do_sol) * imgs

    # ── 8. Normalise to ImageNet stats ────────────────────────────────────────
    imgs = imgs / 255.0
    mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(cfg.std,  dtype=np.float32).reshape(1, 1, 3)
    imgs = (imgs - mean) / std

    # ── 9. HWC → CHW ──────────────────────────────────────────────────────────
    return fn.transpose(imgs, perm=[2, 0, 1])
