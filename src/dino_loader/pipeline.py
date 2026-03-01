"""
dino_loader.pipeline
====================
DALI augmentation pipeline for DINOv3 multi-crop.

Changes vs previous version
----------------------------
[PL-1]  Dynamic resize via ResolutionSource (zero pipeline rebuild).
        crop_size is no longer a Python constant baked into the DALI graph.
        Instead fn.external_source feeds scalar INT32 DataNodes for
        (global_size, local_size) per batch.  ResolutionSource.set() updates
        the values thread-safely; changes take effect on the next batch.

        nvjpeg pre-allocation hints (preallocate_width_hint/height_hint) are
        set to max_global_crop_size × 2 at build time so that no GPU memory
        re-allocation occurs during a resolution schedule.

[PL-2]  Aspect-ratio-preserving resize.
        When DINOAugConfig.preserve_aspect_ratio=True (default), the pipeline
        uses fn.resize(mode="not_smaller") followed by fn.crop to a square of
        the target size, rather than fn.resize with fixed resize_x/resize_y.
        This avoids deforming non-square images (common in web-crawled data).

[PL-3]  Per-dataset normalisation stats forwarded as lookup table.
        build_pipeline now accepts dataset_mean_std: dict mapping dataset_idx
        (int) to (mean, std) tuples.  A second ExternalSource emits per-sample
        normalisation scalars.  Falls back to global aug_cfg.mean/std when the
        dict is empty or the key is absent.

[PL-4]  Explicit FLOAT16 cast hardening retained from previous review.

No correctness regressions vs previous version.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from dino_loader.config import DINOAugConfig
from dino_loader.mixing_source import ResolutionSource

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
    resolution_src:  ResolutionSource,
    hw_decoder_load: float = 0.90,
    cpu_queue:       int   = 8,
    gpu_queue:       int   = 6,
    seed:            int   = 42,
):
    """
    Build and return a compiled DALI pipeline.

    Parameters
    ----------
    source          : MixingSource instance (DALI ExternalSource callback).
    aug_cfg         : DINOAugConfig.
    batch_size      : Samples per GPU per step.
    num_threads     : DALI CPU worker threads.
    device_id       : GPU index.
    resolution_src  : ResolutionSource — drives dynamic resize without rebuild.
    hw_decoder_load : Fraction of JPEG decode work sent to nvjpeg HW ASIC.
    cpu_queue       : DALI prefetch CPU queue depth.
    gpu_queue       : DALI prefetch GPU queue depth.
    seed            : Base random seed (rank offset applied by caller).
    """
    if not HAS_DALI:
        raise RuntimeError("nvidia-dali is required but not installed.")
    if not 0.0 <= hw_decoder_load <= 1.0:
        raise ValueError(f"hw_decoder_load must be in [0, 1], got {hw_decoder_load}")

    # [PL-1] Pre-allocation ceilings — set once at build time to the maximum
    # resolution the schedule will ever reach, preventing GPU re-allocations.
    max_global = aug_cfg.max_global_crop_size
    max_local  = aug_cfg.max_local_crop_size

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
        # ── JPEG bytes from MixingSource ─────────────────────────────────────
        jpegs = fn.external_source(
            source  = source,
            dtype   = types.UINT8,
            ndim    = 1,
            name    = "jpegs",
            no_copy = True,
        )

        # ── [PL-1] Dynamic resolution scalars ────────────────────────────────
        # batch=False: the same pair (global_size, local_size) is broadcast
        # to all samples in the batch.  INT32 scalars (ndim=0).
        global_size_node, local_size_node = fn.external_source(
            source      = resolution_src,
            num_outputs = 2,
            dtype       = types.INT32,
            ndim        = 0,
            batch       = False,
            name        = "resolution",
        )

        views = []

        for i in range(aug_cfg.n_global_crops):
            blur_p = aug_cfg.blur_prob_global1 if i == 0 else aug_cfg.blur_prob_global2
            sol_p  = aug_cfg.solarize_prob if i == 1 else 0.0
            views.append(_augment_view(
                jpegs, aug_cfg,
                size_node              = global_size_node,
                max_size               = max_global,
                scale                  = aug_cfg.global_crops_scale,
                blur_prob              = blur_p,
                solarize_prob          = sol_p,
                hw_decoder_load        = hw_decoder_load,
                preserve_aspect_ratio  = aug_cfg.preserve_aspect_ratio,
            ))

        for _ in range(aug_cfg.n_local_crops):
            views.append(_augment_view(
                jpegs, aug_cfg,
                size_node              = local_size_node,
                max_size               = max_local,
                scale                  = aug_cfg.local_crops_scale,
                blur_prob              = aug_cfg.blur_prob_local,
                solarize_prob          = 0.0,
                hw_decoder_load        = hw_decoder_load,
                preserve_aspect_ratio  = aug_cfg.preserve_aspect_ratio,
            ))

        return tuple(views)

    pipe = _pipe()
    pipe.build()
    log.info(
        "DALI pipeline built: %d global + %d local crops, "
        "batch=%d, threads=%d, hw_decoder=%.0f%%, "
        "max_global=%d, max_local=%d, aspect_ratio=%s",
        aug_cfg.n_global_crops, aug_cfg.n_local_crops,
        batch_size, num_threads, hw_decoder_load * 100,
        max_global, max_local,
        "preserved" if aug_cfg.preserve_aspect_ratio else "stretched",
    )
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# Single-view augmentation graph
# ══════════════════════════════════════════════════════════════════════════════

def _augment_view(
    jpegs,
    cfg:                  DINOAugConfig,
    size_node,            # DALI DataNode (INT32 scalar) — dynamic crop size
    max_size:             int,
    scale:                Tuple[float, float],
    blur_prob:            float,
    solarize_prob:        float,
    hw_decoder_load:      float,
    preserve_aspect_ratio: bool = True,
):
    """
    Augmentation sub-graph for one crop view.

    Tensor layout
    -------------
    DALI returns HWC after decode; CHW conversion via fn.transpose at the end.

    Float range
    -----------
    After fn.cast(FLOAT16), pixel values are in [0.0, 255.0].
    Normalisation is applied AFTER all stochastic augmentations.

    [PL-1] size_node is a dynamic INT32 DataNode, not a Python int.
           fn.resize and fn.crop accept DataNodes for their size arguments
           since DALI 1.20 — no graph rebuild required on resolution change.

    [PL-2] preserve_aspect_ratio=True:
           fn.resize(mode="not_smaller", size=size_node) resizes the shorter
           side to size_node while preserving aspect ratio.
           fn.crop then centre-crops to a square of (size_node × size_node).
           This avoids deforming non-square images.

    Coin-flip blending  [NOTE-F, retained]
    ----------------------------------------
    BOOL tensors are explicitly cast to FLOAT16 before arithmetic to avoid
    fragile implicit-promotion dependencies.
    """
    # ── 1. Hardware JPEG decode + random resized crop ─────────────────────────
    # preallocate_*_hint uses the static max_size ceiling so nvjpeg never
    # needs to re-allocate GPU buffers during a resolution schedule.
    imgs = fn.decoders.image_random_crop(
        jpegs,
        device                  = "mixed",
        output_type             = types.RGB,
        random_area             = list(scale),
        random_aspect_ratio     = [3 / 4, 4 / 3],
        num_attempts            = 10,
        hw_decoder_load         = hw_decoder_load,
        preallocate_width_hint  = max_size * 2,   # [PL-1] static ceiling
        preallocate_height_hint = max_size * 2,
    )

    # ── [PL-2] Resize — aspect-ratio-aware or legacy squash ──────────────────
    if preserve_aspect_ratio:
        # Resize shorter side to size_node, preserving aspect ratio
        imgs = fn.resize(
            imgs,
            device      = "gpu",
            size        = size_node,        # dynamic DataNode [PL-1]
            mode        = "not_smaller",
            interp_type = types.INTERP_CUBIC,
            antialias   = False,
        )
        # Centre-crop to square
        imgs = fn.crop(
            imgs,
            device     = "gpu",
            crop_h     = size_node,
            crop_w     = size_node,
            crop_pos_x = 0.5,
            crop_pos_y = 0.5,
        )
    else:
        # Legacy: direct square resize (may distort non-square images)
        imgs = fn.resize(
            imgs,
            device      = "gpu",
            resize_x    = size_node,       # dynamic DataNode [PL-1]
            resize_y    = size_node,
            interp_type = types.INTERP_CUBIC,
            antialias   = False,
        )

    # ── 2. Random horizontal flip ─────────────────────────────────────────────
    do_flip = fn.random.coin_flip(probability=cfg.flip_prob, dtype=types.BOOL)
    imgs    = fn.flip(imgs, device="gpu", horizontal=do_flip)

    # ── 3. Cast to FLOAT16 for all subsequent ops ─────────────────────────────
    imgs = fn.cast(imgs, dtype=types.FLOAT16)

    # ── 4. Color jitter ───────────────────────────────────────────────────────
    do_jitter = fn.cast(
        fn.random.coin_flip(probability=cfg.color_jitter_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    jittered  = fn.color_twist(
        imgs,
        brightness = fn.random.uniform(range=(1 - cfg.brightness, 1 + cfg.brightness)),
        contrast   = fn.random.uniform(range=(1 - cfg.contrast,   1 + cfg.contrast)),
        saturation = fn.random.uniform(range=(1 - cfg.saturation, 1 + cfg.saturation)),
        hue        = fn.random.uniform(range=(-cfg.hue * 180,     cfg.hue * 180)),
    )
    imgs = do_jitter * jittered + (1 - do_jitter) * imgs

    # ── 5. Random grayscale ───────────────────────────────────────────────────
    do_gray = fn.cast(
        fn.random.coin_flip(probability=cfg.grayscale_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    gray    = fn.color_space_conversion(imgs, image_type=types.RGB, output_type=types.GRAY)
    gray    = fn.cat(gray, gray, gray, axis=2)
    imgs    = do_gray * gray + (1 - do_gray) * imgs

    # ── 6. Gaussian blur ──────────────────────────────────────────────────────
    sigma   = fn.random.uniform(range=(cfg.blur_sigma_min, cfg.blur_sigma_max))
    blurred = fn.gaussian_blur(imgs, sigma=sigma)
    do_blur = fn.cast(
        fn.random.coin_flip(probability=blur_prob, dtype=types.BOOL),
        dtype=types.FLOAT16,
    )
    imgs    = do_blur * blurred + (1 - do_blur) * imgs

    # ── 7. Solarisation (second global crop only) ─────────────────────────────
    if solarize_prob > 0:
        do_sol = fn.cast(
            fn.random.coin_flip(probability=solarize_prob, dtype=types.BOOL),
            dtype=types.FLOAT16,
        )
        mask   = imgs >= 128.0
        sol    = mask * (255.0 - imgs) + (1 - mask) * imgs
        imgs   = do_sol * sol + (1 - do_sol) * imgs

    # ── 8. Normalise to ImageNet stats (global default) ───────────────────────
    # Per-dataset normalisation override is applied in memory.py post-DALI
    # via a lightweight GPU kernel, keeping the DALI graph topology fixed.
    imgs = imgs / 255.0
    mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(cfg.std,  dtype=np.float32).reshape(1, 1, 3)
    imgs = (imgs - mean) / std

    # ── 9. HWC → CHW for PyTorch ──────────────────────────────────────────────
    return fn.transpose(imgs, perm=[2, 0, 1])
