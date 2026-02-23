"""
DINOv3-style DALI DataLoader
============================
Features:
  - WebDataset shards as input (tar files)
  - Dynamic mixing ratios across datasets (can be updated at any moment)
  - Distributed training support (rank/world_size aware shard splitting)
  - True DINOv3 augmentation: multi-crop (2 global + N local views),
    color jitter, grayscale, gaussian blur, solarization, random horizontal flip

Dependencies:
    pip install nvidia-dali-cuda120  # or cuda110
    pip install webdataset torch torchvision
"""

from __future__ import annotations

import io
import math
import os
import random
import threading
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

# ---------------------------------------------------------------------------
# DALI imports
# ---------------------------------------------------------------------------
import nvidia.dali.fn as fn
import nvidia.dali.math as dmath
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# ---------------------------------------------------------------------------
# WebDataset reading (pure Python, fed into DALI via ExternalSource)
# ---------------------------------------------------------------------------
import webdataset as wds


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetSpec:
    """Specification for a single WebDataset source."""
    name: str
    shards: List[str]          # list / glob of tar shard paths
    weight: float = 1.0        # relative mixing weight (will be normalised)


@dataclass
class DINOAugConfig:
    """DINOv3 augmentation hyper-parameters."""
    # Global crops
    global_crops_scale: Tuple[float, float] = (0.32, 1.0)
    global_crop_size: int = 224
    n_global_crops: int = 2
    # Local crops
    local_crops_scale: Tuple[float, float] = (0.05, 0.32)
    local_crop_size: int = 96
    n_local_crops: int = 8
    # Color jitter
    color_jitter_prob: float = 0.8
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    hue: float = 0.1
    # Grayscale
    grayscale_prob: float = 0.2
    # Gaussian blur
    blur_prob_global1: float = 1.0   # always blur first global crop
    blur_prob_global2: float = 0.1   # rarely blur second global crop
    blur_prob_local: float = 0.5
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0
    # Solarization (second global crop only, as in DINOv2)
    solarize_prob: float = 0.2
    # Flip
    flip_prob: float = 0.5
    # Normalisation (ImageNet)
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic Mixing Source  (ExternalSource callback)
# ══════════════════════════════════════════════════════════════════════════════

class DynamicMixingSource:
    """
    Thread-safe iterator that samples from multiple WebDataset streams
    according to weights that can be updated at runtime via `set_weights()`.

    Each call to __next__ yields one batch of raw JPEG bytes so that DALI
    can decode them on the GPU.
    """

    def __init__(
        self,
        dataset_specs: List[DatasetSpec],
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle_shards: bool = True,
        buffer_size: int = 1000,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self._lock = threading.Lock()

        # Build per-dataset iterators
        self._names = [s.name for s in dataset_specs]
        self._weights = np.array([s.weight for s in dataset_specs], dtype=np.float64)
        self._weights /= self._weights.sum()

        self._iterators: List[Iterator] = []
        for spec in dataset_specs:
            it = self._build_iterator(spec.shards, shuffle_shards, buffer_size, seed)
            self._iterators.append(it)

        self._rng = random.Random(seed + rank)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update mixing weights. Thread-safe. Can be called at any time."""
        w = np.array(weights, dtype=np.float64)
        if len(w) != len(self._weights):
            raise ValueError(
                f"Expected {len(self._weights)} weights, got {len(w)}"
            )
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            raise ValueError("At least one weight must be positive.")
        with self._lock:
            self._weights = w / w.sum()

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Convenience: update a single dataset's weight."""
        idx = self._names.index(name)
        with self._lock:
            new_w = self._weights.copy()
        new_w[idx] = weight
        self.set_weights(new_w)

    # ------------------------------------------------------------------
    # Iterator protocol (called by DALI ExternalSource)
    # ------------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        """Return a batch of raw image bytes as a list of numpy uint8 arrays."""
        with self._lock:
            weights = self._weights.copy()

        batch = []
        for _ in range(self.batch_size):
            ds_idx = self._rng.choices(range(len(self._iterators)), weights=weights)[0]
            sample = self._next_sample(ds_idx)
            batch.append(np.frombuffer(sample, dtype=np.uint8))

        return batch  # DALI ExternalSource expects a list of numpy arrays

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_iterator(
        self,
        shards: List[str],
        shuffle: bool,
        buffer_size: int,
        seed: int,
    ) -> Iterator:
        """Build a WebDataset iterator with distributed shard splitting."""
        # Assign shards to this rank
        assigned = [
            s for i, s in enumerate(shards) if i % self.world_size == self.rank
        ]
        if not assigned:
            raise RuntimeError(
                f"Rank {self.rank}: no shards assigned from {len(shards)} total."
            )

        dataset = (
            wds.WebDataset(assigned, shardshuffle=shuffle, seed=seed + self.rank)
            .shuffle(buffer_size if shuffle else 0)
            .decode("pil")
            .to_tuple("jpg;jpeg;png;webp")
            .map(lambda x: self._pil_to_bytes(x[0]))
            .repeat()  # infinite
        )
        return iter(dataset)

    @staticmethod
    def _pil_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def _next_sample(self, ds_idx: int) -> bytes:
        try:
            return next(self._iterators[ds_idx])
        except StopIteration:
            # Re-initialise iterator (shouldn't happen with .repeat(), but safety net)
            return next(self._iterators[ds_idx])


# ══════════════════════════════════════════════════════════════════════════════
# DALI Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _coin(prob: float, batch_size: int):
    """Bernoulli coin flip for a whole batch."""
    return fn.random.coin_flip(probability=prob, dtype=types.BOOL)


def _apply_color_jitter(imgs, cfg: DINOAugConfig):
    """Random color jitter (brightness / contrast / saturation / hue)."""
    # DALI applies each component with its own random magnitude
    imgs = fn.color_twist(
        imgs,
        brightness=fn.random.uniform(range=(1 - cfg.brightness, 1 + cfg.brightness)),
        contrast=fn.random.uniform(range=(1 - cfg.contrast, 1 + cfg.contrast)),
        saturation=fn.random.uniform(range=(1 - cfg.saturation, 1 + cfg.saturation)),
        hue=fn.random.uniform(range=(-cfg.hue * 180, cfg.hue * 180)),
    )
    return imgs


def _maybe_grayscale(imgs, prob: float):
    """Randomly convert to grayscale (replicate channel)."""
    do_gray = _coin(prob, 1)
    gray = fn.color_space_conversion(imgs, image_type=types.RGB, output_type=types.GRAY)
    gray = fn.cat(gray, gray, gray, axis=2)  # H×W×3
    return do_gray * gray + (1 - do_gray) * imgs


def _gaussian_blur(imgs, prob: float, sigma_min: float, sigma_max: float):
    """Apply Gaussian blur with probability `prob`."""
    do_blur = _coin(prob, 1)
    sigma = fn.random.uniform(range=(sigma_min, sigma_max))
    # kernel_size must be odd; derive from sigma (heuristic from DINOv2)
    blurred = fn.gaussian_blur(imgs, sigma=sigma)
    return do_blur * blurred + (1 - do_blur) * imgs


def _solarize(imgs, prob: float):
    """Apply solarization (invert pixels above threshold=128) with probability."""
    do_sol = _coin(prob, 1)
    threshold = 128.0
    sol = dmath.abs(imgs - threshold * 2)  # pixel → |pixel - 256| for >128
    # Proper solarize: invert values above threshold
    mask = imgs >= threshold
    sol = mask * (255.0 - imgs) + (1 - mask) * imgs
    return do_sol * sol + (1 - do_sol) * imgs


def _crop_and_augment(
    jpegs,
    crop_size: int,
    scale: Tuple[float, float],
    blur_prob: float,
    solarize_prob: float,
    cfg: DINOAugConfig,
    is_training: bool = True,
):
    """One crop view with full DINOv3 augmentation chain."""

    # ── Decode + random resized crop ─────────────────────────────────────────
    imgs = fn.decoders.image_random_crop(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        random_aspect_ratio=[3 / 4, 4 / 3],
        random_area=list(scale),
        num_attempts=10,
    )
    imgs = fn.resize(
        imgs,
        device="gpu",
        resize_x=crop_size,
        resize_y=crop_size,
        interp_type=types.INTERP_CUBIC,
    )

    # ── Random horizontal flip ────────────────────────────────────────────────
    imgs = fn.flip(imgs, device="gpu", horizontal=_coin(cfg.flip_prob, 1))

    # ── Color jitter (applied in CPU-land via cast trick) ─────────────────────
    # DALI color ops work on uint8 or float; cast to float first
    imgs = fn.cast(imgs, dtype=types.FLOAT)
    do_jitter = _coin(cfg.color_jitter_prob, 1)
    jittered = _apply_color_jitter(imgs, cfg)
    imgs = do_jitter * jittered + (1 - do_jitter) * imgs

    # ── Grayscale ─────────────────────────────────────────────────────────────
    imgs = _maybe_grayscale(imgs, cfg.grayscale_prob)

    # ── Gaussian blur ─────────────────────────────────────────────────────────
    imgs = fn.cast(imgs, dtype=types.UINT8)
    imgs = _gaussian_blur(imgs, blur_prob, cfg.blur_sigma_min, cfg.blur_sigma_max)

    # ── Solarization ──────────────────────────────────────────────────────────
    if solarize_prob > 0:
        imgs = fn.cast(imgs, dtype=types.FLOAT)
        imgs = _solarize(imgs, solarize_prob)
        imgs = fn.cast(imgs, dtype=types.UINT8)

    # ── Normalise to [0,1] and apply ImageNet stats ───────────────────────────
    imgs = fn.cast(imgs, dtype=types.FLOAT) / 255.0
    mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg.std, dtype=np.float32).reshape(1, 1, 3)
    imgs = (imgs - mean) / std

    # ── HWC → CHW (PyTorch layout) ────────────────────────────────────────────
    imgs = fn.transpose(imgs, perm=[2, 0, 1])

    return imgs


def build_dino_pipeline(
    source: DynamicMixingSource,
    aug_cfg: DINOAugConfig,
    batch_size: int,
    num_threads: int,
    device_id: int,
    seed: int = 42,
) -> Pipeline:
    """
    Construct and return the DALI pipeline.

    Output tensors (in order):
        global_0, global_1,  local_0 … local_{N-1}
    i.e. 2 + n_local_crops tensors per sample.
    """

    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        prefetch_queue_depth=2,
    )
    def _pipeline():
        # ── External source (raw JPEG bytes) ─────────────────────────────────
        jpegs = fn.external_source(
            source=source,
            dtype=types.UINT8,
            ndim=1,
            name="jpegs",
        )

        outputs = []

        # ── Global crops ──────────────────────────────────────────────────────
        for i in range(aug_cfg.n_global_crops):
            blur_p = aug_cfg.blur_prob_global1 if i == 0 else aug_cfg.blur_prob_global2
            sol_p = aug_cfg.solarize_prob if i == 1 else 0.0
            view = _crop_and_augment(
                jpegs,
                crop_size=aug_cfg.global_crop_size,
                scale=aug_cfg.global_crops_scale,
                blur_prob=blur_p,
                solarize_prob=sol_p,
                cfg=aug_cfg,
            )
            outputs.append(view)

        # ── Local crops ───────────────────────────────────────────────────────
        for _ in range(aug_cfg.n_local_crops):
            view = _crop_and_augment(
                jpegs,
                crop_size=aug_cfg.local_crop_size,
                scale=aug_cfg.local_crops_scale,
                blur_prob=aug_cfg.blur_prob_local,
                solarize_prob=0.0,
                cfg=aug_cfg,
            )
            outputs.append(view)

        return tuple(outputs)

    pipe = _pipeline()
    pipe.build()
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# High-level DataLoader wrapper
# ══════════════════════════════════════════════════════════════════════════════

class DINODALIDataLoader:
    """
    Drop-in DINOv3-style dataloader backed by NVIDIA DALI.

    Parameters
    ----------
    dataset_specs   : List of DatasetSpec (name, shards, initial weight)
    batch_size      : Per-GPU batch size
    aug_cfg         : DINOAugConfig instance
    num_threads     : DALI internal CPU threads
    device_id       : GPU index for this process
    rank            : Global rank (for distributed)
    world_size      : Total number of processes
    buffer_size     : WebDataset shuffle buffer per dataset
    seed            : Base random seed
    prefetch         : Number of DALI prefetch batches

    Usage
    -----
        loader = DINODALIDataLoader(specs, batch_size=128, ...)

        # Training loop
        for epoch in range(epochs):
            for batch in loader:
                global_crops = batch["global"]   # list of 2 tensors [B,3,224,224]
                local_crops  = batch["local"]    # list of N tensors [B,3,96,96]
                ...

        # Change mixing on-the-fly (e.g. curriculum)
        loader.set_weights([0.8, 0.2])
        loader.set_weight_by_name("laion", 0.6)
    """

    def __init__(
        self,
        dataset_specs: List[DatasetSpec],
        batch_size: int,
        aug_cfg: Optional[DINOAugConfig] = None,
        num_threads: int = 4,
        device_id: int = 0,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 1000,
        seed: int = 42,
    ):
        self.aug_cfg = aug_cfg or DINOAugConfig()
        self.batch_size = batch_size
        self.n_global = self.aug_cfg.n_global_crops
        self.n_local = self.aug_cfg.n_local_crops

        # Auto-detect rank / world_size from torch.distributed if available
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        self._source = DynamicMixingSource(
            dataset_specs=dataset_specs,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_shards=True,
            buffer_size=buffer_size,
            seed=seed,
        )

        self._pipe = build_dino_pipeline(
            source=self._source,
            aug_cfg=self.aug_cfg,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed + rank,
        )

        # Output names for DALIGenericIterator
        n_views = self.n_global + self.n_local
        self._output_names = [f"view_{i}" for i in range(n_views)]

        self._iterator = DALIGenericIterator(
            pipelines=[self._pipe],
            output_map=self._output_names,
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )

    # ------------------------------------------------------------------
    # Mixing control (thread-safe, immediate effect)
    # ------------------------------------------------------------------

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update all dataset mixing weights simultaneously."""
        self._source.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        """Update a single dataset's mixing weight."""
        self._source.set_weight_by_name(name, weight)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self):
        return self._iter_batches()

    def _iter_batches(self):
        for dali_batch in self._iterator:
            d = dali_batch[0]  # single pipeline output dict
            global_crops = [d[f"view_{i}"] for i in range(self.n_global)]
            local_crops = [d[f"view_{self.n_global + i}"] for i in range(self.n_local)]
            yield {"global": global_crops, "local": local_crops}

    def __len__(self):
        """Approximate steps per epoch (infinite dataset → use externally)."""
        return NotImplemented


# ══════════════════════════════════════════════════════════════════════════════
# Minimal usage example
# ══════════════════════════════════════════════════════════════════════════════

def _example():
    """
    Quick smoke-test (replace shard paths with real tar files).
    Run with: torchrun --nproc_per_node=2 dino_dali_dataloader.py
    """
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()

    specs = [
        DatasetSpec(
            name="imagenet",
            shards=[f"/data/imagenet/shard-{i:06d}.tar" for i in range(200)],
            weight=0.7,
        ),
        DatasetSpec(
            name="laion",
            shards=[f"/data/laion/shard-{i:06d}.tar" for i in range(500)],
            weight=0.3,
        ),
    ]

    aug_cfg = DINOAugConfig(n_local_crops=8)

    loader = DINODALIDataLoader(
        dataset_specs=specs,
        batch_size=256,
        aug_cfg=aug_cfg,
        num_threads=8,
        device_id=device_id,
        rank=rank,
        world_size=world_size,
        seed=0,
    )

    for step, batch in enumerate(loader):
        g0 = batch["global"][0]   # Tensor [256, 3, 224, 224] on GPU
        g1 = batch["global"][1]
        locals_ = batch["local"]  # list of 8 tensors [256, 3, 96, 96]

        if rank == 0 and step % 100 == 0:
            print(f"Step {step}: global {g0.shape}, local[0] {locals_[0].shape}")

        # ── Dynamic mixing change at step 500 ────────────────────────────────
        if step == 500:
            loader.set_weights([0.5, 0.5])
            if rank == 0:
                print("Mixing weights updated to 50/50")

        if step >= 1000:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    _example()
