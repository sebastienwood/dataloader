"""
dino_loader.config
==================
All configuration lives here.  No logic — pure dataclasses.
Serialised to / from JSON for checkpointing (no pickle fragility).

Changes vs previous version
----------------------------
[FIX-19] Added ``shard_extraction_workers`` to ``LoaderConfig``.
         The field was claimed to have been "threaded through" in mixing_source
         [F-3] but was never actually added to the dataclass, making it
         impossible to configure without patching source.  Default is 4 (up
         from the previous hard-coded 2) to prevent DALI starvation when
         workers block on inotify for cold shards at epoch start. [FIX-5]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    """One WebDataset source with an initial mixing weight."""
    name:    str
    shards:  List[str]
    weight:  float = 1.0

    def __post_init__(self):
        if not self.shards:
            raise ValueError(f"DatasetSpec '{self.name}' has no shards.")
        if self.weight < 0:
            raise ValueError(f"DatasetSpec '{self.name}': weight must be ≥ 0.")


# ── Augmentation ──────────────────────────────────────────────────────────────

@dataclass
class DINOAugConfig:
    """DINOv3 multi-crop augmentation parameters."""
    # Global crops
    global_crops_scale:   Tuple[float, float] = (0.32, 1.0)
    global_crop_size:     int   = 224
    n_global_crops:       int   = 2
    # Local crops
    local_crops_scale:    Tuple[float, float] = (0.05, 0.32)
    local_crop_size:      int   = 96
    n_local_crops:        int   = 8
    # Color jitter
    color_jitter_prob:    float = 0.8
    brightness:           float = 0.4
    contrast:             float = 0.4
    saturation:           float = 0.2
    hue:                  float = 0.1
    # Grayscale
    grayscale_prob:       float = 0.2
    # Gaussian blur — intentionally asymmetric across views (DINOv2 paper §A.1)
    blur_prob_global1:    float = 1.0
    blur_prob_global2:    float = 0.1
    blur_prob_local:      float = 0.5
    blur_sigma_min:       float = 0.1
    blur_sigma_max:       float = 2.0
    # Solarisation — second global crop only
    solarize_prob:        float = 0.2
    # Flip
    flip_prob:            float = 0.5
    # ImageNet normalisation
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def n_views(self) -> int:
        return self.n_global_crops + self.n_local_crops


# ── Loader ────────────────────────────────────────────────────────────────────

@dataclass
class LoaderConfig:
    """
    Single config object covering all subsystems.
    Sensible defaults for a GB200 NVL72 node; adjust for your cluster.
    """
    # ── Shard I/O ─────────────────────────────────────────────────────────────
    node_shm_gb:             float = 128.0
    shard_prefetch_window:   int   = 64
    shard_timeout_s:         float = 300.0

    # ── Shard extraction ──────────────────────────────────────────────────────
    # [FIX-19 / FIX-5] This field was referenced in mixing_source.py but never
    # defined here.  Raising the default from 2 → 4 prevents DALI starvation
    # when workers block on inotify waiting for cold shards at epoch start.
    # Rule of thumb: set to ≥ shard_prefetch_window / 16, minimum 4.
    shard_extraction_workers: int  = 4

    # ── DALI pipeline ─────────────────────────────────────────────────────────
    dali_cpu_queue:          int   = 8
    dali_gpu_queue:          int   = 6
    dali_num_threads:        int   = 8
    hw_decoder_load:         float = 0.90

    # ── Output format ─────────────────────────────────────────────────────────
    use_fp8_output:          bool  = True
    output_dtype:            str   = "bf16"

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir:          str   = "/checkpoint/dino/dl"
    checkpoint_every_steps:  int   = 500

    # ── NCCL / topology ───────────────────────────────────────────────────────
    force_topology:          Optional[str] = None

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed:                    int   = 0
    shuffle_buffer:          int   = 2000

    def __post_init__(self):
        if self.output_dtype not in ("bf16", "fp32"):
            raise ValueError(
                f"output_dtype must be 'bf16' or 'fp32', got {self.output_dtype!r}"
            )
        if self.shard_extraction_workers < 1:
            raise ValueError("shard_extraction_workers must be ≥ 1")


# ── Checkpoint state (JSON-serialisable) ─────────────────────────────────────

@dataclass
class CheckpointState:
    step:           int
    epoch:          int
    mixing_weights: List[float]
    dataset_names:  List[str]

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        return cls(**json.loads(path.read_text()))
