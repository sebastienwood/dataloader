"""
dino_loader.config
==================
All configuration lives here.  No logic — pure dataclasses.
Serialised to / from JSON for checkpointing (no pickle fragility).

Changes vs previous version
----------------------------
[CFG-1] DatasetSpec enriched (DinoV3 alignment):
        - shard_quality_scores: Optional[List[float]] — per-shard quality for
          weighted shard sampling (replaces uniform random.choices at shard level).
        - min_sample_quality: Optional[float] — hard filter threshold applied
          per sample via .json sidecar metadata.  Requires wds.TarIterator
          extraction (see mixing_source.py).
        - metadata_key: str — sidecar extension to extract alongside JPEGs.
          Defaults to "json" (standard WebDataset convention).
        - mean / std: Optional per-dataset normalisation statistics.  When None,
          falls back to DINOAugConfig.mean / DINOAugConfig.std (ImageNet).
          Allows LAION-specific stats without changing global aug config.

[CFG-2] DINOAugConfig additions:
        - preserve_aspect_ratio: bool — use resize-then-crop (aspect-ratio-safe)
          instead of fn.resize with fixed output size.  Maps to DALI
          fn.resize(mode="not_smaller") + fn.crop in pipeline.py.
        - resolution_schedule: Optional[List[Tuple[int,int]]] — list of
          (epoch, global_crop_size) pairs for progressive resolution training.
          The loader applies these automatically via set_resolution() without
          rebuilding the DALI pipeline (zero downtime).
        - max_global_crop_size / max_local_crop_size: int — upper bounds used
          to pre-allocate DALI nvjpeg buffers and output tensors at the maximum
          planned resolution, avoiding GPU memory re-allocation during training.

[CFG-3] LoaderConfig additions:
        - shuffle_buffer_size: int — intra-shard sample shuffle buffer depth.
          Previously reserved as a comment; now wired in ShardIterator.
          Default 512: large enough to break within-shard web-crawl correlations
          without exceeding per-rank RAM budgets.
        - stateful_dataloader: bool — expose state_dict() / load_state_dict()
          on DINODataLoader, aligning with the PyTorch StatefulDataLoader
          interface (torchdata ≥ 0.8).  Enables integration with Lightning,
          torchtitan and other frameworks that call these methods automatically.

[CFG-4] CheckpointState additions:
        - global_crop_size / local_crop_size: persisted so that a resumed run
          starts at the correct resolution without re-reading the schedule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    """
    One WebDataset source with mixing weight and optional quality metadata.

    Parameters
    ----------
    name
        Human-readable identifier, used in logs and checkpoint state.
    shards
        List of absolute shard paths (.tar files on Lustre).
    weight
        Initial mixing weight (re-normalised automatically; need not sum to 1).
    shard_quality_scores
        Optional per-shard quality score in [0, 1].  When provided,
        ShardIterator samples shards proportionally to these scores rather
        than uniformly.  Scores are re-normalised internally.
        Length must match ``len(shards)`` if provided.
    min_sample_quality
        Hard filter: samples whose .json sidecar ``quality_score`` field is
        below this threshold are discarded before entering the DALI pipeline.
        Set to None to disable (default, no filtering).
    metadata_key
        WebDataset sidecar extension to extract alongside .jpg files.
        Set to None to skip sidecar extraction (legacy behaviour, faster).
    mean
        Per-channel normalisation mean for this dataset.  When None, the
        global DINOAugConfig.mean is used (ImageNet stats).
    std
        Per-channel normalisation std for this dataset.  When None, the
        global DINOAugConfig.std is used (ImageNet stats).
    """
    name:                 str
    shards:               List[str]
    weight:               float                = 1.0
    shard_quality_scores: Optional[List[float]] = None
    min_sample_quality:   Optional[float]       = None
    metadata_key:         Optional[str]         = "json"
    mean:                 Optional[Tuple[float, float, float]] = None
    std:                  Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if not self.shards:
            raise ValueError(f"DatasetSpec '{self.name}' has no shards.")
        if self.weight < 0:
            raise ValueError(f"DatasetSpec '{self.name}': weight must be ≥ 0.")
        if self.shard_quality_scores is not None:
            if len(self.shard_quality_scores) != len(self.shards):
                raise ValueError(
                    f"DatasetSpec '{self.name}': shard_quality_scores length "
                    f"({len(self.shard_quality_scores)}) must match shards "
                    f"({len(self.shards)})."
                )
            if any(s < 0 for s in self.shard_quality_scores):
                raise ValueError(
                    f"DatasetSpec '{self.name}': all shard_quality_scores must be ≥ 0."
                )
        if self.min_sample_quality is not None and not (0.0 <= self.min_sample_quality <= 1.0):
            raise ValueError(
                f"DatasetSpec '{self.name}': min_sample_quality must be in [0, 1]."
            )


# ── Augmentation ──────────────────────────────────────────────────────────────

@dataclass
class DINOAugConfig:
    """
    DINOv3 multi-crop augmentation parameters.

    Defaults match the DINOv2 paper (Oquab et al., 2023) §A.1 and the
    DinoV3 codebase (augmentations.py).
    """
    # ── Global crops ──────────────────────────────────────────────────────────
    global_crops_scale:   Tuple[float, float] = (0.32, 1.0)
    global_crop_size:     int   = 224
    n_global_crops:       int   = 2

    # ── Local crops ───────────────────────────────────────────────────────────
    local_crops_scale:    Tuple[float, float] = (0.05, 0.32)
    local_crop_size:      int   = 96
    n_local_crops:        int   = 8

    # ── Aspect ratio ──────────────────────────────────────────────────────────
    # [CFG-2] When True: fn.resize(mode="not_smaller") + fn.crop preserves
    # the original image aspect ratio.  When False (legacy): fn.resize with
    # fixed resize_x / resize_y (may distort non-square images).
    preserve_aspect_ratio: bool = True

    # ── Progressive resolution schedule ──────────────────────────────────────
    # [CFG-2] List of (epoch, global_crop_size) pairs, applied in order.
    # Example: [(0, 224), (10, 448), (30, 518)]
    # The loader calls set_resolution() automatically at epoch boundaries.
    # The DALI pipeline is NOT rebuilt; resize is driven by a dynamic DataNode.
    resolution_schedule:  Optional[List[Tuple[int, int]]] = None

    # ── Pre-allocation ceilings ───────────────────────────────────────────────
    # [CFG-2] nvjpeg decode buffers and output tensors are allocated once at
    # max_global_crop_size.  Set to the largest resolution in the schedule.
    # Defaults to global_crop_size (no schedule = no overhead).
    max_global_crop_size: Optional[int] = None
    max_local_crop_size:  Optional[int] = None

    # ── Color jitter ──────────────────────────────────────────────────────────
    color_jitter_prob:    float = 0.8
    brightness:           float = 0.4
    contrast:             float = 0.4
    saturation:           float = 0.2
    hue:                  float = 0.1

    # ── Grayscale ─────────────────────────────────────────────────────────────
    grayscale_prob:       float = 0.2

    # ── Gaussian blur (intentionally asymmetric across views, DINOv2 §A.1) ───
    blur_prob_global1:    float = 1.0
    blur_prob_global2:    float = 0.1
    blur_prob_local:      float = 0.5
    blur_sigma_min:       float = 0.1
    blur_sigma_max:       float = 2.0

    # ── Solarisation (second global crop only) ────────────────────────────────
    solarize_prob:        float = 0.2

    # ── Flip ──────────────────────────────────────────────────────────────────
    flip_prob:            float = 0.5

    # ── ImageNet normalisation (global default; per-dataset override in DatasetSpec) ──
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self):
        if self.max_global_crop_size is None:
            object.__setattr__(self, "max_global_crop_size", self.global_crop_size)
        if self.max_local_crop_size is None:
            object.__setattr__(self, "max_local_crop_size", self.local_crop_size)
        if self.resolution_schedule is not None:
            for epoch, size in self.resolution_schedule:
                if epoch < 0 or size <= 0:
                    raise ValueError(
                        f"resolution_schedule entry ({epoch}, {size}) is invalid."
                    )
            # Ensure schedule is sorted by epoch
            object.__setattr__(
                self, "resolution_schedule",
                sorted(self.resolution_schedule, key=lambda x: x[0])
            )

    @property
    def n_views(self) -> int:
        return self.n_global_crops + self.n_local_crops

    def crop_size_at_epoch(self, epoch: int) -> int:
        """Return the global crop size dictated by the resolution schedule."""
        if not self.resolution_schedule:
            return self.global_crop_size
        size = self.global_crop_size
        for ep, s in self.resolution_schedule:
            if epoch >= ep:
                size = s
        return size


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
    shm_warn_threshold:      float = 0.85  # warn at 85% /dev/shm utilisation

    # ── Shard extraction ──────────────────────────────────────────────────────
    shard_extraction_workers: int  = 4
    cpu_affinity_enabled:     bool = False

    # ── Intra-shard shuffle ───────────────────────────────────────────────────
    # [CFG-3] Reservoir-style shuffle buffer within ShardIterator.next_sample().
    # Breaks within-shard web-crawl correlations (consecutive images from the
    # same URL cluster).  Set to 0 to disable.
    shuffle_buffer_size:      int  = 512

    # ── DALI pipeline ─────────────────────────────────────────────────────────
    dali_cpu_queue:          int   = 8
    dali_gpu_queue:          int   = 6
    dali_num_threads:        int   = 8
    hw_decoder_load:         float = 0.90

    # ── Output dtype / FP8 ────────────────────────────────────────────────────
    use_fp8_output:          bool  = True
    output_dtype:            str   = "bf16"   # "bf16" | "fp32"

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir:          str   = "/checkpoint/dino/dl"
    checkpoint_every_steps:  int   = 500

    # ── Distributed topology ──────────────────────────────────────────────────
    force_topology:          Optional[str] = None   # "nvl72" | "pcie" | None

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed:                    int   = 0

    # ── StatefulDataLoader interface ──────────────────────────────────────────
    # [CFG-3] When True, DINODataLoader exposes state_dict() / load_state_dict()
    # matching the torchdata StatefulDataLoader protocol.  Enables transparent
    # integration with Lightning, torchtitan, and other frameworks.
    stateful_dataloader:     bool  = True

    def __post_init__(self):
        valid_dtypes = {"bf16", "fp32"}
        if self.output_dtype not in valid_dtypes:
            raise ValueError(
                f"output_dtype must be one of {valid_dtypes}, got '{self.output_dtype}'"
            )
        if not 0.0 <= self.shm_warn_threshold <= 1.0:
            raise ValueError("shm_warn_threshold must be in [0, 1].")
        if not 0.0 <= self.hw_decoder_load <= 1.0:
            raise ValueError("hw_decoder_load must be in [0, 1].")


# ── Checkpoint state ──────────────────────────────────────────────────────────

@dataclass
class CheckpointState:
    """
    Minimal dataloader state for resume.

    JSON-serialised (not pickle) for stability across Python versions and
    conda environments.

    [CFG-4] Added global_crop_size / local_crop_size so that resumed runs
    start at the correct resolution without re-applying the schedule from
    epoch 0.
    """
    step:              int
    epoch:             int
    dataset_names:     List[str]
    mixing_weights:    List[float]
    global_crop_size:  int = 224   # [CFG-4]
    local_crop_size:   int = 96    # [CFG-4]

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.rename(path)   # POSIX-atomic

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        data = json.loads(path.read_text())
        return cls(**data)
