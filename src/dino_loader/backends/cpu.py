"""
dino_loader.backends.cpu
========================
Pure-Python / PyTorch CPU backend.

Replaces every NVIDIA-specific component with dependency-free equivalents so
the full dino_loader stack can be exercised on any machine — a developer
laptop, a GitHub Actions runner, or a pytest job without GPU allocation.

Stage mapping
-------------
Stage 1  Shard I/O          → InProcessShardCache  (dict in RAM, no /dev/shm)
Stage 2  JPEG Extraction    → unchanged (wds.TarIterator / legacy parser)
Stage 3  Augmentation       → CPUAugPipeline + torchvision transforms
Stage 4  H2D Transfer       → NullH2DStream        (tensors stay on CPU)
Stage 5  FP8 Quantisation   → NullFP8Formatter     (identity / BF16 passthrough)
Dist.    init_distributed() → StubDistribEnv       (rank=0, world=1, no NCCL)

Design principles
-----------------
* Zero hard dependencies beyond ``torch`` and ``Pillow``.
  torchvision is used when present; a fallback PIL-only path handles the
  colour-jitter and blur ops when torchvision is absent.
* The CPU augmentation reproduces the *statistical* distribution of the DALI
  augmentation closely enough that models trained on DALI produce the same
  qualitative outputs when evaluated against CPU-generated batches.
* All public objects implement the same method signatures as their DALI / SLURM
  counterparts so that loader.py can use either backend without modification.
* The InProcessShardCache supports the full ``get_view`` context-manager
  protocol, allowing ShardIterator to run unmodified.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

from dino_loader.config import DINOAugConfig
from dino_loader.distributed import ClusterTopology, DistribEnv

log = logging.getLogger(__name__)

try:
    import torchvision.transforms.functional as TF
    import torchvision.transforms as TV
    HAS_TV = True
except ImportError:
    HAS_TV = False
    log.warning(
        "torchvision not installed — CPU backend uses PIL-only augmentation. "
        "Install torchvision for higher-fidelity transforms."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — In-process shard cache
# ══════════════════════════════════════════════════════════════════════════════

class InProcessShardCache:
    """
    Shard cache backed by an in-process LRU dict.

    Drop-in replacement for ``NodeSharedShardCache``.  All /dev/shm, inotify,
    asyncio, and SLURM-specific code is replaced with a simple threading.Lock
    + OrderedDict LRU.

    Thread safety: get/get_view/prefetch are safe to call from multiple
    extraction worker threads simultaneously.
    """

    def __init__(
        self,
        job_id:          str   = "cpu_test",
        node_master:     bool  = True,
        max_gb:          float = 1.0,
        prefetch_window: int   = 4,
        timeout_s:       float = 30.0,
        warn_threshold:  float = 0.85,
    ) -> None:
        self._max_bytes     = int(max_gb * (1 << 30))
        self._warn_threshold = warn_threshold
        self._lru:   OrderedDict[str, bytes] = OrderedDict()
        self._total: int = 0
        self._lock   = threading.Lock()

    # ── Public API (mirrors NodeSharedShardCache) ─────────────────────────────

    def prefetch(self, shard_path: str) -> None:
        """No-op: no async pre-fetch on the CPU backend."""
        pass

    def get(self, shard_path: str) -> bytes:
        """Return raw bytes for *shard_path*, reading from disk if not cached."""
        with self._lock:
            if shard_path in self._lru:
                self._lru.move_to_end(shard_path)
                return self._lru[shard_path]

        data = self._read(shard_path)

        with self._lock:
            self._evict_for(len(data))
            self._lru[shard_path] = data
            self._total += len(data)

        return data

    @contextlib.contextmanager
    def get_view(self, shard_path: str) -> Iterator[memoryview]:
        """Yield a memoryview into the cached shard bytes."""
        data = self.get(shard_path)
        yield memoryview(data)

    @property
    def utilisation(self) -> float:
        if self._max_bytes == 0:
            return 0.0
        with self._lock:
            return self._total / self._max_bytes

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _read(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _evict_for(self, incoming: int) -> None:
        """Evict LRU entries until there is room for *incoming* bytes."""
        while self._lru and self._total + incoming > self._max_bytes:
            _, evicted = self._lru.popitem(last=False)
            self._total -= len(evicted)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — CPU augmentation pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _random_resized_crop(
    img: Image.Image,
    size: int,
    scale: Tuple[float, float],
    ratio: Tuple[float, float] = (3 / 4, 4 / 3),
) -> Image.Image:
    """RandomResizedCrop without torchvision dependency."""
    if HAS_TV:
        return TV.RandomResizedCrop(
            size   = size,
            scale  = scale,
            ratio  = ratio,
            interpolation = TV.InterpolationMode.BICUBIC,
        )(img)

    # Pure PIL fallback
    w, h = img.size
    area = w * h
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect      = random.uniform(*ratio)
        cw = int(round((target_area * aspect) ** 0.5))
        ch = int(round((target_area / aspect) ** 0.5))
        if 0 < cw <= w and 0 < ch <= h:
            x = random.randint(0, w - cw)
            y = random.randint(0, h - ch)
            return img.crop((x, y, x + cw, y + ch)).resize(
                (size, size), Image.BICUBIC,
            )
    # Fallback: centre crop
    short = min(w, h)
    x, y  = (w - short) // 2, (h - short) // 2
    return img.crop((x, y, x + short, y + short)).resize(
        (size, size), Image.BICUBIC
    )


def _color_jitter(
    img: Image.Image,
    brightness: float,
    contrast:   float,
    saturation: float,
    hue:        float,
    prob:       float,
) -> Image.Image:
    if random.random() > prob:
        return img
    if HAS_TV:
        return TV.ColorJitter(
            brightness = brightness,
            contrast   = contrast,
            saturation = saturation,
            hue        = hue,
        )(img)

    # PIL-only path
    from PIL import ImageEnhance
    def _enhance(img, cls, lo, hi):
        factor = random.uniform(max(0.0, 1 - lo), 1 + hi)
        return cls(img).enhance(factor)

    ops = [
        (ImageEnhance.Brightness, brightness, brightness),
        (ImageEnhance.Contrast,   contrast,   contrast),
        (ImageEnhance.Color,      saturation, saturation),
    ]
    random.shuffle(ops)
    for cls, lo, hi in ops:
        img = _enhance(img, cls, lo, hi)
    return img


def _gaussian_blur(img: Image.Image, sigma_min: float, sigma_max: float, prob: float) -> Image.Image:
    if random.random() > prob:
        return img
    sigma = random.uniform(sigma_min, sigma_max)
    if HAS_TV:
        # kernel_size must be odd; derive from sigma as DALI does
        ks = max(3, int(sigma * 4 + 1) | 1)
        return TF.gaussian_blur(img, kernel_size=ks, sigma=sigma)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def _solarize(img: Image.Image, threshold: int = 128) -> Image.Image:
    if HAS_TV:
        return TF.solarize(img, threshold)
    from PIL import ImageOps
    return ImageOps.solarize(img, threshold)


def _to_tensor_normalized(
    img: Image.Image,
    mean: Tuple[float, float, float],
    std:  Tuple[float, float, float],
) -> torch.Tensor:
    """Convert PIL image to a normalised float tensor in CHW layout."""
    if HAS_TV:
        t = TF.to_tensor(img)               # [0,1] float32, CHW
        return TF.normalize(t, mean=list(mean), std=list(std))

    # Fallback
    arr = np.asarray(img, dtype=np.float32) / 255.0   # HWC
    arr = (arr - np.array(mean)) / np.array(std)
    return torch.from_numpy(arr.transpose(2, 0, 1))   # CHW


def _augment_one(
    jpeg_bytes: bytes,
    aug_cfg:    DINOAugConfig,
    crop_size:  int,
    scale:      Tuple[float, float],
    blur_prob:  float,
    sol_prob:   float,
) -> torch.Tensor:
    """
    Apply DINOv2-style augmentation to a single JPEG and return a CHW tensor.
    Matches the statistical distribution of the DALI pipeline closely.
    """
    try:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    except Exception:
        # Corrupt JPEG: return a zero tensor
        return torch.zeros(3, crop_size, crop_size)

    # 1. RandomResizedCrop
    img = _random_resized_crop(img, crop_size, scale)

    # 2. RandomHorizontalFlip
    if random.random() < aug_cfg.flip_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 3. ColorJitter
    img = _color_jitter(
        img,
        aug_cfg.brightness,
        aug_cfg.contrast,
        aug_cfg.saturation,
        aug_cfg.hue,
        aug_cfg.color_jitter_prob,
    )

    # 4. RandomGrayscale
    if random.random() < aug_cfg.grayscale_prob:
        img = img.convert("L").convert("RGB")

    # 5. GaussianBlur
    img = _gaussian_blur(img, aug_cfg.blur_sigma_min, aug_cfg.blur_sigma_max, blur_prob)

    # 6. Solarize
    if sol_prob > 0 and random.random() < sol_prob:
        img = _solarize(img)

    # 7. Normalize → tensor
    return _to_tensor_normalized(img, aug_cfg.mean, aug_cfg.std)


class CPUAugPipeline:
    """
    Pure-Python DINOv2 multi-crop augmentation pipeline.

    Implements the same augmentation graph as the DALI pipeline
    (``pipeline.py``) but runs on CPU using PIL + (optionally) torchvision.

    The pipeline is driven by a ``MixingSource``-compatible callable
    (``source``) that returns batches of raw JPEG bytes.
    """

    def __init__(
        self,
        source:         Any,
        aug_cfg:        DINOAugConfig,
        batch_size:     int,
        resolution_src: Any,
        seed:           int = 0,
    ) -> None:
        self._source        = source
        self._aug_cfg       = aug_cfg
        self._batch_size    = batch_size
        self._resolution_src = resolution_src
        random.seed(seed)

    def run_one_batch(self) -> Dict[str, torch.Tensor]:
        """
        Execute one augmentation step.

        Returns a dict matching the DALIGenericIterator output format:
        ``{"view_0": Tensor[B,C,H,W], ..., "view_N": Tensor[B,C,H,W]}``
        """
        # Query current resolution
        global_size_arr, local_size_arr = self._resolution_src()
        global_size = int(global_size_arr)
        local_size  = int(local_size_arr)

        # Fetch JPEG batch from MixingSource
        jpeg_batch: List[np.ndarray] = self._source()
        assert len(jpeg_batch) == self._batch_size, (
            f"MixingSource returned {len(jpeg_batch)} samples, expected {self._batch_size}"
        )

        cfg = self._aug_cfg
        output: Dict[str, List[torch.Tensor]] = {
            f"view_{i}": [] for i in range(cfg.n_views)
        }

        for jpeg_arr in jpeg_batch:
            jpeg_bytes = bytes(jpeg_arr)
            view_idx   = 0

            # Global crops
            for i in range(cfg.n_global_crops):
                blur_p = cfg.blur_prob_global1 if i == 0 else cfg.blur_prob_global2
                sol_p  = cfg.solarize_prob if i == 1 else 0.0
                t = _augment_one(
                    jpeg_bytes, cfg,
                    crop_size  = global_size,
                    scale      = cfg.global_crops_scale,
                    blur_prob  = blur_p,
                    sol_prob   = sol_p,
                )
                output[f"view_{view_idx}"].append(t)
                view_idx += 1

            # Local crops
            for _ in range(cfg.n_local_crops):
                t = _augment_one(
                    jpeg_bytes, cfg,
                    crop_size  = local_size,
                    scale      = cfg.local_crops_scale,
                    blur_prob  = cfg.blur_prob_local,
                    sol_prob   = 0.0,
                )
                output[f"view_{view_idx}"].append(t)
                view_idx += 1

        return {k: torch.stack(v) for k, v in output.items()}


class CPUPipelineIterator:
    """
    Wraps a ``CPUAugPipeline`` in the ``DALIGenericIterator`` API expected by
    ``loader.py``.

    The DALI iterator yields ``[{view_name: tensor, ...}]`` (a list of length 1,
    one element per pipeline).  This class mirrors that structure so
    ``_iter_batches()`` in loader.py requires zero modification.
    """

    def __init__(
        self,
        pipeline:   CPUAugPipeline,
        output_map: List[str],
        batch_size: int,
    ) -> None:
        self._pipe      = pipeline
        self._output_map = output_map
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self) -> List[Dict[str, torch.Tensor]]:
        if self._exhausted:
            raise StopIteration
        try:
            batch_dict = self._pipe.run_one_batch()
            return [batch_dict]   # wrap in list — DALI returns list of pipeline outputs
        except StopIteration:
            self._exhausted = True
            raise

    def reset(self) -> None:
        """Called by loader.py's set_epoch(). No-op for the CPU backend."""
        self._exhausted = False


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Null H2D stream
# ══════════════════════════════════════════════════════════════════════════════

class NullH2DStream:
    """
    No-op H2DStream. Tensors already on CPU; no transfer needed.

    Implements the same context-manager and method API as ``H2DStream`` so
    loader.py works without modification.
    """

    def __init__(self, device: torch.device, topo: ClusterTopology) -> None:
        self._device = device

    @contextlib.contextmanager
    def transfer(self, cpu_batch: Dict[str, List[torch.Tensor]]):
        yield cpu_batch

    def send(self, cpu_batch: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        return cpu_batch

    def wait(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Null FP8 formatter
# ══════════════════════════════════════════════════════════════════════════════

class NullFP8Formatter:
    """
    Identity FP8 formatter — returns the input tensor unchanged.

    Allows ``use_fp8_output=True`` to be set in LoaderConfig without crashing
    on CPU-only machines where Transformer Engine is unavailable.
    """

    def quantise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


# ══════════════════════════════════════════════════════════════════════════════
# Distributed stub
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StubClusterTopology:
    """
    Minimal ClusterTopology that satisfies the attribute access in memory.py,
    loader.py, and distributed.py without probing any hardware.
    """
    nvlink_gen:              int  = 0
    nvlink_lanes_per_gpu:    int  = 0
    gpus_per_nvlink_domain:  int  = 1
    is_grace_blackwell:      bool = False
    has_pcie:                bool = False
    has_infiniband:          bool = False
    has_sharp:               bool = False
    has_nvlink_sharp:        bool = False
    cuda_major:              int  = 0
    cuda_minor:              int  = 0

    @property
    def is_nvl72(self) -> bool:
        return False

    @property
    def label(self) -> str:
        return "CPU-stub"


@dataclass
class StubDistribEnv:
    """
    Mimics ``DistribEnv`` for single-rank CPU testing.

    No NCCL, no GPU, no SLURM env vars required.
    """
    rank:             int               = 0
    world_size:       int               = 1
    local_rank:       int               = 0
    local_world_size: int               = 1
    topology:         StubClusterTopology = None  # type: ignore

    def __post_init__(self):
        if self.topology is None:
            self.topology = StubClusterTopology()


# ══════════════════════════════════════════════════════════════════════════════
# CPUBackend — assembles all stages
# ══════════════════════════════════════════════════════════════════════════════

class CPUBackend:
    """
    Concrete backend: pure-Python CPU path.

    Used by ``get_backend("cpu")`` and as the automatic fallback when DALI is
    not installed.
    """

    # ── BackendProtocol identity ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def supports_fp8(self) -> bool:
        return False

    @property
    def supports_gpu(self) -> bool:
        return False

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def build_shard_cache(
        self,
        job_id:          str   = "cpu_test",
        node_master:     bool  = True,
        max_gb:          float = 1.0,
        prefetch_window: int   = 4,
        timeout_s:       float = 30.0,
        warn_threshold:  float = 0.85,
    ) -> InProcessShardCache:
        return InProcessShardCache(
            job_id          = job_id,
            node_master     = node_master,
            max_gb          = max_gb,
            prefetch_window = prefetch_window,
            timeout_s       = timeout_s,
            warn_threshold  = warn_threshold,
        )

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def build_pipeline(
        self,
        source,
        aug_cfg,
        batch_size:      int,
        num_threads:     int,
        device_id:       int,
        resolution_src,
        hw_decoder_load: float = 0.90,
        cpu_queue:       int   = 8,
        gpu_queue:       int   = 6,
        seed:            int   = 42,
    ) -> CPUAugPipeline:
        log.info(
            "CPUBackend: building CPU augmentation pipeline "
            "(batch=%d, global=%d, local=%d)",
            batch_size,
            aug_cfg.global_crop_size,
            aug_cfg.local_crop_size,
        )
        return CPUAugPipeline(
            source         = source,
            aug_cfg        = aug_cfg,
            batch_size     = batch_size,
            resolution_src = resolution_src,
            seed           = seed,
        )

    def build_pipeline_iterator(
        self,
        pipeline:   CPUAugPipeline,
        output_map: List[str],
        batch_size: int,
    ) -> CPUPipelineIterator:
        return CPUPipelineIterator(
            pipeline   = pipeline,
            output_map = output_map,
            batch_size = batch_size,
        )

    # ── Stage 4 ───────────────────────────────────────────────────────────────

    def build_h2d_stream(self, device: torch.device, topo) -> NullH2DStream:
        return NullH2DStream(device=device, topo=topo)

    # ── Stage 5 ───────────────────────────────────────────────────────────────

    def build_fp8_formatter(self) -> NullFP8Formatter:
        return NullFP8Formatter()

    # ── Distributed ──────────────────────────────────────────────────────────

    def init_distributed(
        self,
        rank:             int = 0,
        world_size:       int = 1,
        local_rank:       int = 0,
        local_world_size: int = 1,
        force_topology:   Optional[str] = None,
    ) -> StubDistribEnv:
        return StubDistribEnv(
            rank             = rank,
            world_size       = world_size,
            local_rank       = local_rank,
            local_world_size = local_world_size,
        )
