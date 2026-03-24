"""
dino_loader.backends.cpu
========================
Pure-Python / PyTorch CPU backend.

Used for unit tests, CI, and rapid iteration on laptops. All DALI-specific
logic lives in DALIBackend; this backend replaces it with PIL + torchvision.

[CPU-AUG-1] build_pipeline dispatches on AugmentationSpec subtype via
            isinstance chains (replacing structural pattern matching which
            requires __match_args__ and is less explicit).
[CPU-AUG-2] EvalAugSpec → CPUEvalPipeline: deterministic resize + centre-crop.
[CPU-AUG-3] LeJEPAAugSpec → CPULeJEPAPipeline: context + N target crops.
[CPU-AUG-4] UserAugSpec → CPUUserAugPipeline: decodes JPEG with PIL,
            normalises, then calls aug_fn.
"""

import contextlib
import io
import logging
import random
import threading
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter

from dino_loader.augmentation import (
    AugmentationSpec,
    DinoV2AugSpec,
    EvalAugSpec,
    LeJEPAAugSpec,
    UserAugSpec,
)
from dino_loader.config import DINOAugConfig

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


# Stage 1 — In-process shard cache

class InProcessShardCache:
    """Shard cache backed by an in-process LRU dict."""

    def __init__(
        self,
        job_id:          str   = "cpu_test",
        node_master:     bool  = True,
        max_gb:          float = 1.0,
        prefetch_window: int   = 4,
        timeout_s:       float = 30.0,
        warn_threshold:  float = 0.85,
    ) -> None:
        self._max_bytes      = int(max_gb * (1 << 30))
        self._warn_threshold = warn_threshold
        self._lru:   OrderedDict[str, bytes] = OrderedDict()
        self._total: int = 0
        self._lock   = threading.Lock()

    def prefetch(self, shard_path: str) -> None:
        pass

    def get(self, shard_path: str) -> bytes:
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
        data = self.get(shard_path)
        yield memoryview(data)

    @property
    def utilisation(self) -> float:
        if self._max_bytes == 0:
            return 0.0
        with self._lock:
            return self._total / self._max_bytes

    @staticmethod
    def _read(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _evict_for(self, incoming: int) -> None:
        while self._lru and self._total + incoming > self._max_bytes:
            _, evicted = self._lru.popitem(last=False)
            self._total -= len(evicted)


# Augmentation helpers

def _random_resized_crop(
    img:   Image.Image,
    size:  int,
    scale: tuple[float, float],
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> Image.Image:
    if HAS_TV:
        return TV.RandomResizedCrop(
            size          = size,
            scale         = scale,
            ratio         = ratio,
            interpolation = TV.InterpolationMode.BICUBIC,
        )(img)
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
            return img.crop((x, y, x + cw, y + ch)).resize((size, size), Image.BICUBIC)
    short = min(w, h)
    x, y  = (w - short) // 2, (h - short) // 2
    return img.crop((x, y, x + short, y + short)).resize((size, size), Image.BICUBIC)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    if HAS_TV:
        return TV.CenterCrop(size)(img)
    w, h = img.size
    x, y = (w - size) // 2, (h - size) // 2
    return img.crop((x, y, x + size, y + size))


def _resize_shorter(img: Image.Image, size: int) -> Image.Image:
    if HAS_TV:
        return TV.Resize(size, interpolation=TV.InterpolationMode.BICUBIC)(img)
    w, h  = img.size
    ratio = size / min(w, h)
    return img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)


def _color_jitter(
    img:        Image.Image,
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
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )(img)
    from PIL import ImageEnhance
    ops = [
        (ImageEnhance.Brightness, brightness),
        (ImageEnhance.Contrast,   contrast),
        (ImageEnhance.Color,      saturation),
    ]
    random.shuffle(ops)
    for cls, strength in ops:
        img = cls(img).enhance(random.uniform(max(0.0, 1 - strength), 1 + strength))
    return img


def _gaussian_blur(
    img:       Image.Image,
    sigma_min: float,
    sigma_max: float,
    prob:      float,
) -> Image.Image:
    if random.random() > prob:
        return img
    sigma = random.uniform(sigma_min, sigma_max)
    if HAS_TV:
        ks = max(3, int(sigma * 4 + 1) | 1)
        return TF.gaussian_blur(img, kernel_size=ks, sigma=sigma)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def _to_tensor_normalized(
    img:  Image.Image,
    mean: tuple[float, float, float],
    std:  tuple[float, float, float],
) -> torch.Tensor:
    if HAS_TV:
        t = TF.to_tensor(img)
        return TF.normalize(t, mean=list(mean), std=list(std))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean)) / np.array(std)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _augment_one(
    jpeg_bytes: bytes,
    aug_cfg:    DINOAugConfig,
    crop_size:  int,
    scale:      tuple[float, float],
    blur_prob:  float,
    sol_prob:   float,
) -> torch.Tensor:
    """Apply DINOv2-style augmentation to a single JPEG."""
    try:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    except Exception:
        return torch.zeros(3, crop_size, crop_size)

    img = _random_resized_crop(img, crop_size, scale)
    if random.random() < aug_cfg.flip_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = _color_jitter(
        img, aug_cfg.brightness, aug_cfg.contrast,
        aug_cfg.saturation, aug_cfg.hue, aug_cfg.color_jitter_prob,
    )
    if random.random() < aug_cfg.grayscale_prob:
        img = img.convert("L").convert("RGB")
    img = _gaussian_blur(img, aug_cfg.blur_sigma_min, aug_cfg.blur_sigma_max, blur_prob)
    if sol_prob > 0 and random.random() < sol_prob and HAS_TV:
        img = TF.solarize(img, 128)
    return _to_tensor_normalized(img, aug_cfg.mean, aug_cfg.std)


# CPU pipeline implementations

class CPUAugPipeline:
    """DINOv2 multi-crop CPU augmentation pipeline."""

    def __init__(
        self,
        source:         Any,
        aug_cfg:        DINOAugConfig,
        batch_size:     int,
        resolution_src: Any,
        seed:           int = 0,
    ) -> None:
        self._source         = source
        self._aug_cfg        = aug_cfg
        self._batch_size     = batch_size
        self._resolution_src = resolution_src
        random.seed(seed)

    def run_one_batch(self) -> dict[str, torch.Tensor]:
        global_size_arr, local_size_arr = self._resolution_src()
        global_size = int(global_size_arr)
        local_size  = int(local_size_arr)

        jpeg_batch: list[np.ndarray] = self._source()
        assert len(jpeg_batch) == self._batch_size

        cfg    = self._aug_cfg
        output: dict[str, list[torch.Tensor]] = {
            f"view_{i}": [] for i in range(cfg.n_views)
        }

        for jpeg_arr in jpeg_batch:
            jpeg_bytes = bytes(jpeg_arr)
            view_idx   = 0
            for i in range(cfg.n_global_crops):
                blur_p = cfg.blur_prob_global1 if i == 0 else cfg.blur_prob_global2
                sol_p  = cfg.solarize_prob if i == 1 else 0.0
                t = _augment_one(
                    jpeg_bytes, cfg,
                    crop_size=global_size, scale=cfg.global_crops_scale,
                    blur_prob=blur_p, sol_prob=sol_p,
                )
                output[f"view_{view_idx}"].append(t)
                view_idx += 1
            for _ in range(cfg.n_local_crops):
                t = _augment_one(
                    jpeg_bytes, cfg,
                    crop_size=local_size, scale=cfg.local_crops_scale,
                    blur_prob=cfg.blur_prob_local, sol_prob=0.0,
                )
                output[f"view_{view_idx}"].append(t)
                view_idx += 1

        return {k: torch.stack(v) for k, v in output.items()}


class CPUEvalPipeline:
    """Eval-mode CPU pipeline: deterministic resize + centre-crop, single view."""

    def __init__(self, source: Any, aug_spec: EvalAugSpec, batch_size: int) -> None:
        self._source     = source
        self._aug_spec   = aug_spec
        self._batch_size = batch_size

    def run_one_batch(self) -> dict[str, torch.Tensor]:
        jpeg_batch: list[np.ndarray] = self._source()
        assert len(jpeg_batch) == self._batch_size

        spec        = self._aug_spec
        resize_size = int(spec.crop_size * 256 / 224)
        tensors: list[torch.Tensor] = []

        for jpeg_arr in jpeg_batch:
            try:
                img = Image.open(io.BytesIO(bytes(jpeg_arr))).convert("RGB")
                img = _resize_shorter(img, resize_size)
                img = _center_crop(img, spec.crop_size)
                t   = _to_tensor_normalized(img, spec.mean, spec.std)
            except Exception:
                t = torch.zeros(3, spec.crop_size, spec.crop_size)
            tensors.append(t)

        return {"view_0": torch.stack(tensors)}


class CPULeJEPAPipeline:
    """LeJEPA CPU pipeline: context + N target crops."""

    def __init__(self, source: Any, aug_spec: LeJEPAAugSpec, batch_size: int) -> None:
        self._source     = source
        self._aug_spec   = aug_spec
        self._batch_size = batch_size

    def run_one_batch(self) -> dict[str, torch.Tensor]:
        jpeg_batch: list[np.ndarray] = self._source()
        assert len(jpeg_batch) == self._batch_size

        spec   = self._aug_spec
        output: dict[str, list[torch.Tensor]] = {
            "context": [],
            **{f"target_{i}": [] for i in range(spec.n_target_views)},
        }

        for jpeg_arr in jpeg_batch:
            jpeg_bytes = bytes(jpeg_arr)
            try:
                img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))

            ctx = _random_resized_crop(img, spec.context_crop_size, spec.context_scale)
            ctx = _color_jitter(ctx, 0.8, 0.8, 0.8, 0.2, 0.8)
            if random.random() < 0.5:
                ctx = ctx.transpose(Image.FLIP_LEFT_RIGHT)
            output["context"].append(_to_tensor_normalized(ctx, spec.mean, spec.std))

            for i in range(spec.n_target_views):
                tgt = _random_resized_crop(img, spec.target_crop_size, spec.target_scale)
                output[f"target_{i}"].append(_to_tensor_normalized(tgt, spec.mean, spec.std))

        return {k: torch.stack(v) for k, v in output.items()}


class CPUUserAugPipeline:
    """CPU pipeline for UserAugSpec: decode → normalise → user aug_fn."""

    def __init__(self, source: Any, aug_spec: UserAugSpec, batch_size: int) -> None:
        self._source     = source
        self._aug_spec   = aug_spec
        self._batch_size = batch_size

    def run_one_batch(self) -> dict[str, torch.Tensor]:
        jpeg_batch: list[np.ndarray] = self._source()
        assert len(jpeg_batch) == self._batch_size

        spec    = self._aug_spec
        tensors: list[torch.Tensor] = []

        for jpeg_arr in jpeg_batch:
            try:
                img = Image.open(io.BytesIO(bytes(jpeg_arr))).convert("RGB")
                img = _resize_shorter(img, spec.decode_size)
                t   = _to_tensor_normalized(img, spec.mean, spec.std)
            except Exception:
                t = torch.zeros(3, spec.decode_size, spec.decode_size)
            tensors.append(t)

        decoded_batch = torch.stack(tensors)  # [B, C, H, W]
        return spec.aug_fn(decoded_batch)


class CPUPipelineIterator:
    """Wraps any CPU*Pipeline in the DALIGenericIterator API."""

    def __init__(self, pipeline: Any, output_map: list[str], batch_size: int) -> None:
        self._pipe       = pipeline
        self._output_map = output_map
        self._exhausted  = False

    def __iter__(self) -> "CPUPipelineIterator":
        return self

    def __next__(self) -> list[dict[str, torch.Tensor]]:
        if self._exhausted:
            raise StopIteration
        try:
            return [self._pipe.run_one_batch()]
        except StopIteration:
            self._exhausted = True
            raise

    def reset(self) -> None:
        self._exhausted = False


# Null H2D / FP8 stubs

class NullH2DStream:
    def __init__(self, device: torch.device, topo: Any) -> None:
        self._device = device

    @contextlib.contextmanager
    def transfer(
        self, cpu_batch: dict[str, list[torch.Tensor]]
    ) -> Iterator[dict[str, list[torch.Tensor]]]:
        yield cpu_batch

    def send(
        self, cpu_batch: dict[str, list[torch.Tensor]]
    ) -> dict[str, list[torch.Tensor]]:
        return cpu_batch

    def wait(self) -> None:
        pass


class NullFP8Formatter:
    def quantise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


# Distributed stubs

@dataclass
class StubClusterTopology:
    nvlink_gen:             int  = 0
    nvlink_lanes_per_gpu:   int  = 0
    gpus_per_nvlink_domain: int  = 1
    has_pcie:               bool = False
    has_infiniband:         bool = False
    has_sharp:              bool = False
    has_nvlink_sharp:       bool = False
    cuda_major:             int  = 0
    cuda_minor:             int  = 0

    @property
    def is_nvl72(self) -> bool:
        return False

    @property
    def label(self) -> str:
        return "CPU-stub"


@dataclass
class StubDistribEnv:
    rank:             int                        = 0
    world_size:       int                        = 1
    local_rank:       int                        = 0
    local_world_size: int                        = 1
    topology:         StubClusterTopology | None = None

    def __post_init__(self) -> None:
        if self.topology is None:
            self.topology = StubClusterTopology()


# CPUBackend — assembles all stages

class CPUBackend:
    """Concrete backend: pure-Python CPU path for tests and CI."""

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def supports_fp8(self) -> bool:
        return False

    @property
    def supports_gpu(self) -> bool:
        return False

    def build_shard_cache(
        self,
        job_id:          str   = "cpu_test",
        node_master:     bool  = True,
        max_gb:          float = 1.0,
        prefetch_window: int   = 4,
        timeout_s:       float = 30.0,
        warn_threshold:  float = 0.85,
        **kwargs: Any,
    ) -> InProcessShardCache:
        return InProcessShardCache(
            job_id          = job_id,
            node_master     = node_master,
            max_gb          = max_gb,
            prefetch_window = prefetch_window,
            timeout_s       = timeout_s,
            warn_threshold  = warn_threshold,
        )

    def build_pipeline(
        self,
        source:             Any,
        aug_spec:           AugmentationSpec,
        aug_cfg:            Any         = None,  # ignored; kept for signature compat
        batch_size:         int         = 1,
        num_threads:        int         = 1,
        device_id:          int         = 0,
        resolution_src:     Any         = None,
        hw_decoder_load:    float       = 0.90,
        cpu_queue:          int         = 8,
        gpu_queue:          int         = 6,
        seed:               int         = 42,
        specs:              Any         = None,
        fuse_normalization: bool        = False,
        dali_fp8_output:    bool        = False,
    ) -> Any:
        """Dispatch to the appropriate CPU pipeline based on aug_spec type."""
        if isinstance(aug_spec, DinoV2AugSpec):
            log.info("CPUBackend: DinoV2AugSpec pipeline (batch=%d)", batch_size)
            return CPUAugPipeline(
                source         = source,
                aug_cfg        = aug_spec.aug_cfg,
                batch_size     = batch_size,
                resolution_src = resolution_src,
                seed           = seed,
            )
        if isinstance(aug_spec, EvalAugSpec):
            log.info("CPUBackend: EvalAugSpec pipeline (crop=%d)", aug_spec.crop_size)
            return CPUEvalPipeline(
                source=source, aug_spec=aug_spec, batch_size=batch_size
            )
        if isinstance(aug_spec, LeJEPAAugSpec):
            log.info("CPUBackend: LeJEPAAugSpec pipeline")
            return CPULeJEPAPipeline(
                source=source, aug_spec=aug_spec, batch_size=batch_size
            )
        if isinstance(aug_spec, UserAugSpec):
            log.info(
                "CPUBackend: UserAugSpec pipeline (decode_size=%d)", aug_spec.decode_size
            )
            return CPUUserAugPipeline(
                source=source, aug_spec=aug_spec, batch_size=batch_size
            )
        msg = f"CPUBackend: unsupported aug_spec type {type(aug_spec).__name__}."
        raise TypeError(msg)

    def build_pipeline_iterator(
        self,
        pipeline:   Any,
        aug_spec:   Any,
        output_map: list[str],
        batch_size: int,
    ) -> CPUPipelineIterator:
        return CPUPipelineIterator(
            pipeline   = pipeline,
            output_map = output_map,
            batch_size = batch_size,
        )

    def build_h2d_stream(self, device: torch.device, topo: Any) -> NullH2DStream:
        return NullH2DStream(device=device, topo=topo)

    def build_fp8_formatter(self) -> NullFP8Formatter:
        return NullFP8Formatter()

    def init_distributed(
        self,
        rank:             int = 0,
        world_size:       int = 1,
        local_rank:       int = 0,
        local_world_size: int = 1,
        force_topology:   str | None = None,
    ) -> StubDistribEnv:
        return StubDistribEnv(
            rank             = rank,
            world_size       = world_size,
            local_rank       = local_rank,
            local_world_size = local_world_size,
        )
