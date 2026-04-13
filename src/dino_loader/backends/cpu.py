"""dino_loader.backends.cpu
========================
Backend CPU pur Python / PyTorch.

Utilisé pour les tests unitaires, la CI et l'itération rapide sans GPU.
Toute la logique DALI vit dans ``DALIBackend`` ; ce backend la remplace par
PIL + torchvision.

Parallélisme CPU [PERF-CPU]
----------------------------
``CPUAugPipeline.run_one_batch()`` parallélise le décodage et l'augmentation
JPEG avec un ``ThreadPoolExecutor`` interne.  Pour ``batch_size=512``, le gain
est ~4–8× selon le nombre de cœurs disponibles.  Le nombre de workers est
borné à ``min(batch_size, os.cpu_count() or 4, 16)``.

Cycle de vie du ThreadPoolExecutor [PERF-THREAD]
-------------------------------------------------
``CPUAugPipeline`` expose ``close()`` pour arrêter explicitement son executor.
``CPUBackend.build_pipeline`` crée et retourne le pipeline ; l'appelant
(``loader.py``) est responsable de ``close()`` lors du teardown.

Dispatching sur AugmentationSpec [CPU-AUG-1..4]
-------------------------------------------------
``CPUBackend.build_pipeline`` sélectionne la bonne implémentation selon le
type de spec via ``isinstance``.

Normalisation [NORM]
--------------------
Toutes les conversions passent par ``NormStats.to_dali_scale()`` ou
``NormStats.to_numpy()`` — aucune multiplication ad-hoc par 255.

Dtype [FIX-DTYPE]
-----------------
Tous les pipelines lisent ``pipeline_cfg.output_dtype`` et castent la sortie
vers le dtype torch correspondant.
"""

import contextlib
import io
import logging
import os
import random
import threading
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchvision.transforms as TV
import torchvision.transforms.functional as TF
from PIL import Image

from dino_loader.augmentation import (
    AugmentationSpec,
    DinoV2AugSpec,
    EvalAugSpec,
    LeJEPAAugSpec,
    UserAugSpec,
)
from dino_loader.config import DINOAugConfig, NormStats, PipelineConfig
from dino_loader.sources.resolution import ResolutionSource

log = logging.getLogger(__name__)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _resolve_torch_dtype(pipeline_cfg: PipelineConfig | None) -> torch.dtype:
    if pipeline_cfg is None:
        return torch.bfloat16
    return _DTYPE_MAP.get(pipeline_cfg.output_dtype, torch.bfloat16)


# ---------------------------------------------------------------------------
# Stage 1 — Cache de shards in-process
# ---------------------------------------------------------------------------


class InProcessShardCache:
    """Cache de shards LRU en mémoire de processus.

    Conçu pour les tests et la CI — pas une alternative à ``NodeSharedShardCache``
    en production.  Ne supporte pas /dev/shm, le prefetch async, ni le mmap pool.

    Args:
        max_gb: Budget maximum en Go.

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
        self._max_bytes = int(max_gb * (1 << 30))
        self._lru:   OrderedDict[str, bytes] = OrderedDict()
        self._total: int = 0
        self._lock   = threading.Lock()

    def prefetch(self, shard_path: str) -> None:
        """No-op — prefetching non supporté par le cache in-process."""

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
        yield memoryview(self.get(shard_path))

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


# ---------------------------------------------------------------------------
# Paramètres d'augmentation par vue — évite la répétition des arguments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ViewAugParams:
    """Paramètres d'augmentation pour une seule vue.

    Regroupe les paramètres qui varient entre les vues globales et locales
    pour éviter de les passer un par un à ``_augment_one``.
    """

    crop_size: int
    scale:     tuple[float, float]
    blur_prob: float
    sol_prob:  float


# ---------------------------------------------------------------------------
# Helpers d'augmentation
# ---------------------------------------------------------------------------


def _random_resized_crop(
    img:   Image.Image,
    size:  int,
    scale: tuple[float, float],
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> Image.Image:
    return TV.RandomResizedCrop(
        size          = size,
        scale         = scale,
        ratio         = ratio,
        interpolation = TV.InterpolationMode.BICUBIC,
    )(img)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    return TV.CenterCrop(size)(img)


def _resize_shorter(img: Image.Image, size: int) -> Image.Image:
    return TV.Resize(size, interpolation=TV.InterpolationMode.BICUBIC)(img)


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
    return TV.ColorJitter(
        brightness=brightness, contrast=contrast,
        saturation=saturation, hue=hue,
    )(img)


def _gaussian_blur(
    img:       Image.Image,
    sigma_min: float,
    sigma_max: float,
    prob:      float,
) -> Image.Image:
    if random.random() > prob:
        return img
    sigma = random.uniform(sigma_min, sigma_max)
    ks = max(3, int(sigma * 4 + 1) | 1)
    return TF.gaussian_blur(img, kernel_size=ks, sigma=sigma)


def _to_tensor_normalized(
    img:       Image.Image,
    stats:     NormStats,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convertit une image PIL en tenseur normalisé."""
    mean_arr, std_arr = stats.to_numpy()
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=mean_arr.tolist(), std=std_arr.tolist())
    return t.to(out_dtype)


def _augment_one(
    jpeg_bytes: bytes,
    aug_cfg:    DINOAugConfig,
    params:     _ViewAugParams,
    out_dtype:  torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Augmentation DINOv2 sur un seul JPEG — thread-safe (pas d'état partagé).

    Args:
        jpeg_bytes: Bytes JPEG bruts.
        aug_cfg:    Configuration d'augmentation (probabilités, sigmas, etc.).
        params:     Paramètres spécifiques à cette vue (crop_size, scale, proba).
        out_dtype:  Dtype de sortie.

    """
    try:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    except Exception:  # noqa: BLE001
        return torch.zeros(3, params.crop_size, params.crop_size, dtype=out_dtype)

    img = _random_resized_crop(img, params.crop_size, params.scale)
    if random.random() < aug_cfg.flip_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = _color_jitter(
        img, aug_cfg.brightness, aug_cfg.contrast,
        aug_cfg.saturation, aug_cfg.hue, aug_cfg.color_jitter_prob,
    )
    if random.random() < aug_cfg.grayscale_prob:
        img = img.convert("L").convert("RGB")
    img = _gaussian_blur(img, aug_cfg.blur_sigma_min, aug_cfg.blur_sigma_max, params.blur_prob)
    if params.sol_prob > 0 and random.random() < params.sol_prob:
        img = TF.solarize(img, 128)
    return _to_tensor_normalized(img, aug_cfg.norm_stats, out_dtype=out_dtype)


# ---------------------------------------------------------------------------
# Pipelines CPU
# ---------------------------------------------------------------------------


class CPUAugPipeline:
    """Pipeline d'augmentation multi-crop DINOv2 sur CPU.

    [PERF-CPU] Le décodage et l'augmentation de chaque JPEG sont parallélisés
    via un ``ThreadPoolExecutor`` interne.  Le GIL est relâché pendant les
    opérations PIL (I/O decode) — le gain réel dépend de la charge CPU.

    [PERF-THREAD] Appeler ``close()`` explicitement pour libérer les threads
    dès que le pipeline n'est plus utilisé.
    """

    _MAX_WORKERS: int = min(os.cpu_count() or 4, 16)

    def __init__(
        self,
        source:         Any,
        aug_cfg:        DINOAugConfig,
        batch_size:     int,
        resolution_src: ResolutionSource,
        seed:           int        = 0,
        out_dtype:      torch.dtype = torch.bfloat16,
    ) -> None:
        self._source         = source
        self._aug_cfg        = aug_cfg
        self._batch_size     = batch_size
        self._resolution_src = resolution_src
        self._out_dtype      = out_dtype
        random.seed(seed)
        self._executor = ThreadPoolExecutor(
            max_workers        = min(batch_size, self._MAX_WORKERS),
            thread_name_prefix = "cpu-aug",
        )
        self._closed = False

    def run_one_batch(self) -> dict[str, torch.Tensor]:
        """Produit un batch augmenté en parallèle."""
        if self._closed:
            msg = "CPUAugPipeline.run_one_batch() called after close()"
            raise RuntimeError(msg)

        global_size_arr, local_size_arr = self._resolution_src()
        global_size = int(global_size_arr)
        local_size  = int(local_size_arr)

        jpeg_batch: list[np.ndarray] = self._source()
        assert len(jpeg_batch) == self._batch_size

        cfg = self._aug_cfg

        # Pré-calculer les _ViewAugParams pour toutes les vues une seule fois.
        view_params: list[_ViewAugParams] = []
        for i in range(cfg.n_global_crops):
            blur_p = cfg.blur_prob_global1 if i == 0 else cfg.blur_prob_global2
            sol_p  = cfg.solarize_prob if i == 1 else 0.0
            view_params.append(_ViewAugParams(
                crop_size = global_size,
                scale     = cfg.global_crops_scale,
                blur_prob = blur_p,
                sol_prob  = sol_p,
            ))
        for _ in range(cfg.n_local_crops):
            view_params.append(_ViewAugParams(
                crop_size = local_size,
                scale     = cfg.local_crops_scale,
                blur_prob = cfg.blur_prob_local,
                sol_prob  = 0.0,
            ))

        out_dtype = self._out_dtype

        def _augment_sample(jpeg_arr: np.ndarray) -> list[torch.Tensor]:
            jpeg_bytes = bytes(jpeg_arr)
            return [
                _augment_one(jpeg_bytes, cfg, params, out_dtype)
                for params in view_params
            ]

        futures = {
            self._executor.submit(_augment_sample, jpeg): idx
            for idx, jpeg in enumerate(jpeg_batch)
        }

        results: list[list[torch.Tensor]] = [[] for _ in range(self._batch_size)]
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

        return {
            f"view_{view_idx}": torch.stack(
                [results[sample_idx][view_idx] for sample_idx in range(self._batch_size)],
            )
            for view_idx in range(cfg.n_views)
        }

    def close(self) -> None:
        """Arrête l'executor et libère les threads."""
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class CPUEvalPipeline:
    """Pipeline d'évaluation CPU : resize + centre-crop déterministe."""

    def __init__(
        self,
        source:     Any,
        aug_spec:   EvalAugSpec,
        batch_size: int,
        out_dtype:  torch.dtype = torch.bfloat16,
    ) -> None:
        self._source     = source
        self._aug_spec   = aug_spec
        self._batch_size = batch_size
        self._out_dtype  = out_dtype

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
                t   = _to_tensor_normalized(img, spec.norm_stats, out_dtype=self._out_dtype)
            except Exception:  # noqa: BLE001
                t = torch.zeros(3, spec.crop_size, spec.crop_size, dtype=self._out_dtype)
            tensors.append(t)

        return {"view_0": torch.stack(tensors)}

    def close(self) -> None:
        """No-op — pas d'executor à libérer."""


class CPULeJEPAPipeline:
    """Pipeline LeJEPA CPU : crop contexte + N crops cibles."""

    def __init__(
        self,
        source:     Any,
        aug_spec:   LeJEPAAugSpec,
        batch_size: int,
        out_dtype:  torch.dtype = torch.bfloat16,
    ) -> None:
        self._source     = source
        self._aug_spec   = aug_spec
        self._batch_size = batch_size
        self._out_dtype  = out_dtype

    def run_one_batch(self) -> dict[str, torch.Tensor]:
        jpeg_batch: list[np.ndarray] = self._source()
        assert len(jpeg_batch) == self._batch_size

        spec   = self._aug_spec
        output: dict[str, list[torch.Tensor]] = {
            "context": [],
            **{f"target_{i}": [] for i in range(spec.n_target_views)},
        }

        for jpeg_arr in jpeg_batch:
            try:
                img = Image.open(io.BytesIO(bytes(jpeg_arr))).convert("RGB")
            except Exception:  # noqa: BLE001
                img = Image.new("RGB", (224, 224))

            ctx = _random_resized_crop(img, spec.context_crop_size, spec.context_scale)
            ctx = _color_jitter(ctx, 0.8, 0.8, 0.8, 0.2, 0.8)
            if random.random() < 0.5:
                ctx = ctx.transpose(Image.FLIP_LEFT_RIGHT)
            output["context"].append(
                _to_tensor_normalized(ctx, spec.norm_stats, out_dtype=self._out_dtype),
            )
            for i in range(spec.n_target_views):
                tgt = _random_resized_crop(img, spec.target_crop_size, spec.target_scale)
                output[f"target_{i}"].append(
                    _to_tensor_normalized(tgt, spec.norm_stats, out_dtype=self._out_dtype),
                )

        return {k: torch.stack(v) for k, v in output.items()}

    def close(self) -> None:
        """No-op — pas d'executor à libérer."""


class CPUUserAugPipeline:
    """Pipeline CPU pour UserAugSpec : decode → normalise → aug_fn utilisateur."""

    def __init__(
        self,
        source:     Any,
        aug_spec:   UserAugSpec,
        batch_size: int,
        out_dtype:  torch.dtype = torch.bfloat16,
    ) -> None:
        self._source     = source
        self._aug_spec   = aug_spec
        self._batch_size = batch_size
        self._out_dtype  = out_dtype

    def run_one_batch(self) -> dict[str, torch.Tensor]:
        jpeg_batch: list[np.ndarray] = self._source()
        assert len(jpeg_batch) == self._batch_size

        spec    = self._aug_spec
        tensors: list[torch.Tensor] = []

        for jpeg_arr in jpeg_batch:
            try:
                img = Image.open(io.BytesIO(bytes(jpeg_arr))).convert("RGB")
                img = _resize_shorter(img, spec.decode_size)
                t   = _to_tensor_normalized(img, spec.norm_stats, out_dtype=self._out_dtype)
            except Exception:  # noqa: BLE001
                t = torch.zeros(3, spec.decode_size, spec.decode_size, dtype=self._out_dtype)
            tensors.append(t)

        return spec.aug_fn(torch.stack(tensors))

    def close(self) -> None:
        """No-op — pas d'executor à libérer."""


class CPUPipelineIterator:
    """Enveloppe un pipeline CPU dans l'API ``DALIGenericIterator``."""

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


# ---------------------------------------------------------------------------
# Stubs no-op H2D / FP8
# ---------------------------------------------------------------------------


class NullH2DStream:
    """Stream H2D no-op pour le backend CPU."""

    def __init__(self, device: torch.device, topo: Any) -> None:
        self._device = device

    @contextlib.contextmanager
    def transfer(
        self, cpu_batch: dict[str, list[torch.Tensor]],
    ) -> Iterator[dict[str, list[torch.Tensor]]]:
        yield cpu_batch

    def send(
        self, cpu_batch: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        return cpu_batch

    def wait(self) -> None:
        pass


class NullFP8Formatter:
    """Formatter FP8 identité pour le backend CPU."""

    def quantise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


# ---------------------------------------------------------------------------
# Stubs distributed
# ---------------------------------------------------------------------------


@dataclass
class StubClusterTopology:
    """Topologie de cluster minimale pour le backend CPU."""

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
    def is_grace_blackwell(self) -> bool:
        return False

    @property
    def label(self) -> str:
        return "CPU-stub"


@dataclass
class StubDistribEnv:
    """Environnement distribué minimal pour le backend CPU."""

    rank:             int                        = 0
    world_size:       int                        = 1
    local_rank:       int                        = 0
    local_world_size: int                        = 1
    topology:         StubClusterTopology | None = None

    def __post_init__(self) -> None:
        if self.topology is None:
            self.topology = StubClusterTopology()


# ---------------------------------------------------------------------------
# CPUBackend
# ---------------------------------------------------------------------------


class CPUBackend:
    """Backend concret : chemin CPU pur pour les tests et la CI."""

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
        job_id:           str   = "cpu_test",
        node_master:      bool  = True,
        max_gb:           float = 1.0,
        prefetch_window:  int   = 4,
        timeout_s:        float = 30.0,
        warn_threshold:   float = 0.85,
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
        source:       Any,
        aug_spec:     AugmentationSpec,
        pipeline_cfg: PipelineConfig,
        specs:        Any = None,
    ) -> Any:
        """Dispatche sur le type de spec et construit le pipeline CPU approprié.

        Le pipeline retourné expose ``close()`` — appeler explicitement
        lors du teardown pour libérer les threads.

        Args:
            source:       Source callable compatible MixingSource.
            aug_spec:     Spec d'augmentation.
            pipeline_cfg: Paramètres de construction.
            specs:        Ignoré par le backend CPU.

        Raises:
            TypeError: Si le type de spec n'est pas reconnu.

        """
        resolution_src = getattr(source, "_resolution_src", None)
        out_dtype      = _resolve_torch_dtype(pipeline_cfg)

        if isinstance(aug_spec, DinoV2AugSpec):
            log.info("CPUBackend: DinoV2AugSpec pipeline (dtype=%s)", out_dtype)
            return CPUAugPipeline(
                source         = source,
                aug_cfg        = aug_spec.aug_cfg,
                batch_size     = getattr(source, "_batch_size", 1),
                resolution_src = resolution_src,
                seed           = pipeline_cfg.seed,
                out_dtype      = out_dtype,
            )
        if isinstance(aug_spec, EvalAugSpec):
            log.info("CPUBackend: EvalAugSpec pipeline (crop=%d, dtype=%s)", aug_spec.crop_size, out_dtype)
            return CPUEvalPipeline(
                source     = source,
                aug_spec   = aug_spec,
                batch_size = getattr(source, "_batch_size", 1),
                out_dtype  = out_dtype,
            )
        if isinstance(aug_spec, LeJEPAAugSpec):
            log.info("CPUBackend: LeJEPAAugSpec pipeline (dtype=%s)", out_dtype)
            return CPULeJEPAPipeline(
                source     = source,
                aug_spec   = aug_spec,
                batch_size = getattr(source, "_batch_size", 1),
                out_dtype  = out_dtype,
            )
        if isinstance(aug_spec, UserAugSpec):
            log.info("CPUBackend: UserAugSpec pipeline (decode=%d, dtype=%s)", aug_spec.decode_size, out_dtype)
            return CPUUserAugPipeline(
                source     = source,
                aug_spec   = aug_spec,
                batch_size = getattr(source, "_batch_size", 1),
                out_dtype  = out_dtype,
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
