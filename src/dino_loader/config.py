"""dino_loader.config
==================
All loader-level configuration lives here.  No logic — pure dataclasses.
Serialised to / from JSON for checkpointing (no pickle fragility).

Changes
-------
[CFG-S1]  DatasetSpec.shard_sampling — explicit sampling mode.
[CFG-S2]  DatasetSpec.prob — alias for weight=.
[CFG-S3]  LoaderConfig.debug_log_keys — per-sample key logging.
[CFG-S4]  LoaderConfig.fuse_normalization — DALI-fused per-dataset norm.
[CFG-S5]  LoaderConfig.dali_fp8_output — in-graph FP8 cast.
[CFG-S6]  LoaderConfig.stall_timeout_s — configurable watchdog timeout.
[CFG-B4]  use_fp8_output validates TE presence at construction time.
[CFG-M4]  heartbeat_stale_s configurable (default 300 s).
[CFG-ARCH3] prometheus_port — opt-in Prometheus metrics endpoint.
[CFG-NORM] NormStats dataclass — canonical normalisation stats with
           to_dali_scale() helper, eliminating ad-hoc ×255 conversions.

Queue sizing note
-----------------
dali_cpu_queue defaults to 16 (previously 8).  AsyncPrefetchIterator has been
removed from loader.py because DALI's own prefetch queues already provide
equivalent double-buffering natively.  The larger cpu_queue value compensates
for the removed application-level buffer, keeping the GPU compute pipeline
fully saturated on NVL72 / B200 hardware where Lustre latency is variable.
A value of 16 corresponds to ~2 full batches in-flight per worker thread at
dali_num_threads=8.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# NormStats — canonical per-dataset normalisation statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormStats:
    """Per-channel normalisation statistics in [0, 1] float32 scale.

    All internal storage uses the [0, 1] convention (matching PyTorch
    torchvision defaults).  Conversions to other scales (e.g. DALI's [0, 255]
    convention) are performed explicitly via the helper methods below,
    eliminating ad-hoc ``× 255`` multiplications scattered across the codebase.

    Attributes:
        mean: Per-channel mean in [0, 1].  Shape: (3,).
        std: Per-channel std in [0, 1].  Positive values only.

    Example::

        stats = NormStats.imagenet()
        mean_dali, std_dali = stats.to_dali_scale()
        # → ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])

    """

    mean: tuple[float, float, float]
    std:  tuple[float, float, float]

    def __post_init__(self) -> None:
        """Validate that std values are strictly positive."""
        if any(s <= 0.0 for s in self.std):
            msg = f"NormStats.std must be strictly positive, got {self.std}."
            raise ValueError(msg)

    def to_dali_scale(self) -> tuple[list[float], list[float]]:
        """Return (mean, std) converted to [0, 255] scale for DALI pipelines.

        DALI's ``fn.normalize`` and ``fn.crop_mirror_normalize`` operators
        expect values in [0, 255] when the input is a decoded uint8 image.

        Returns:
            A pair ``(mean_255, std_255)`` where each element is a list of
            three floats in [0, 255].

        """
        mean_arr = np.array(self.mean, dtype=np.float32) * 255.0
        std_arr  = np.array(self.std,  dtype=np.float32) * 255.0
        return mean_arr.tolist(), std_arr.tolist()

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) as float32 numpy arrays in [0, 1] scale.

        Returns:
            A pair ``(mean_arr, std_arr)`` of shape ``(3,)`` float32 arrays.

        """
        return (
            np.array(self.mean, dtype=np.float32),
            np.array(self.std,  dtype=np.float32),
        )

    @classmethod
    def imagenet(cls) -> "NormStats":
        """Standard ImageNet normalisation statistics.

        Returns:
            ``NormStats`` with ImageNet mean/std (torchvision convention).

        """
        return cls(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    @classmethod
    def from_config(
        cls,
        mean: tuple[float, float, float] | None,
        std:  tuple[float, float, float] | None,
        fallback: "NormStats | None" = None,
    ) -> "NormStats":
        """Build a ``NormStats`` from optional per-dataset overrides.

        When both *mean* and *std* are ``None``, returns *fallback* (or
        ``NormStats.imagenet()`` when *fallback* is also ``None``).

        Args:
            mean: Per-channel mean or ``None`` to use the fallback.
            std: Per-channel std or ``None`` to use the fallback.
            fallback: Stats to use when no per-dataset values are provided.

        Returns:
            A ``NormStats`` instance with the resolved values.

        """
        base = fallback if fallback is not None else cls.imagenet()
        resolved_mean = mean if mean is not None else base.mean
        resolved_std  = std  if std  is not None else base.std
        return cls(mean=resolved_mean, std=resolved_std)


# ---------------------------------------------------------------------------
# DINOAugConfig
# ---------------------------------------------------------------------------


@dataclass
class DINOAugConfig:
    """DINOv2/v3 multi-crop augmentation configuration.

    Attributes:
        global_crop_size: Initial global crop resolution in pixels.
        local_crop_size: Initial local crop resolution in pixels.
        n_global_crops: Number of large crops per image.
        n_local_crops: Number of small crops per image.
        global_crops_scale: RandomResizedCrop scale range for global views.
        local_crops_scale: RandomResizedCrop scale range for local views.
        blur_prob_global1: Gaussian blur probability for first global crop.
        blur_prob_global2: Gaussian blur probability for second global crop.
        blur_prob_local: Gaussian blur probability for local crops.
        solarize_prob: Solarization probability (second global crop only).
        color_jitter_prob: Color jitter application probability.
        grayscale_prob: Grayscale conversion probability.
        preserve_aspect_ratio: Resize shorter side then centre-crop.
        resolution_schedule: List of (epoch, global_crop_size) for progressive
            resolution. Applied automatically by set_epoch().
        max_global_crop_size: nvjpeg pre-allocation ceiling. Defaults to
            global_crop_size.
        max_local_crop_size: nvjpeg pre-allocation ceiling. Defaults to
            local_crop_size.
        mean: Per-channel normalisation mean in [0, 1] (ImageNet defaults).
        std: Per-channel normalisation std in [0, 1] (ImageNet defaults).

    """

    global_crop_size:   int   = 224
    local_crop_size:    int   = 96
    n_global_crops:     int   = 2
    n_local_crops:      int   = 8
    global_crops_scale: tuple[float, float] = (0.32, 1.0)
    local_crops_scale:  tuple[float, float] = (0.05, 0.32)

    blur_prob_global1:  float = 1.0
    blur_prob_global2:  float = 0.1
    blur_prob_local:    float = 0.5
    solarize_prob:      float = 0.2
    color_jitter_prob:  float = 0.8
    grayscale_prob:     float = 0.2

    # Augmentation geometry
    blur_sigma_min:     float = 0.1
    blur_sigma_max:     float = 2.0
    brightness:         float = 0.8
    contrast:           float = 0.8
    saturation:         float = 0.8
    hue:                float = 0.2
    flip_prob:          float = 0.5

    preserve_aspect_ratio: bool = True

    resolution_schedule:  list[tuple[int, int]] = field(default_factory=list)
    max_global_crop_size: int  = 0   # 0 → set to global_crop_size in __post_init__
    max_local_crop_size:  int  = 0   # 0 → set to local_crop_size in __post_init__

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        """Validate and normalise fields after construction."""
        if self.max_global_crop_size == 0:
            self.max_global_crop_size = self.global_crop_size
        if self.max_local_crop_size == 0:
            self.max_local_crop_size = self.local_crop_size

        if self.resolution_schedule:
            self.resolution_schedule = sorted(self.resolution_schedule, key=lambda x: x[0])
            for epoch, _ in self.resolution_schedule:
                if epoch < 0:
                    msg = (
                        f"DINOAugConfig: resolution_schedule epochs must be ≥ 0, "
                        f"got epoch={epoch}."
                    )
                    raise ValueError(msg)

    @property
    def n_views(self) -> int:
        """Total number of crops (global + local)."""
        return self.n_global_crops + self.n_local_crops

    @property
    def norm_stats(self) -> NormStats:
        """Global normalisation statistics as a ``NormStats`` instance."""
        return NormStats(mean=self.mean, std=self.std)

    def crop_size_at_epoch(self, epoch: int) -> int:
        """Return the global crop size dictated by the resolution schedule.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            The global crop size in pixels for this epoch.

        """
        if not self.resolution_schedule:
            return self.global_crop_size
        size = self.global_crop_size
        for sched_epoch, sched_size in self.resolution_schedule:
            if epoch >= sched_epoch:
                size = sched_size
        return size


# ---------------------------------------------------------------------------
# LoaderConfig
# ---------------------------------------------------------------------------


@dataclass
class LoaderConfig:
    """All runtime knobs for DINODataLoader.

    Attributes:
        node_shm_gb: /dev/shm budget per node in GB.
        shard_prefetch_window: Max concurrent Lustre → /dev/shm downloads.
        shard_timeout_s: Max seconds a non-master rank waits for a shard.
        shard_extraction_workers: Thread-pool workers for tar → JPEG extraction.
        heartbeat_stale_s: Seconds without heartbeat before a /dev/shm dir is
            considered orphaned.
        dali_cpu_queue: DALI CPU-side prefetch queue depth.
            Default 16 (increased from 8 after AsyncPrefetchIterator removal —
            DALI's own queues now provide all pipeline buffering).
        dali_gpu_queue: DALI GPU-side prefetch queue depth.
        dali_num_threads: DALI CPU worker threads for pre-decode operations.
        hw_decoder_load: Fraction of JPEG decodes routed to nvjpeg HW ASIC.
        shuffle_buffer_size: In-memory sample reservoir depth per ShardIterator.
        use_fp8_output: Quantise output to FP8 E4M3. Requires transformer-engine.
        dali_fp8_output: Fuse FP8 cast into DALI graph. Requires use_fp8_output.
        fuse_normalization: Fuse per-dataset normalisation into the DALI kernel.
        output_dtype: Intermediate dtype before optional FP8 cast.
        stateful_dataloader: Enable state_dict() / load_state_dict() interface.
        checkpoint_dir: Where JSON checkpoint files are written.
        checkpoint_every_steps: Checkpoint frequency (rank 0 only).
        force_topology: Override topology detection: "pcie" | None.
        seed: Base random seed.
        debug_log_keys: Path to per-sample key audit log (disable in production;
            no size limit — use only for short one-shot debug sessions).
        stall_timeout_s: Seconds before raising on no batches. 0 = disabled.
        shm_warn_threshold: /dev/shm utilisation fraction that triggers a warning.
        prometheus_port: If set, start a prometheus_client HTTP server on this port.
        adaptive_prefetch: Enable adaptive prefetch window PID controller.
        adaptive_prefetch_target_util: Target /dev/shm utilisation (0, 1].

    """

    # I/O
    node_shm_gb:              float = 128.0
    shard_prefetch_window:    int   = 64
    shard_timeout_s:          float = 300.0
    shard_extraction_workers: int   = 4

    heartbeat_stale_s:        float = 300.0

    # DALI — cpu_queue raised to 16 after AsyncPrefetchIterator removal.
    dali_cpu_queue:           int   = 16
    dali_gpu_queue:           int   = 6
    dali_num_threads:         int   = 8
    hw_decoder_load:          float = 0.90

    # Data
    shuffle_buffer_size:      int   = 512

    # Output precision
    use_fp8_output:           bool  = False
    dali_fp8_output:          bool  = False
    fuse_normalization:       bool  = True
    output_dtype:             str   = "bf16"

    # Checkpointing
    stateful_dataloader:      bool  = True
    checkpoint_dir:           str   = "/tmp/dino_loader_ckpt"
    checkpoint_every_steps:   int   = 500

    # Cluster
    force_topology:           str | None = None
    seed:                     int   = 0

    # Debug — no file rotation; for short one-shot sessions only.
    debug_log_keys:           str | None = None

    # Watchdog
    stall_timeout_s:          float = 600.0

    # SHM monitoring
    shm_warn_threshold:       float = 0.90

    # Prometheus metrics endpoint (opt-in)
    prometheus_port:          int | None = None

    # Adaptive prefetch PID controller (opt-in)
    adaptive_prefetch:              bool  = False
    adaptive_prefetch_target_util:  float = 0.80

    def __post_init__(self) -> None:
        """Validate all fields at construction time."""
        if self.use_fp8_output:
            try:
                import transformer_engine.pytorch  # noqa: F401, PLC0415
            except ImportError:
                msg = (
                    "LoaderConfig: use_fp8_output=True requires transformer-engine. "
                    "Install with: pip install transformer-engine~=2.12\n"
                    "Or set use_fp8_output=False to use BF16 output."
                )
                raise ValueError(msg) from None

        if self.dali_fp8_output and not self.use_fp8_output:
            msg = "LoaderConfig: dali_fp8_output=True requires use_fp8_output=True."
            raise ValueError(msg)

        if self.output_dtype not in ("bf16", "fp32"):
            msg = (
                f"LoaderConfig: output_dtype must be 'bf16' or 'fp32', "
                f"got {self.output_dtype!r}."
            )
            raise ValueError(msg)

        if not (0.0 <= self.hw_decoder_load <= 1.0):
            msg = (
                f"LoaderConfig: hw_decoder_load must be in [0.0, 1.0], "
                f"got {self.hw_decoder_load}."
            )
            raise ValueError(msg)

        if self.stall_timeout_s < 0:
            msg = (
                f"LoaderConfig: stall_timeout_s must be ≥ 0 (0 = disabled), "
                f"got {self.stall_timeout_s}."
            )
            raise ValueError(msg)

        if not (0.0 <= self.shm_warn_threshold <= 1.0):
            msg = (
                f"LoaderConfig: shm_warn_threshold must be in [0.0, 1.0], "
                f"got {self.shm_warn_threshold}."
            )
            raise ValueError(msg)

        if self.heartbeat_stale_s <= 0:
            msg = (
                f"LoaderConfig: heartbeat_stale_s must be > 0, "
                f"got {self.heartbeat_stale_s}."
            )
            raise ValueError(msg)

        if self.adaptive_prefetch and not (0.0 < self.adaptive_prefetch_target_util <= 1.0):
            msg = (
                f"LoaderConfig: adaptive_prefetch_target_util must be in (0, 1], "
                f"got {self.adaptive_prefetch_target_util}."
            )
            raise ValueError(msg)

        if self.prometheus_port is not None:
            if not (1 <= self.prometheus_port <= 65535):
                msg = (
                    f"LoaderConfig: prometheus_port must be in [1, 65535], "
                    f"got {self.prometheus_port}."
                )
                raise ValueError(msg)
            try:
                import prometheus_client  # noqa: F401, PLC0415
            except ImportError:
                msg = (
                    f"LoaderConfig: prometheus_port={self.prometheus_port} requires "
                    "prometheus_client.  Install with: pip install prometheus-client"
                )
                raise ValueError(msg) from None

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LoaderConfig":
        """Deserialise from a plain dict, ignoring unknown keys.

        Args:
            d: Dictionary previously produced by ``to_dict()``.

        Returns:
            A ``LoaderConfig`` instance with the values from *d*.

        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# CheckpointState
# ---------------------------------------------------------------------------


@dataclass
class CheckpointState:
    """Persisted dataloader state — JSON-serialisable."""

    step:             int
    epoch:            int
    dataset_names:    list[str]
    mixing_weights:   list[float]
    global_crop_size: int = 224
    local_crop_size:  int = 96

    def save(self, path: Path) -> None:
        """Write atomically via a .tmp file with SHA-256 integrity envelope.

        Args:
            path: Destination file path (must be on a POSIX-rename-atomic filesystem).

        """
        import hashlib  # noqa: PLC0415
        payload      = asdict(self)
        payload_json = json.dumps(payload, indent=2)
        checksum     = hashlib.sha256(payload_json.encode()).hexdigest()
        envelope     = {"payload": payload, "sha256": checksum}
        tmp          = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(envelope, indent=2))
            tmp.rename(path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointState":
        """Deserialise from a plain dict, ignoring unknown keys.

        Args:
            d: Dictionary previously produced by ``to_dict()``.

        Returns:
            A ``CheckpointState`` instance.

        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        """Load and verify checkpoint, supporting both envelope and legacy formats.

        Args:
            path: Path to the JSON checkpoint file.

        Returns:
            A ``CheckpointState`` instance.

        Raises:
            ValueError: If the SHA-256 integrity check fails (envelope format only).

        """
        import hashlib  # noqa: PLC0415
        import logging  # noqa: PLC0415

        log = logging.getLogger(__name__)
        raw = json.loads(path.read_text())

        if "payload" in raw and "sha256" in raw:
            payload_json = json.dumps(raw["payload"], indent=2)
            expected     = hashlib.sha256(payload_json.encode()).hexdigest()
            if raw["sha256"] != expected:
                msg = (
                    f"Checkpoint {path} failed integrity check: "
                    f"stored sha256={raw['sha256']!r}, computed={expected!r}. "
                    "File may be corrupt or truncated."
                )
                raise ValueError(msg)
            data = raw["payload"]
        else:
            # Legacy flat format — no checksum.  Load with a warning.
            log.warning(
                "Checkpoint %s uses legacy flat format (no SHA-256 envelope). "
                "Resave with the current code to upgrade the format.",
                path,
            )
            data = raw

        data.setdefault("global_crop_size", 224)
        data.setdefault("local_crop_size",  96)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
