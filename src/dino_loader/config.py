"""dino_loader.config
==================
Toute la configuration du loader vit ici.  Pas de logique — des dataclasses
pures.  Sérialisées vers/depuis JSON pour le checkpointing (pas de pickle).

Règle stricte
-------------
Ce module n'importe rien de ``dino_loader``.  Il peut importer de
``dino_datasets`` uniquement pour le ré-export de ``DatasetSpec``.
``CheckpointState`` est un pur dataclass sans méthodes d'I/O — la logique
de sauvegarde/chargement avec SHA-256 vit dans ``checkpoint.py``.

Corrections intégrées
---------------------
[FIX-DTYPE]       PipelineConfig expose ``output_dtype`` et le propage
                  depuis LoaderConfig.
[FIX-CHECKSUM]    Supprimé de ce module — dans checkpoint.py.
[FIX-CONTEXTLIB]  Supprimé de ce module — dans checkpoint.py.
[FIX-STATEFUL]    stateful_dataloader default → False pour permettre
                  LoaderConfig() sans checkpoint_dir (utile en tests/CI).
"""

from dataclasses import asdict, dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# NormStats — statistiques de normalisation canoniques
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormStats:
    """Statistiques de normalisation par canal en [0, 1] float32.

    Tout le stockage interne utilise la convention [0, 1].  Les conversions
    vers d'autres échelles se font uniquement au point d'utilisation.

    Attributes:
        mean: Moyenne par canal en [0, 1].  Shape: (3,).
        std:  Écart-type par canal en [0, 1].  Doit être strictement positif.

    Example::

        stats = NormStats.imagenet()
        mean_dali, std_dali = stats.to_dali_scale()
        # → ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])

    """

    mean: tuple[float, float, float]
    std:  tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(s <= 0.0 for s in self.std):
            msg = f"NormStats.std must be strictly positive, got {self.std}."
            raise ValueError(msg)

    def to_dali_scale(self) -> tuple[list[float], list[float]]:
        """Retourne ``(mean, std)`` en échelle [0, 255] pour les pipelines DALI."""
        mean_arr = np.array(self.mean, dtype=np.float32) * 255.0
        std_arr  = np.array(self.std,  dtype=np.float32) * 255.0
        return mean_arr.tolist(), std_arr.tolist()

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Retourne ``(mean, std)`` comme tableaux float32 numpy en [0, 1]."""
        return (
            np.array(self.mean, dtype=np.float32),
            np.array(self.std,  dtype=np.float32),
        )

    @classmethod
    def imagenet(cls) -> "NormStats":
        """Statistiques de normalisation ImageNet standard."""
        return cls(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    @classmethod
    def from_config(
        cls,
        mean:     tuple[float, float, float] | None,
        std:      tuple[float, float, float] | None,
        fallback: "NormStats | None" = None,
    ) -> "NormStats":
        """Construit depuis des overrides optionnels par dataset.

        Args:
            mean:     Moyenne par canal, ou ``None`` pour utiliser le fallback.
            std:      Écart-type, ou ``None`` pour utiliser le fallback.
            fallback: Stats à utiliser quand aucune valeur n'est fournie.

        """
        base          = fallback if fallback is not None else cls.imagenet()
        resolved_mean = mean if mean is not None else base.mean
        resolved_std  = std  if std  is not None else base.std
        return cls(mean=resolved_mean, std=resolved_std)


# ---------------------------------------------------------------------------
# SharedExtractionPoolConfig
# ---------------------------------------------------------------------------


@dataclass
class SharedExtractionPoolConfig:
    """Configuration du pool d'extraction partagé entre les ShardIterators.

    Attributes:
        max_workers:           Nombre maximum de threads d'extraction.
        queue_depth_per_shard: Profondeur de la file de samples par shard.

    """

    max_workers:           int = 16
    queue_depth_per_shard: int = 256

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            msg = f"SharedExtractionPoolConfig.max_workers must be ≥ 1, got {self.max_workers}."
            raise ValueError(msg)
        if self.queue_depth_per_shard < 1:
            msg = (
                f"SharedExtractionPoolConfig.queue_depth_per_shard must be ≥ 1, "
                f"got {self.queue_depth_per_shard}."
            )
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

_DTYPE_TO_TORCH: dict[str, str] = {
    "bf16": "bfloat16",
    "fp32": "float32",
}

_DTYPE_TO_DALI: dict[str, str] = {
    "bf16": "FLOAT16",
    "fp32": "FLOAT",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Paramètres de construction du pipeline d'augmentation.

    Attributes:
        num_threads:        Threads CPU DALI pour les opérations pré-décodage.
        device_id:          Index GPU cible.
        hw_decoder_load:    Fraction du décodage JPEG envoyée au HW nvjpeg.
        cpu_queue:          Profondeur de la file de prefetch CPU DALI.
        gpu_queue:          Profondeur de la file GPU DALI.
        seed:               Seed RNG du pipeline (offset par rank appliqué).
        fuse_normalization: Fusionner mean/std par-dataset dans le kernel DALI.
        dali_fp8_output:    Fusionner le cast FP8 dans le graphe DALI.
        output_dtype:       Dtype de normalisation — ``"bf16"`` ou ``"fp32"``.

    """

    num_threads:        int   = 8
    device_id:          int   = 0
    hw_decoder_load:    float = 0.90
    cpu_queue:          int   = 16
    gpu_queue:          int   = 6
    seed:               int   = 0
    fuse_normalization: bool  = True
    dali_fp8_output:    bool  = False
    output_dtype:       str   = "bf16"

    @property
    def torch_dtype_str(self) -> str:
        """Retourne le nom du dtype torch (ex. ``"bfloat16"``)."""
        return _DTYPE_TO_TORCH.get(self.output_dtype, "bfloat16")

    @property
    def dali_dtype_str(self) -> str:
        """Retourne le nom du type DALI (ex. ``"FLOAT16"``)."""
        return _DTYPE_TO_DALI.get(self.output_dtype, "FLOAT16")

    @classmethod
    def from_loader_config(
        cls,
        cfg:       "LoaderConfig",
        device_id: int,
        rank:      int,
    ) -> "PipelineConfig":
        """Construit depuis un ``LoaderConfig`` et les valeurs runtime.

        Args:
            cfg:       Configuration niveau loader.
            device_id: Index GPU local.
            rank:      Rang global (utilisé pour le seed par-rank).

        """
        return cls(
            num_threads        = cfg.dali_num_threads,
            device_id          = device_id,
            hw_decoder_load    = cfg.hw_decoder_load,
            cpu_queue          = cfg.dali_cpu_queue,
            gpu_queue          = cfg.dali_gpu_queue,
            seed               = cfg.seed + rank,
            fuse_normalization = cfg.fuse_normalization,
            dali_fp8_output    = cfg.use_fp8_output and cfg.dali_fp8_output,
            output_dtype       = cfg.output_dtype,
        )


# ---------------------------------------------------------------------------
# DINOAugConfig
# ---------------------------------------------------------------------------


@dataclass
class DINOAugConfig:
    """Configuration d'augmentation multi-crop DINOv2/v3.

    Attributes:
        global_crop_size:   Résolution initiale du crop global en pixels.
        local_crop_size:    Résolution initiale du crop local en pixels.
        n_global_crops:     Nombre de grands crops par image.
        n_local_crops:      Nombre de petits crops par image.
        global_crops_scale: Plage de scale RandomResizedCrop pour les vues globales.
        local_crops_scale:  Plage de scale pour les vues locales.
        blur_prob_global1:  Probabilité de flou gaussien pour le 1er crop global.
        blur_prob_global2:  Probabilité pour le 2e crop global.
        blur_prob_local:    Probabilité pour les crops locaux.
        solarize_prob:      Probabilité de solarisation (2e crop global uniquement).
        color_jitter_prob:  Probabilité d'application du color jitter.
        grayscale_prob:     Probabilité de conversion en niveaux de gris.
        preserve_aspect_ratio: Resize le côté court puis centre-crop.
        resolution_schedule: Liste de ``(epoch, global_crop_size)`` pour la
            résolution progressive.  Appliquée automatiquement par ``set_epoch()``.
        max_global_crop_size: Plafond de pré-allocation nvjpeg.
        max_local_crop_size:  Plafond de pré-allocation nvjpeg.
        mean: Moyenne de normalisation par canal en [0, 1] (défauts ImageNet).
        std:  Écart-type par canal en [0, 1].

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

    blur_sigma_min:     float = 0.1
    blur_sigma_max:     float = 2.0
    brightness:         float = 0.8
    contrast:           float = 0.8
    saturation:         float = 0.8
    hue:                float = 0.2
    flip_prob:          float = 0.5

    preserve_aspect_ratio: bool = True

    resolution_schedule:  list[tuple[int, int]] = field(default_factory=list)
    max_global_crop_size: int  = 0
    max_local_crop_size:  int  = 0

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
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
        """Nombre total de crops (global + local)."""
        return self.n_global_crops + self.n_local_crops

    @property
    def norm_stats(self) -> NormStats:
        """Statistiques de normalisation globales sous forme de ``NormStats``."""
        return NormStats(mean=self.mean, std=self.std)

    def crop_size_at_epoch(self, epoch: int) -> int:
        """Retourne la taille de crop global dictée par le schedule de résolution.

        Args:
            epoch: Époque d'entraînement courante (index 0).

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
    """Tous les paramètres runtime de DINODataLoader.

    Attributes:
        node_shm_gb:              Budget /dev/shm par nœud en Go.
        shard_prefetch_window:    Max de téléchargements Lustre → /dev/shm simultanés.
        shard_timeout_s:          Max secondes qu'un rang non-master attend un shard.
        heartbeat_stale_s:        Secondes sans heartbeat avant détection d'orphelin.
        extraction_pool:          Config du pool d'extraction partagé.
        dali_cpu_queue:           Profondeur de la file CPU DALI (≥ 16).
        dali_gpu_queue:           Profondeur de la file GPU DALI.
        dali_num_threads:         Threads CPU DALI.
        hw_decoder_load:          Fraction du décodage JPEG via le HW nvjpeg.
        shuffle_buffer_size:      Profondeur du réservoir de shuffle par ShardIterator.
        use_fp8_output:           Quantiser la sortie en FP8 E4M3. Nécessite TE.
        dali_fp8_output:          Fusionner le cast FP8 dans le graphe DALI.
        fuse_normalization:       Fusionner la normalisation par-dataset dans DALI.
        output_dtype:             Dtype de normalisation (``"bf16"`` ou ``"fp32"``).
        stateful_dataloader:      Active state_dict() / load_state_dict().
                                  Défaut ``False`` — passer ``True`` pour les runs
                                  de production avec ``checkpoint_dir`` renseigné.
        checkpoint_dir:           Répertoire d'écriture des checkpoints JSON.
        checkpoint_every_steps:   Fréquence de checkpoint (rank 0 uniquement).
        force_topology:           Override de la détection topology.
        seed:                     Seed RNG de base.
        debug_log_keys:           Chemin vers le log d'audit par sample.
        stall_timeout_s:          Secondes avant levée sur absence de batch. 0 = désactivé.
        shm_warn_threshold:       Fraction d'utilisation /dev/shm déclenchant un warning.
        prometheus_port:          Si défini, démarre un serveur HTTP Prometheus.
        adaptive_prefetch:        Active le contrôleur PID de prefetch adaptatif.
        adaptive_prefetch_target_util: Cible d'utilisation /dev/shm (0, 1].

    """

    # I/O
    node_shm_gb:           float = 128.0
    shard_prefetch_window: int   = 64
    shard_timeout_s:       float = 300.0

    heartbeat_stale_s: float = 300.0

    extraction_pool: SharedExtractionPoolConfig = field(
        default_factory=SharedExtractionPoolConfig,
    )

    dali_cpu_queue:    int   = 16
    dali_gpu_queue:    int   = 6
    dali_num_threads:  int   = 8
    hw_decoder_load:   float = 0.90

    shuffle_buffer_size: int  = 512

    use_fp8_output:     bool = False
    dali_fp8_output:    bool = False
    fuse_normalization: bool = True
    output_dtype:       str  = "bf16"

    # [FIX-STATEFUL] Default False so LoaderConfig() works without checkpoint_dir.
    # Production runs must pass stateful_dataloader=True with a valid checkpoint_dir.
    stateful_dataloader:    bool = False
    checkpoint_dir:         str  = ""
    checkpoint_every_steps: int  = 500

    force_topology: str | None = None
    seed:           int        = 0

    debug_log_keys: str | None = None

    stall_timeout_s:    float      = 600.0
    shm_warn_threshold: float      = 0.90
    prometheus_port:    int | None = None

    adaptive_prefetch:             bool  = False
    adaptive_prefetch_target_util: float = 0.80

    def __post_init__(self) -> None:
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

        if self.output_dtype not in _DTYPE_TO_TORCH:
            msg = (
                f"LoaderConfig: output_dtype must be one of "
                f"{list(_DTYPE_TO_TORCH)}, got {self.output_dtype!r}."
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

        if self.stateful_dataloader and not self.checkpoint_dir:
            msg = (
                "LoaderConfig: stateful_dataloader=True requires a non-empty "
                "checkpoint_dir pointing to a shared filesystem (Lustre/NFS). "
                "Example: checkpoint_dir='/lustre/ckpts/my_job/dl'"
            )
            raise ValueError(msg)

    def to_dict(self) -> dict:
        """Sérialise en dict plat."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LoaderConfig":
        """Désérialise depuis un dict, en ignorant les clés inconnues."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# CheckpointState — pur dataclass, sans logique I/O
# ---------------------------------------------------------------------------


@dataclass
class CheckpointState:
    """État persisté du dataloader — sérialisable en JSON.

    Ce dataclass est volontairement sans méthodes d'I/O.  La logique de
    sauvegarde/chargement avec SHA-256 et écriture atomique vit dans
    ``DataLoaderCheckpointer`` (``checkpoint.py``).

    Attributes:
        step:             Étape globale courante.
        epoch:            Époque courante.
        dataset_names:    Noms ordonnés des datasets.
        mixing_weights:   Poids de mixage normalisés courants.
        global_crop_size: Taille de crop global active.
        local_crop_size:  Taille de crop local active.

    """

    step:             int
    epoch:            int
    dataset_names:    list[str]
    mixing_weights:   list[float]
    global_crop_size: int = 224
    local_crop_size:  int = 96

    def to_dict(self) -> dict:
        """Sérialise en dict plat."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointState":
        """Désérialise depuis un dict, en ignorant les clés inconnues."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})