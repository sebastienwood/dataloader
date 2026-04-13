"""dino_loader.memory
==================
In-memory data structures and GPU transfer utilities.

[MEM-1] Batch dataclass with metadata and masks.
[MEM-4] FP8Formatter: no-op guard for dali_fp8_output=True path.

AsyncPrefetchIterator a été supprimé : les queues internes DALI
(prefetch_queue_depth) fournissent un double-buffering natif équivalent.

Note: Grace-Blackwell / NVL72 managed-memory paths have been removed.
      This loader targets B200, H200, H100 with standard PCIe topology.

allocate_buffers — correction [FIX-BUF]
----------------------------------------
La version précédente retournait une liste de deux tenseurs par type de crop
(double-buffer).  Après la suppression d'AsyncPrefetchIterator, ce double
n'avait plus d'utilisateur et consommait de la mémoire paginée inutilement.
allocate_buffers retourne désormais un seul tenseur paginé par type.

allocate_buffers — correction [FIX-PIN]
----------------------------------------
pin_memory() est conditionnel : si CUDA n'est pas disponible (machine CI
sans GPU ou driver trop ancien), on retourne un tenseur CPU non paginé.
Cela évite RuntimeError en tests et CI tout en gardant la performance
maximale en production GPU.

FP8Formatter.quantise — correction [FIX-FP8]
----------------------------------------------
La garde sur les tenseurs déjà FP8 est remplacée par un assert : ce cas ne
peut pas se produire légitimement (le loader contrôle ce qui entre dans
quantise).  Un assert échoue tôt et clairement plutôt que de logger un
warning silencieux qui serait ignoré en production.
"""

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from dino_env import ClusterTopology

from dino_loader.config import DINOAugConfig

log = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    log.debug("transformer-engine not installed — FP8 output disabled.")


@dataclass
class Batch:
    """One training batch with all views on GPU.

    Attributes:
        global_crops: List of global-crop tensors (BF16 or FP8).
        local_crops: List of local-crop tensors.
        metadata: Per-sample sidecar dicts. None when absent.
        masks: iBOT token mask tensor (bool, shape batch×n_tokens) or None.

    """

    global_crops: list
    local_crops:  list
    metadata:     list[dict | None] = field(default_factory=list)
    masks:        Any | None        = None

    def __iter__(self):
        """Convenience unpack as (global_crops, local_crops)."""
        return iter((self.global_crops, self.local_crops))


def _can_pin() -> bool:
    """True when CUDA is available and functional enough for pin_memory()."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def allocate_buffers(
    batch_size: int,
    aug_cfg:    DINOAugConfig,
    topo:       ClusterTopology,
    device:     torch.device,
    dtype:      torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Allocate pinned host buffers sized to max crop dimensions.

    [FIX-BUF] Retourne un seul tenseur paginé par type de crop (global/local)
    au lieu d'une liste de deux.  Le double-buffer de l'ancienne version était
    un vestige d'AsyncPrefetchIterator, supprimé car sans utilisateur.

    [FIX-PIN] pin_memory() is skipped when CUDA is unavailable (CI/test
    environments with no GPU or an incompatible driver) — returns a regular
    CPU tensor instead.  Production GPU runs get pinned memory as before.

    Uses max_global_crop_size / max_local_crop_size so no re-allocation
    occurs when set_resolution() is called mid-training.

    Args:
        batch_size: Per-GPU batch size.
        aug_cfg: Augmentation config providing max crop dimensions.
        topo: Cluster topology (accepted for API compat, not used).
        device: Target CUDA device.
        dtype: Tensor dtype (default bfloat16).

    Returns:
        Dict with 'global' and 'local' keys, each a single tensor
        (pinned if CUDA is available, plain CPU otherwise).

    """
    pin = _can_pin()

    def _buf(size: int) -> torch.Tensor:
        t = torch.zeros(batch_size, 3, size, size, dtype=dtype)
        return t.pin_memory() if pin else t

    return {
        "global": _buf(aug_cfg.max_global_crop_size),
        "local":  _buf(aug_cfg.max_local_crop_size),
    }


class H2DStream:
    """Async host-to-device transfer on a dedicated CUDA stream."""

    def __init__(self, device: torch.device, topo: ClusterTopology) -> None:
        """Initialise the H2D stream."""
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        log.info("H2DStream: PCIe path, dedicated CUDA stream on %s", device)

    @contextlib.contextmanager
    def transfer(
        self, cpu_batch: dict[str, list[torch.Tensor]],
    ) -> Iterator[dict[str, list[torch.Tensor]]]:
        """Async H2D transfer context manager."""
        with torch.cuda.stream(self._stream):
            gpu_batch = {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }
        torch.cuda.current_stream().wait_stream(self._stream)
        yield gpu_batch

    def send(
        self, cpu_batch: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        """Non-context-manager variant; caller must call wait() before use."""
        with torch.cuda.stream(self._stream):
            return {
                key: [t.to(self._device, non_blocking=True) for t in tensors]
                for key, tensors in cpu_batch.items()
            }

    def wait(self) -> None:
        """Synchronise the dedicated stream with the current CUDA stream."""
        torch.cuda.current_stream().wait_stream(self._stream)


class FP8Formatter:
    """Quantise BF16 tensors to FP8 E4M3 using Transformer Engine.

    Uses a rolling amax window (length 16) matching TE's internal convention
    so that FP8TensorMeta objects can be passed directly into te.fp8_autocast().

    When LoaderConfig.dali_fp8_output=True, loader.py does NOT construct this
    class (self._fp8 = None), so quantise() is never called from the hot path.
    Falls back to identity when TE is not installed.
    """

    _AMAX_WINDOW = 16

    def __init__(self) -> None:
        """Initialise FP8Formatter."""
        if HAS_TE:
            self._meta = te.fp8.FP8TensorMeta()
            self._meta.scale     = torch.ones(1)
            self._meta.scale_inv = torch.ones(1)
            self._meta.amax_history = torch.zeros(self._AMAX_WINDOW, 1)
            log.info("FP8Formatter: Transformer Engine path active")
        else:
            self._meta = None
            log.info("FP8Formatter: TE not installed — identity (no-op)")

    def quantise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantise *tensor* to FP8 E4M3, or return it unchanged if TE absent.

        [FIX-FP8] Le cas tensor.dtype already FP8 est remplacé par un assert.
        Ce cas ne peut pas se produire légitimement : le loader contrôle ce
        qui entre dans quantise() et ne l'appelle jamais sur un tenseur déjà
        quantifié.  Un assert échoue tôt et clairement.

        Args:
            tensor: BF16 (or FP32) tensor to quantise.

        Returns:
            FP8 E4M3 tensor, or *tensor* unchanged if TE is not installed.

        """
        assert tensor.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2), (
            f"FP8Formatter.quantise called on already-FP8 tensor (dtype={tensor.dtype}). "
            "This indicates a bug in the loader — check dali_fp8_output config."
        )
        if not HAS_TE or self._meta is None:
            return tensor
        return te.fp8.cast_to_fp8(tensor, self._meta, 0, te.fp8.Float8Tensor)