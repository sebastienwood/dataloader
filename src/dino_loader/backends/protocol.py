"""
dino_loader.backends.protocol
==============================
Abstract interface (Protocol) that every backend must satisfy.

Every method corresponds to one of the five pipeline stages:

Stage 1  — Shard I/O          → build_shard_cache()
Stage 2  — JPEG Extraction    → handled by MixingSource / ShardIterator
           (extraction is backend-agnostic; wds.TarIterator / legacy parser
           run on bytes regardless of how those bytes arrived)
Stage 3  — Augmentation       → build_pipeline()
Stage 4  — H2D Transfer       → build_h2d_stream()
Stage 5  — FP8 Quantisation   → build_fp8_formatter()

Additionally:
           init_distributed()  → bootstraps a DistribEnv without SLURM
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class BackendProtocol(Protocol):
    """
    Structural protocol for dino_loader pipeline backends.

    All methods return objects that conform to the interfaces expected by
    loader.py.  Backends are free to return thin wrappers, mock objects, or
    real hardware-backed instances — as long as the call signatures match.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``"cpu"`` or ``"dali"``."""
        ...

    @property
    def supports_fp8(self) -> bool:
        """True iff Transformer Engine FP8 output is available on this backend."""
        ...

    @property
    def supports_gpu(self) -> bool:
        """True iff GPU tensors are produced by this backend."""
        ...

    # ── Stage 1: Shard cache ──────────────────────────────────────────────────

    def build_shard_cache(
        self,
        job_id:             str,
        node_master:        bool,
        max_gb:             float,
        prefetch_window:    int,
        timeout_s:          float,
        warn_threshold:     float,
    ) -> Any:
        """
        Return a shard-cache object with the ``NodeSharedShardCache`` API:
        ``prefetch(path)``, ``get(path)``, ``get_view(path)``,
        ``utilisation`` property.
        """
        ...

    # ── Stage 3: Augmentation pipeline ───────────────────────────────────────

    def build_pipeline(
        self,
        source:           Any,   # MixingSource or compatible
        aug_cfg:          Any,   # DINOAugConfig
        batch_size:       int,
        num_threads:      int,
        device_id:        int,
        resolution_src:   Any,   # ResolutionSource
        hw_decoder_load:  float,
        cpu_queue:        int,
        gpu_queue:        int,
        seed:             int,
    ) -> Any:
        """
        Return an augmentation pipeline object that supports the iteration
        protocol used by loader.py:

        The pipeline is consumed via a ``DALIGenericIterator``-compatible
        wrapper.  For the CPU backend, the returned object is wrapped in
        ``CPUPipelineIterator``.
        """
        ...

    def build_pipeline_iterator(
        self,
        pipeline:   Any,
        output_map: List[str],
        batch_size: int,
    ) -> Any:
        """
        Wrap the pipeline in an iterator that yields dicts of
        ``{view_name: tensor}``, matching the ``DALIGenericIterator`` API
        used in loader.py.

        For DALI: returns a ``DALIGenericIterator``.
        For CPU:  returns a ``CPUPipelineIterator``.
        """
        ...

    # ── Stage 4: H2D transfer ─────────────────────────────────────────────────

    def build_h2d_stream(
        self,
        device: Any,       # torch.device
        topo:   Any,       # ClusterTopology
    ) -> Any:
        """
        Return an H2DStream-compatible object with ``transfer()`` context
        manager and ``send()`` / ``wait()`` methods.
        """
        ...

    # ── Stage 5: FP8 formatter ────────────────────────────────────────────────

    def build_fp8_formatter(self) -> Optional[Any]:
        """
        Return an FP8Formatter-compatible object (``quantise(tensor)`` method),
        or ``None`` if FP8 is unavailable / disabled for this backend.
        """
        ...

    # ── Distributed bootstrap ─────────────────────────────────────────────────

    def init_distributed(
        self,
        rank:             int = 0,
        world_size:       int = 1,
        local_rank:       int = 0,
        local_world_size: int = 1,
        force_topology:   Optional[str] = None,
    ) -> Any:
        """
        Return a ``DistribEnv``-compatible object.

        For DALI: delegates to ``slurm_init()`` or reads env vars.
        For CPU:  constructs a fake single-rank DistribEnv with a stub
                  ClusterTopology; does NOT call torch.distributed.init().
        """
        ...
