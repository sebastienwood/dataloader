"""
dino_loader.backends.protocol
==============================
Abstract interface (Protocol) that every backend must satisfy.

Each backend implements the same call signatures. Backend-specific concerns
(e.g. NormSource for DALI, no-op stubs for CPU) are fully encapsulated inside
the backend. loader.py is backend-agnostic.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BackendProtocol(Protocol):
    """Structural protocol for dino_loader pipeline backends."""

    @property
    def name(self) -> str: ...

    @property
    def supports_fp8(self) -> bool: ...

    @property
    def supports_gpu(self) -> bool: ...

    def build_shard_cache(
        self,
        job_id:          str,
        node_master:     bool,
        max_gb:          float,
        prefetch_window: int,
        timeout_s:       float,
        warn_threshold:  float,
        **kwargs: Any,
    ) -> Any: ...

    def build_pipeline(
        self,
        source:             Any,
        aug_spec:           Any,
        aug_cfg:            Any         = None,
        batch_size:         int         = 1,
        num_threads:        int         = 8,
        device_id:          int         = 0,
        resolution_src:     Any         = None,
        hw_decoder_load:    float       = 0.90,
        cpu_queue:          int         = 8,
        gpu_queue:          int         = 6,
        seed:               int         = 42,
        specs:              list | None = None,
        fuse_normalization: bool        = False,
        dali_fp8_output:    bool        = False,
    ) -> Any: ...

    def build_pipeline_iterator(
        self,
        pipeline:   Any,
        aug_spec:   Any,
        output_map: list[str],
        batch_size: int,
    ) -> Any: ...

    def build_h2d_stream(self, device: Any, topo: Any) -> Any: ...

    def build_fp8_formatter(self) -> Any | None: ...

    def init_distributed(
        self,
        rank:             int = 0,
        world_size:       int = 1,
        local_rank:       int = 0,
        local_world_size: int = 1,
        force_topology:   str | None = None,
    ) -> Any: ...
