"""dino_loader.backends
====================
Pluggable backend abstraction for dino_loader.

A *backend* provides concrete implementations of the five pipeline stages so
that the rest of the codebase (loader.py, mixing_source.py, memory.py) can run
without any NVIDIA hardware, SLURM allocation, or DALI installation.

Backends
--------
``"dali"``  (default)
    Production path. Requires nvidia-dali, a CUDA GPU, and an NCCL-capable
    cluster.  Selected automatically when all dependencies are present.

``"cpu"``
    Pure-Python / PyTorch CPU path.  PIL + torchvision transforms replace DALI.
    No GPU, no SLURM, no shared memory.  Designed for unit tests, CI, and rapid
    iteration on laptops.

Public API
----------
::

    from dino_loader.backends import get_backend, BackendProtocol

    backend = get_backend("cpu")            # or "dali" / "auto"
    pipe    = backend.build_pipeline(...)
    h2d     = backend.build_h2d_stream(...)
    cache   = backend.build_shard_cache(...)
    env     = backend.init_distributed(rank=0, world_size=1)

"""

from __future__ import annotations

from typing import Literal

from dino_loader.backends.protocol import BackendProtocol

BackendName = Literal["dali", "cpu", "auto"]


def get_backend(name: BackendName = "auto") -> BackendProtocol:
    """Return a concrete backend instance.

    Parameters
    ----------
    name : "auto" | "dali" | "cpu"
        * ``"auto"``  — use DALI if available, fall back to CPU silently.
        * ``"dali"``  — always use the DALI backend; raise if unavailable.
        * ``"cpu"``   — always use the CPU backend.

    """
    if name == "cpu":
        from dino_loader.backends.cpu import CPUBackend
        return CPUBackend()

    if name == "dali":
        from dino_loader.backends.dali_backend import DALIBackend
        return DALIBackend()

    # "auto"
    try:
        import nvidia.dali  # noqa: F401
        import torch
        if torch.cuda.is_available():
            from dino_loader.backends.dali_backend import DALIBackend
            return DALIBackend()
    except ImportError:
        pass

    from dino_loader.backends.cpu import CPUBackend
    return CPUBackend()
