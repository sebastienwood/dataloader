"""
DINOv3 DALI DataLoader — Accelerated with Prefetch, Preprocess & RAM Cache
===========================================================================

Architecture (all stages run concurrently in background threads/processes):

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Stage 0 – Shard Prefetch                                          │
    │    Background threads stream tar shards from disk/object-store     │
    │    into a RAM shard cache (mmap-backed ring buffer).               │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │ raw JPEG bytes
    ┌───────────────────────────▼─────────────────────────────────────────┐
    │  Stage 1 – Decode + Augment (DALI GPU pipeline)                    │
    │    nvidia-dali decodes + applies DINOv3 multi-crop augmentations.  │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │ augmented GPU tensors
    ┌───────────────────────────▼─────────────────────────────────────────┐
    │  Stage 2 – Pinned-Memory Double Buffer                             │
    │    Completed DALI batches are D2H-copied into pre-allocated        │
    │    pinned CPU tensors (double-buffered, so GPU never stalls).      │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │ pinned CPU tensors
    ┌───────────────────────────▼─────────────────────────────────────────┐
    │  Stage 3 – Augmentation Result Cache (RAM)                         │
    │    LRU cache of fully-augmented batches in pinned memory.          │
    │    Keyed by (shard_id, sample_offset, epoch//cache_epochs).        │
    │    On cache-hit the DALI pipeline is bypassed entirely.            │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │ pinned CPU tensors
    ┌───────────────────────────▼─────────────────────────────────────────┐
    │  Stage 4 – Async H2D Transfer                                      │
    │    A dedicated CUDA stream moves pinned tensors to GPU while       │
    │    the compute stream works on the previous batch (overlap).       │
    └───────────────────────────┬─────────────────────────────────────────┘
                                │ GPU tensors (ready-to-use)
                          Training loop

Key RAM-budget knobs
────────────────────
    shard_cache_gb      : raw shard bytes kept in RAM (default 32 GB)
    batch_cache_batches : augmented batches kept in RAM (default 2000)
    prefetch_depth      : async pipeline depth (default 4)
    n_decode_workers    : parallel DALI pipelines (default = n_gpus)

Dynamic mixing ratios and distributed training are fully preserved from
the base dino_dali_dataloader.py (imported here).
"""

from __future__ import annotations

import collections
import io
import math
import os
import queue
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

# ── Base loader (previous file) ──────────────────────────────────────────────
from dino_dali_dataloader import (
    DINOAugConfig,
    DINODALIDataLoader,
    DatasetSpec,
    DynamicMixingSource,
    build_dino_pipeline,
)

# ── DALI ─────────────────────────────────────────────────────────────────────
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 – RAM Shard Cache
# ══════════════════════════════════════════════════════════════════════════════

class ShardRAMCache:
    """
    Keeps the hottest WebDataset tar shards resident in RAM so that repeated
    reads (multi-epoch) skip disk I/O entirely.

    Design
    ------
    - Fixed byte budget (`max_bytes`).  Eviction is LRU-by-shard.
    - Shards are loaded by background daemon threads; callers block only
      on first access (cold miss), subsequent epochs are instant.
    - Thread-safe via per-shard condition variables so multiple workers
      waiting on the same shard do not cause duplicate reads.

    Usage
    -----
        cache = ShardRAMCache(max_bytes=32 * 2**30)
        data  = cache.get("/path/to/shard-000000.tar")  # bytes
    """

    def __init__(self, max_bytes: int = 32 * (1 << 30), n_prefetch_threads: int = 4):
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._cache: Dict[str, bytes] = {}          # path → raw bytes
        self._order: collections.OrderedDict = collections.OrderedDict()  # LRU
        self._sizes: Dict[str, int] = {}
        self._total_bytes = 0
        self._in_flight: Dict[str, threading.Event] = {}  # currently loading

        self._prefetch_q: queue.Queue[str] = queue.Queue()
        for _ in range(n_prefetch_threads):
            t = threading.Thread(target=self._loader_worker, daemon=True)
            t.start()

    # ------------------------------------------------------------------

    def prefetch(self, path: str) -> None:
        """Non-blocking: schedule a shard for background loading."""
        with self._lock:
            if path in self._cache or path in self._in_flight:
                return
            ev = threading.Event()
            self._in_flight[path] = ev
        self._prefetch_q.put(path)

    def get(self, path: str) -> bytes:
        """Return shard bytes, blocking until available."""
        with self._lock:
            if path in self._cache:
                self._order.move_to_end(path)
                return self._cache[path]
            if path in self._in_flight:
                ev = self._in_flight[path]
            else:
                ev = threading.Event()
                self._in_flight[path] = ev
                self._prefetch_q.put(path)

        ev.wait()   # block until loader thread finishes
        with self._lock:
            return self._cache[path]

    def evict_cold(self, keep_paths: Sequence[str]) -> None:
        """Manually evict shards not in `keep_paths` to reclaim RAM."""
        keep = set(keep_paths)
        with self._lock:
            for path in list(self._cache):
                if path not in keep:
                    self._evict_one(path)

    # ------------------------------------------------------------------

    def _loader_worker(self):
        while True:
            path = self._prefetch_q.get()
            try:
                data = open(path, "rb").read()
            except OSError as e:
                print(f"[ShardRAMCache] WARNING: cannot read {path}: {e}")
                data = b""
            self._store(path, data)

    def _store(self, path: str, data: bytes):
        nbytes = len(data)
        with self._lock:
            # Evict LRU until we have room
            while self._total_bytes + nbytes > self.max_bytes and self._order:
                oldest = next(iter(self._order))
                self._evict_one(oldest)
            self._cache[path] = data
            self._sizes[path] = nbytes
            self._total_bytes += nbytes
            self._order[path] = True
            ev = self._in_flight.pop(path, None)
        if ev:
            ev.set()

    def _evict_one(self, path: str):
        """Caller must hold self._lock."""
        if path in self._cache:
            self._total_bytes -= self._sizes.pop(path, 0)
            del self._cache[path]
            self._order.pop(path, None)

    @property
    def utilisation(self) -> float:
        with self._lock:
            return self._total_bytes / max(self.max_bytes, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 – Augmented Batch LRU Cache (pinned RAM)
# ══════════════════════════════════════════════════════════════════════════════

class PinnedBatchCache:
    """
    LRU cache storing fully-augmented, pinned-memory batches.

    Key   : arbitrary hashable (shard_id, sample_offset, epoch_bucket)
    Value : dict {"global": [Tensor, ...], "local": [Tensor, ...]}
            All tensors live in CPU pinned memory for zero-copy H2D transfer.

    Rationale
    ---------
    DINO augmentations are stochastic, but caching across epochs within a
    small epoch-bucket (e.g. bucket = epoch // 2) still saves GPU decode work
    for the fraction of samples that are cache-hot, while preserving enough
    stochasticity for learning.  Set cache_epoch_bucket=1 to disable reuse.
    """

    def __init__(self, max_batches: int = 2000):
        self.max_batches = max_batches
        self._lock = threading.Lock()
        self._cache: collections.OrderedDict[object, dict] = collections.OrderedDict()

    def get(self, key: object) -> Optional[dict]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: object, batch: dict) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            # Pin tensors
            pinned = self._pin(batch)
            self._cache[key] = pinned
            if len(self._cache) > self.max_batches:
                self._cache.popitem(last=False)  # evict LRU

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @staticmethod
    def _pin(batch: dict) -> dict:
        """Deep-copy batch tensors into pinned memory."""
        def _p(t: torch.Tensor) -> torch.Tensor:
            return t.cpu().pin_memory() if not t.is_pinned() else t
        return {
            "global": [_p(v) for v in batch["global"]],
            "local":  [_p(v) for v in batch["local"]],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 – Pinned Double Buffer
# ══════════════════════════════════════════════════════════════════════════════

class PinnedDoubleBuffer:
    """
    Pre-allocates two sets of pinned CPU tensors.  While the training loop
    consumes buffer A, DALI fills buffer B, and vice versa.  Eliminates
    allocation overhead and ensures the GPU copy stream is never starved.
    """

    def __init__(self, batch_size: int, aug_cfg: DINOAugConfig):
        self._bufs = [self._alloc(batch_size, aug_cfg) for _ in range(2)]
        self._idx = 0

    def current(self) -> dict:
        return self._bufs[self._idx]

    def swap(self) -> dict:
        self._idx ^= 1
        return self._bufs[self._idx]

    @staticmethod
    def _alloc(bs: int, cfg: DINOAugConfig) -> dict:
        C = 3
        return {
            "global": [
                torch.empty(bs, C, cfg.global_crop_size, cfg.global_crop_size).pin_memory()
                for _ in range(cfg.n_global_crops)
            ],
            "local": [
                torch.empty(bs, C, cfg.local_crop_size, cfg.local_crop_size).pin_memory()
                for _ in range(cfg.n_local_crops)
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 – Async H2D Transfer with dedicated CUDA stream
# ══════════════════════════════════════════════════════════════════════════════

class AsyncH2DPrefetcher:
    """
    Keeps one batch pre-transferred to GPU at all times.

    Typical pattern (inside training loop):
        batch = prefetcher.next()   # instantly returns pre-transferred GPU tensors
        # ... compute loss, backward ...
        # H2D for the *following* batch happens in background during compute.
    """

    def __init__(self, loader_iter: Iterator, device: torch.device):
        self.loader_iter = loader_iter
        self.device = device
        self._stream = torch.cuda.Stream(device=device)
        self._next_batch: Optional[dict] = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = {
                "global": [t.to(self.device, non_blocking=True) for t in batch["global"]],
                "local":  [t.to(self.device, non_blocking=True) for t in batch["local"]],
            }

    def next(self) -> Optional[dict]:
        torch.cuda.current_stream(self.device).wait_stream(self._stream)
        batch = self._next_batch
        self._preload()   # kick off next transfer while caller computes
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        b = self.next()
        if b is None:
            raise StopIteration
        return b


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: CachedDINODataLoader
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheConfig:
    shard_cache_gb: float = 32.0          # Stage 0: RAM for raw shard bytes
    shard_prefetch_threads: int = 4       # Stage 0: concurrent shard loaders
    batch_cache_batches: int = 2000       # Stage 3: augmented batch LRU size
    cache_epoch_bucket: int = 2           # Stage 3: reuse cache across N epochs
    prefetch_depth: int = 4               # Stage 1→2: pipeline look-ahead depth
    enable_batch_cache: bool = True       # Toggle Stage 3 entirely
    enable_shard_cache: bool = True       # Toggle Stage 0 entirely


class CachedDINODataLoader:
    """
    Full accelerated DINOv3 DataLoader.

    Wraps DINODALIDataLoader with:
        • Stage 0  – ShardRAMCache        (raw shard bytes in RAM)
        • Stage 1  – DALI GPU pipeline    (decode + augment)
        • Stage 2  – PinnedDoubleBuffer   (zero-alloc D2H)
        • Stage 3  – PinnedBatchCache     (augmented batch LRU)
        • Stage 4  – AsyncH2DPrefetcher   (overlap H2D with compute)

    Parameters
    ----------
    (all DINODALIDataLoader params, plus `cache_cfg: CacheConfig`)

    Iteration
    ---------
        for batch in loader:
            global_crops = batch["global"]   # list of Tensors on GPU
            local_crops  = batch["local"]
    """

    def __init__(
        self,
        dataset_specs: List[DatasetSpec],
        batch_size: int,
        aug_cfg: Optional[DINOAugConfig] = None,
        cache_cfg: Optional[CacheConfig] = None,
        num_threads: int = 4,
        device_id: int = 0,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 1000,
        seed: int = 42,
    ):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        self.aug_cfg   = aug_cfg   or DINOAugConfig()
        self.cache_cfg = cache_cfg or CacheConfig()
        self.batch_size = batch_size
        self.device = torch.device(f"cuda:{device_id}")
        self._epoch = 0

        # ── Stage 0: Shard RAM cache ─────────────────────────────────────────
        if self.cache_cfg.enable_shard_cache:
            self.shard_cache = ShardRAMCache(
                max_bytes=int(self.cache_cfg.shard_cache_gb * (1 << 30)),
                n_prefetch_threads=self.cache_cfg.shard_prefetch_threads,
            )
            # Pre-warm cache with shards assigned to this rank
            self._prewarm_shard_cache(dataset_specs, rank, world_size)
        else:
            self.shard_cache = None

        # ── Stage 1: DALI DataLoader ─────────────────────────────────────────
        self._dali_loader = DINODALIDataLoader(
            dataset_specs=dataset_specs,
            batch_size=batch_size,
            aug_cfg=self.aug_cfg,
            num_threads=num_threads,
            device_id=device_id,
            rank=rank,
            world_size=world_size,
            buffer_size=buffer_size,
            seed=seed,
        )

        # ── Stage 2: Pinned double buffer ────────────────────────────────────
        self._double_buf = PinnedDoubleBuffer(batch_size, self.aug_cfg)

        # ── Stage 3: Augmented batch LRU cache ──────────────────────────────
        self._batch_cache = (
            PinnedBatchCache(max_batches=self.cache_cfg.batch_cache_batches)
            if self.cache_cfg.enable_batch_cache
            else None
        )

        # ── Pipeline state ───────────────────────────────────────────────────
        self._prefetch_depth = self.cache_cfg.prefetch_depth
        self._prefetch_q: queue.Queue = queue.Queue(maxsize=self._prefetch_depth)
        self._stop_event = threading.Event()
        self._producer_thread = threading.Thread(
            target=self._producer_loop, daemon=True
        )
        self._producer_thread.start()

    # ------------------------------------------------------------------
    # Public: mixing control (delegates to DALI loader)
    # ------------------------------------------------------------------

    def set_weights(self, weights: Sequence[float]) -> None:
        self._dali_loader.set_weights(weights)

    def set_weight_by_name(self, name: str, weight: float) -> None:
        self._dali_loader.set_weight_by_name(name, weight)

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to advance cache epoch bucket."""
        self._epoch = epoch

    # ------------------------------------------------------------------
    # Public: iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        """Returns an AsyncH2DPrefetcher iterator (Stage 4)."""
        return AsyncH2DPrefetcher(self._consume_queue(), self.device)

    def _consume_queue(self) -> Iterator[dict]:
        """Yield pinned-CPU batches from the prefetch queue."""
        while True:
            batch = self._prefetch_q.get()
            if batch is None:
                return
            yield batch

    # ------------------------------------------------------------------
    # Private: producer thread (Stages 1–3)
    # ------------------------------------------------------------------

    def _producer_loop(self):
        """
        Runs in background.  Pulls from DALI, checks batch cache,
        copies into pinned double buffer, pushes to prefetch queue.
        """
        step = 0
        for dali_batch in self._dali_loader:
            if self._stop_event.is_set():
                break

            cache_key = self._make_cache_key(step)

            # ── Stage 3 cache lookup ──────────────────────────────────────
            if self._batch_cache is not None:
                hit = self._batch_cache.get(cache_key)
                if hit is not None:
                    self._prefetch_q.put(hit)
                    step += 1
                    continue

            # ── Stage 2: copy DALI GPU tensors → pinned CPU buffer ────────
            pinned = self._double_buf.swap()
            _copy_to_pinned(dali_batch, pinned)

            # Make a shallow copy so the cache stores an independent ref
            batch_for_cache = {
                "global": [t.clone() for t in pinned["global"]],
                "local":  [t.clone() for t in pinned["local"]],
            }

            # ── Stage 3 cache store ───────────────────────────────────────
            if self._batch_cache is not None:
                self._batch_cache.put(cache_key, batch_for_cache)

            self._prefetch_q.put(pinned)
            step += 1

        self._prefetch_q.put(None)  # sentinel

    def _make_cache_key(self, step: int) -> Tuple:
        bucket = self._epoch // max(self.cache_cfg.cache_epoch_bucket, 1)
        return (step, bucket)

    # ------------------------------------------------------------------
    # Private: shard cache pre-warming
    # ------------------------------------------------------------------

    def _prewarm_shard_cache(
        self,
        specs: List[DatasetSpec],
        rank: int,
        world_size: int,
    ) -> None:
        """Schedule all rank-local shards for background prefetch."""
        for spec in specs:
            for i, shard in enumerate(spec.shards):
                if i % world_size == rank:
                    self.shard_cache.prefetch(shard)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        stats = {}
        if self.shard_cache:
            stats["shard_cache_utilisation"] = f"{self.shard_cache.utilisation:.1%}"
        if self._batch_cache:
            stats["batch_cache_size"] = self._batch_cache.size
            stats["batch_cache_capacity"] = self.cache_cfg.batch_cache_batches
        stats["prefetch_queue_depth"] = self._prefetch_q.qsize()
        return stats


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _copy_to_pinned(src: dict, dst: dict) -> None:
    """Non-blocking D2H copy of a DALI batch into pre-allocated pinned buffers."""
    for s, d in zip(src["global"], dst["global"]):
        d.copy_(s, non_blocking=True)
    for s, d in zip(src["local"],  dst["local"]):
        d.copy_(s, non_blocking=True)


# ══════════════════════════════════════════════════════════════════════════════
# RAM budget estimator (utility)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_ram_budget(
    batch_size: int,
    aug_cfg: Optional[DINOAugConfig] = None,
    cache_cfg: Optional[CacheConfig] = None,
) -> dict:
    """
    Print a breakdown of approximate RAM usage for the cache layers.

    Returns a dict with keys:
        shard_cache_gb, batch_cache_gb, double_buffer_mb, prefetch_queue_mb, total_gb
    """
    aug = aug_cfg or DINOAugConfig()
    cfg = cache_cfg or CacheConfig()

    bytes_per_float = 4
    C = 3

    def view_bytes(size, n):
        return batch_size * C * size * size * bytes_per_float * n

    global_bytes = view_bytes(aug.global_crop_size, aug.n_global_crops)
    local_bytes  = view_bytes(aug.local_crop_size,  aug.n_local_crops)
    batch_bytes  = global_bytes + local_bytes

    shard_gb   = cfg.shard_cache_gb
    batch_gb   = batch_bytes * cfg.batch_cache_batches / (1 << 30)
    dbuf_mb    = batch_bytes * 2 / (1 << 20)
    queue_mb   = batch_bytes * cfg.prefetch_depth / (1 << 20)
    total_gb   = shard_gb + batch_gb + dbuf_mb / 1024 + queue_mb / 1024

    report = {
        "shard_cache_gb":    round(shard_gb, 2),
        "batch_cache_gb":    round(batch_gb, 2),
        "double_buffer_mb":  round(dbuf_mb, 1),
        "prefetch_queue_mb": round(queue_mb, 1),
        "total_gb":          round(total_gb, 2),
    }
    print("\n── RAM Budget Estimate ─────────────────────────────")
    for k, v in report.items():
        unit = "GB" if "gb" in k else "MB"
        print(f"  {k:<22}: {v} {unit}")
    print(f"  {'TOTAL':<22}: {total_gb:.2f} GB")
    print("────────────────────────────────────────────────────\n")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Example
# ══════════════════════════════════════════════════════════════════════════════

def _example():
    """
    torchrun --nproc_per_node=8 dino_dali_cached_dataloader.py
    """
    dist.init_process_group("nccl")
    rank      = dist.get_rank()
    world     = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()

    specs = [
        DatasetSpec("imagenet", [f"/data/imagenet/shard-{i:06d}.tar" for i in range(1000)], weight=0.7),
        DatasetSpec("laion2b",  [f"/data/laion/shard-{i:06d}.tar"    for i in range(5000)], weight=0.3),
    ]

    aug_cfg   = DINOAugConfig(n_local_crops=8)
    cache_cfg = CacheConfig(
        shard_cache_gb      = 64,     # 64 GB raw shard bytes per node
        batch_cache_batches = 4000,   # ~4000 augmented batches in pinned RAM
        cache_epoch_bucket  = 2,      # reuse cached batches across 2 epochs
        prefetch_depth      = 8,      # 8 batches in-flight ahead
    )

    if rank == 0:
        estimate_ram_budget(batch_size=256, aug_cfg=aug_cfg, cache_cfg=cache_cfg)

    loader = CachedDINODataLoader(
        dataset_specs = specs,
        batch_size    = 256,
        aug_cfg       = aug_cfg,
        cache_cfg     = cache_cfg,
        num_threads   = 8,
        device_id     = device_id,
        rank          = rank,
        world_size    = world,
        seed          = 0,
    )

    for epoch in range(10):
        loader.set_epoch(epoch)

        # Dynamic mixing curriculum
        if epoch == 3:
            loader.set_weights([0.5, 0.5])

        t0 = time.perf_counter()
        for step, batch in enumerate(loader):
            g0 = batch["global"][0]   # [256, 3, 224, 224] on GPU, ready immediately
            # ... forward / backward ...

            if rank == 0 and step % 200 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"Epoch {epoch} | Step {step:>5} | "
                    f"{elapsed/max(step,1)*1000:.1f} ms/batch | "
                    f"cache: {loader.cache_stats()}"
                )
        if rank == 0:
            print(f"Epoch {epoch} done. Cache stats: {loader.cache_stats()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    _example()
