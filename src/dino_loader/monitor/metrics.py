"""
dino_loader.monitor.metrics
===========================
Shared-memory metrics publishing and collection for zero-impact monitoring.

Allows lock-free updates to counters and timers from multiple dataloader 
processes (up to 8 local ranks per node). The CLI monitor can read these 
counters without blocking the fast path.
"""

import ctypes
import logging
import os
from multiprocessing import shared_memory
from typing import Optional

log = logging.getLogger(__name__)

# We support up to 8 GPUs/ranks per node.
MAX_LOCAL_RANKS = 8

class MetricsStruct(ctypes.Structure):
    _fields_ = [
        # Shard Cache (Rank 0 only usually for Lustre)
        ("lustre_read_time_ms", ctypes.c_int64),
        ("lustre_bytes_read", ctypes.c_int64),
        
        # Shard Cache Wait
        ("shard_wait_time_ms", ctypes.c_int64),
        ("shard_cache_utilization_pct", ctypes.c_float),
        
        # Pipeline Data
        ("pipeline_yield_time_ms", ctypes.c_int64),
        ("mixing_source_queue_depth", ctypes.c_int64),
        
        # End-of-loader
        ("loader_batches_yielded", ctypes.c_int64),
        
        # Misc timing
        ("h2d_transfer_time_ms", ctypes.c_int64),
        
        # Network stalls (cumulative)
        ("network_stall_time_ms", ctypes.c_int64),
        
        # Multi-node stall (cumulative)
        ("multinode_stall_time_ms", ctypes.c_int64),
    ]

class RankMetricsArray(ctypes.Structure):
    _fields_ = [("ranks", MetricsStruct * MAX_LOCAL_RANKS)]


class MetricsRegistry:
    """
    Publisher/Subscriber side for metrics.
    Set `create=True` on the Node Master to initialize the block.
    Set `create=False` on other processes or the CLI monitor to attach to it.
    """
    def __init__(self, job_id: str = "dino", create: bool = False, local_rank: int = 0):
        self.name = f"dino_metrics_{job_id}"
        self.local_rank = min(local_rank, MAX_LOCAL_RANKS - 1)
        self.size = ctypes.sizeof(RankMetricsArray)
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.data: Optional[RankMetricsArray] = None
        
        try:
            if create:
                try:
                    existing = shared_memory.SharedMemory(name=self.name)
                    existing.unlink()
                    existing.close()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            else:
                self.shm = shared_memory.SharedMemory(name=self.name)
                
            self.data = RankMetricsArray.from_buffer(self.shm.buf)
            
            # Clear it if we just created it
            if create:
                ctypes.memset(ctypes.addressof(self.data), 0, self.size)
                
        except Exception as e:
            log.warning(f"Could not initialize shared memory metrics for {self.name}: {e}")

    @property
    def metrics(self) -> Optional[MetricsStruct]:
        if self.data is not None:
            return self.data.ranks[self.local_rank]
        return None
        
    def read_all_ranks(self) -> Optional[RankMetricsArray]:
        """Used by the monitoring CLI to get a coherent snapshot of all ranks."""
        return self.data

    def inc(self, field: str, value: int = 1):
        """Helper to increment an integer metric."""
        if self.data is not None:
            current = getattr(self.data.ranks[self.local_rank], field)
            setattr(self.data.ranks[self.local_rank], field, current + value)

    def set(self, field: str, value: float):
        """Helper to set a metric absolute value."""
        if self.data is not None:
            setattr(self.data.ranks[self.local_rank], field, value)

    def close(self):
        if self.shm is not None:
            self.shm.close()

    def unlink(self):
        if self.shm is not None:
            self.shm.unlink()

# Global un-initialized instance. Dataloader/Monitor will initialize it.
_REGISTRY: Optional[MetricsRegistry] = None

def init_registry(job_id: str, create: bool, local_rank: int):
    global _REGISTRY
    _REGISTRY = MetricsRegistry(job_id, create, local_rank)

def get_registry() -> Optional[MetricsRegistry]:
    return _REGISTRY
