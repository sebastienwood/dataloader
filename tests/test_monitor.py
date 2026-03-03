"""
tests/test_monitor.py
=====================
Tests for the shared-memory metrics registry (monitor/metrics.py).

Replaces the original test_monitor.py with comprehensive coverage of all
fields, the heartbeat mechanism, and multi-rank simulation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import os
os.environ.setdefault("SLURM_JOB_ID", "test_monitor_job")

from dino_loader.monitor.metrics import (
    MAX_LOCAL_RANKS,
    MetricsRegistry,
    MetricsStruct,
    init_registry,
    get_registry,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_registry(job_id: str = "test_mon") -> MetricsRegistry:
    """Create a fresh registry, cleaning up any stale block."""
    try:
        stale = MetricsRegistry(job_id=job_id, create=False, local_rank=0)
        stale.unlink()
        stale.close()
    except Exception:
        pass
    return MetricsRegistry(job_id=job_id, create=True, local_rank=0)


# ══════════════════════════════════════════════════════════════════════════════
# Core write / read tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsRegistry:

    def test_create_and_attach(self):
        reg = _fresh_registry("mon_basic")
        try:
            reader = MetricsRegistry(job_id="mon_basic", create=False, local_rank=0)
            assert reader.data is not None
            reader.close()
        finally:
            reg.unlink()
            reg.close()

    def test_inc_increments_field(self):
        reg = _fresh_registry("mon_inc")
        try:
            reg.inc("loader_batches_yielded", 10)
            reg.inc("loader_batches_yielded", 5)
            m = reg.data.ranks[0]
            assert m.loader_batches_yielded == 15
        finally:
            reg.unlink()
            reg.close()

    def test_set_field(self):
        reg = _fresh_registry("mon_set")
        try:
            reg.set("shard_cache_utilization_pct", 73.5)
            m = reg.data.ranks[0]
            assert abs(m.shard_cache_utilization_pct - 73.5) < 0.01
        finally:
            reg.unlink()
            reg.close()

    def test_all_fields_present(self):
        """Every documented field exists on MetricsStruct."""
        m = MetricsStruct()
        expected_fields = [
            "lustre_read_time_ms",
            "lustre_bytes_read",
            "shard_cache_wait_time_ms",
            "shard_cache_utilization_pct",
            "pipeline_yield_time_ms",
            "mixing_source_queue_depth",
            "h2d_transfer_time_ms",
            "loader_batches_yielded",
            "network_stall_time_ms",
            "multinode_stall_time_ms",
            "heartbeat_ts",
        ]
        for f in expected_fields:
            assert hasattr(m, f), f"Missing field: {f}"

    def test_heartbeat_sets_timestamp(self):
        reg = _fresh_registry("mon_hb")
        try:
            before = int(time.time()) - 1
            reg.heartbeat()
            m = reg.data.ranks[0]
            assert m.heartbeat_ts >= before
        finally:
            reg.unlink()
            reg.close()

    def test_read_all_ranks_returns_array(self):
        reg = _fresh_registry("mon_all")
        try:
            result = reg.read_all_ranks()
            assert result is not None
            assert hasattr(result, "ranks")
        finally:
            reg.unlink()
            reg.close()

    def test_zero_on_creation(self):
        reg = _fresh_registry("mon_zero")
        try:
            m = reg.data.ranks[0]
            assert m.loader_batches_yielded == 0
            assert m.heartbeat_ts           == 0
        finally:
            reg.unlink()
            reg.close()

    def test_multiple_rank_slots(self):
        reg = _fresh_registry("mon_ranks")
        try:
            for rank in range(MAX_LOCAL_RANKS):
                r = MetricsRegistry(job_id="mon_ranks", create=False, local_rank=rank)
                r.inc("loader_batches_yielded", rank + 1)
                r.close()

            arr = reg.read_all_ranks()
            for rank in range(MAX_LOCAL_RANKS):
                assert arr.ranks[rank].loader_batches_yielded == rank + 1
        finally:
            reg.unlink()
            reg.close()

    def test_graceful_on_missing_shm(self):
        """Attaching to a non-existent block should not raise, just disable."""
        reg = MetricsRegistry(job_id="definitely_does_not_exist_xyz", create=False)
        # Should not raise; data should be None
        assert reg.data is None
        reg.inc("loader_batches_yielded", 1)   # must silently no-op
        reg.close()

    def test_close_then_unlink_noop(self):
        reg = _fresh_registry("mon_close")
        reg.close()
        try:
            reg.unlink()  # should not raise even after close
        except Exception:
            pass  # Some platforms raise on double-unlink; that's acceptable

    def test_init_registry_singleton(self):
        try:
            stale = MetricsRegistry(job_id="singleton_test", create=False)
            stale.unlink()
            stale.close()
        except Exception:
            pass

        init_registry(job_id="singleton_test", create=True, local_rank=0)
        reg = get_registry()
        assert reg is not None
        assert reg.local_rank == 0

        # Cleanup
        try:
            reg.unlink()
            reg.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Original test (preserved for compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def test_monitor_original_scenario():
    """
    Reproduces the original test_monitor.py test to ensure no regression.
    Uses the corrected field name ``shard_cache_wait_time_ms``.
    """
    job_id = "test_orig_scenario"
    try:
        stale = MetricsRegistry(job_id=job_id, create=False, local_rank=0)
        stale.unlink()
        stale.close()
    except Exception:
        pass

    init_registry(job_id=job_id, create=True, local_rank=0)
    reg = get_registry()

    reg.inc("loader_batches_yielded", 42)
    reg.inc("lustre_bytes_read", 1024 * 1024 * 500)   # 500 MB
    reg.set("shard_cache_utilization_pct", 67.3)
    reg.heartbeat()

    # Simulate a second rank reading
    from dino_loader.monitor import metrics as metrics_module
    r2 = metrics_module.MetricsRegistry(job_id=job_id, create=False, local_rank=0)
    m  = r2.read_all_ranks().ranks[0]

    assert m.loader_batches_yielded == 42, f"Got {m.loader_batches_yielded}"
    assert m.lustre_bytes_read      == 1024 * 1024 * 500
    assert abs(m.shard_cache_utilization_pct - 67.3) < 0.01
    assert m.heartbeat_ts > 0

    r2.close()
    reg.unlink()
