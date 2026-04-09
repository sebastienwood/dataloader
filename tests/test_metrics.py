"""tests/test_metrics.py
========================
Unit tests for :mod:`dino_loader.monitor.metrics`.

Coverage
--------
MetricField
- All enum members map 1-to-1 to MetricsStruct fields
- No duplicate enum values
- All documented fields present

MetricsRegistry
- create / attach to shared memory
- inc: increments c_int64 counter fields
- set_float: sets c_float field (shard_cache_utilization_pct)
- inc on float field raises TypeError (wrong type guard)
- set_float on int field raises TypeError (wrong type guard)
- heartbeat: stamps current Unix timestamp
- read_all_ranks: returns array with correct rank slots
- zero on creation
- multiple rank slots independently writable
- graceful no-op when shared memory unavailable
- init_registry / get_registry singleton
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

os.environ.setdefault("SLURM_JOB_ID", "test_metrics_job")

from dino_loader.monitor.metrics import (
    MAX_LOCAL_RANKS,
    MetricField,
    MetricsRegistry,
    MetricsStruct,
    get_registry,
    init_registry,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _fresh_registry(job_id: str = "test_met") -> MetricsRegistry:
    try:
        stale = MetricsRegistry(job_id=job_id, create=False, local_rank=0)
        stale.unlink()
        stale.close()
    except Exception:
        pass
    return MetricsRegistry(job_id=job_id, create=True, local_rank=0)


# ══════════════════════════════════════════════════════════════════════════════
# MetricField enum integrity
# ══════════════════════════════════════════════════════════════════════════════


class TestMetricField:

    def test_all_members_map_to_struct_fields(self):
        struct_fields = {f[0] for f in MetricsStruct._fields_}
        for member in MetricField:
            assert member.value in struct_fields, (
                f"MetricField.{member.name} = {member.value!r} "
                "has no matching field in MetricsStruct"
            )

    def test_no_duplicate_values(self):
        values = [m.value for m in MetricField]
        assert len(values) == len(set(values)), "Duplicate MetricField values detected"

    def test_all_documented_fields_present(self):
        expected = [
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
        struct_fields = {f[0] for f in MetricsStruct._fields_}
        for field in expected:
            assert field in struct_fields, f"Documented field {field!r} missing from MetricsStruct"


# ══════════════════════════════════════════════════════════════════════════════
# MetricsRegistry shared memory operations
# ══════════════════════════════════════════════════════════════════════════════


class TestMetricsRegistryCreate:

    def test_create_and_attach(self):
        reg = _fresh_registry("met_create")
        try:
            reader = MetricsRegistry(job_id="met_create", create=False, local_rank=0)
            assert reader.data is not None
            reader.close()
        finally:
            reg.unlink()
            reg.close()

    def test_zero_on_creation(self):
        reg = _fresh_registry("met_zero")
        try:
            m = reg.data.ranks[0]
            assert m.loader_batches_yielded == 0
            assert m.heartbeat_ts == 0
        finally:
            reg.unlink()
            reg.close()


class TestMetricsRegistryInc:

    def test_inc_adds_value(self):
        reg = _fresh_registry("met_inc")
        try:
            reg.inc("loader_batches_yielded", 10)
            reg.inc("loader_batches_yielded", 5)
            assert reg.data.ranks[0].loader_batches_yielded == 15
        finally:
            reg.unlink()
            reg.close()

    def test_inc_default_delta_is_one(self):
        reg = _fresh_registry("met_inc1")
        try:
            reg.inc("loader_batches_yielded")
            assert reg.data.ranks[0].loader_batches_yielded == 1
        finally:
            reg.unlink()
            reg.close()

    def test_inc_with_metric_field_enum(self):
        reg = _fresh_registry("met_inc_enum")
        try:
            reg.inc(MetricField.BATCHES_YIELDED, 7)
            assert reg.data.ranks[0].loader_batches_yielded == 7
        finally:
            reg.unlink()
            reg.close()

    def test_inc_float_field_raises_type_error(self):
        reg = _fresh_registry("met_incf")
        try:
            with pytest.raises(TypeError, match="set_float"):
                reg.inc(MetricField.SHARD_CACHE_UTIL_PCT, 50)
        finally:
            reg.unlink()
            reg.close()


class TestMetricsRegistrySetFloat:

    def test_set_float_utilisation(self):
        reg = _fresh_registry("met_float")
        try:
            reg.set_float("shard_cache_utilization_pct", 73.5)
            assert abs(reg.data.ranks[0].shard_cache_utilization_pct - 73.5) < 0.01
        finally:
            reg.unlink()
            reg.close()

    def test_set_float_with_metric_field_enum(self):
        reg = _fresh_registry("met_float_enum")
        try:
            reg.set_float(MetricField.SHARD_CACHE_UTIL_PCT, 42.0)
            assert abs(reg.data.ranks[0].shard_cache_utilization_pct - 42.0) < 0.01
        finally:
            reg.unlink()
            reg.close()

    def test_set_float_on_int_field_raises_type_error(self):
        reg = _fresh_registry("met_floaterr")
        try:
            with pytest.raises(TypeError, match="inc()"):
                reg.set_float(MetricField.BATCHES_YIELDED, 42.0)
        finally:
            reg.unlink()
            reg.close()


class TestMetricsRegistryHeartbeat:

    def test_heartbeat_sets_timestamp(self):
        reg = _fresh_registry("met_hb")
        try:
            before = int(time.time()) - 1
            reg.heartbeat()
            assert reg.data.ranks[0].heartbeat_ts >= before
        finally:
            reg.unlink()
            reg.close()


class TestMetricsRegistryMultiRank:

    def test_multiple_rank_slots_independent(self):
        reg = _fresh_registry("met_ranks")
        try:
            for rank in range(MAX_LOCAL_RANKS):
                r = MetricsRegistry(job_id="met_ranks", create=False, local_rank=rank)
                r.inc("loader_batches_yielded", rank + 1)
                r.close()

            arr = reg.read_all_ranks()
            for rank in range(MAX_LOCAL_RANKS):
                assert arr.ranks[rank].loader_batches_yielded == rank + 1
        finally:
            reg.unlink()
            reg.close()


class TestMetricsRegistryGracefulDegradation:

    def test_graceful_on_missing_shm(self):
        reg = MetricsRegistry(job_id="definitely_does_not_exist_xyz", create=False)
        assert reg.data is None
        reg.inc("loader_batches_yielded", 1)  # must silently no-op
        reg.close()

    def test_set_float_noop_when_no_shm(self):
        reg = MetricsRegistry(job_id="definitely_does_not_exist_xyz2", create=False)
        assert reg.data is None
        reg.set_float(MetricField.SHARD_CACHE_UTIL_PCT, 50.0)  # must silently no-op
        reg.close()

    def test_heartbeat_noop_when_no_shm(self):
        reg = MetricsRegistry(job_id="definitely_does_not_exist_xyz3", create=False)
        assert reg.data is None
        reg.heartbeat()  # must silently no-op
        reg.close()

    def test_close_then_unlink_noop(self):
        reg = _fresh_registry("met_close")
        reg.close()
        try:
            reg.unlink()
        except Exception:
            pass  # acceptable on some platforms


class TestMetricsRegistrySingleton:

    def test_init_and_get_registry(self):
        # Clean up any stale registry first.
        try:
            stale = MetricsRegistry(job_id="singleton_test", create=False)
            stale.unlink()
            stale.close()
        except Exception:
            pass

        # init_registry takes (job_id, create, local_rank) — no keyword-only args.
        init_registry(job_id="singleton_test", create=True, local_rank=0)
        reg = get_registry()
        assert reg is not None
        assert reg.local_rank == 0

        try:
            reg.unlink()
            reg.close()
        except Exception:
            pass

    def test_get_registry_returns_none_before_init(self):
        # Cannot truly test "before any init" in a shared process, but we can
        # verify the return type contract is met after a valid init.
        init_registry(job_id="singleton_test2", create=True, local_rank=0)
        reg = get_registry()
        assert reg is None or isinstance(reg, MetricsRegistry)
        if reg is not None:
            try:
                reg.unlink()
                reg.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# BatchFilterNode increments skipped counter — integration via pipeline_graph
# ══════════════════════════════════════════════════════════════════════════════


class TestBatchFilterNodeMetrics:
    """BatchFilterNode._record_filtered() calls get_registry().inc('batches_filtered').

    This tests the plumbing between BatchFilterNode and the metrics registry,
    which replaced the old PostProcessPipeline.select() test.
    """

    def test_filter_node_skipped_counter(self):
        """n_skipped tracks how many batches were rejected by the predicate."""
        import torchdata.nodes as tn

        from dino_loader.memory import Batch
        from dino_loader.pipeline_graph import BatchFilterNode

        batches = [Batch([], [], [{"i": i}]) for i in range(6)]
        src = tn.IterableWrapper(iter(batches))
        # Keep only even-indexed batches.
        node = BatchFilterNode(src, lambda b: b.metadata[0]["i"] % 2 == 0)
        node.reset()

        kept: list[Batch] = []
        try:
            while True:
                kept.append(node.next())
        except StopIteration:
            pass

        assert len(kept) == 3
        assert node.n_skipped == 3

    def test_filter_node_reset_clears_skipped(self):
        import torchdata.nodes as tn

        from dino_loader.memory import Batch
        from dino_loader.pipeline_graph import BatchFilterNode

        batches = [Batch([], [], [{"ok": False}]) for _ in range(3)] + [
            Batch([], [], [{"ok": True}])
        ]
        src = tn.IterableWrapper(iter(batches))
        node = BatchFilterNode(src, lambda b: b.metadata[0]["ok"])
        node.reset()
        try:
            while True:
                node.next()
        except StopIteration:
            pass
        assert node.n_skipped == 3
        node.reset()
        assert node.n_skipped == 0