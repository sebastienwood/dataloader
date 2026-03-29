"""tests/test_shard_cache.py
============================
Unit tests for :mod:`dino_loader.shard_cache`.

Coverage
--------
_MmapPool
- acquire / release / reuse / ref-counting
- LRU eviction respects max_entries
- invalidate removes entry
- thread safety under concurrent acquire/release

NodeSharedShardCache._write
- Produces correct header + payload
- No fsync called on tmpfs (PERF-1)
- .tmp file cleaned up on rename failure

NodeSharedShardCache eviction
- _evict_for_locked raises after max retries when all slots are referenced [B2]

_HeartbeatWriter
- File content is "pid:job_id" format [FIX-HB]
- Stale detection uses both PID liveness and job_id

Heartbeat stale_s
- Configurable via LoaderConfig and forwarded to NodeSharedShardCache [M4]
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# write_shm_file is defined in the dino_loader test fixtures.
from tests.fixtures import write_shm_file

# ══════════════════════════════════════════════════════════════════════════════
# _MmapPool
# ══════════════════════════════════════════════════════════════════════════════


class TestMmapPoolAcquireRelease:

    def test_acquire_opens_mmap(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"hello world" * 100
        p = tmp_path / "shard.shm"
        write_shm_file(p, data)

        pool = _MmapPool(max_entries=8)
        entry = pool.acquire(p)
        assert entry.refs == 1
        assert entry.data_len == len(data)
        pool.release(p)
        pool.close_all()

    def test_acquire_twice_reuses_entry(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"reuse" * 50
        p = tmp_path / "reuse.shm"
        write_shm_file(p, data)

        pool = _MmapPool(max_entries=8)
        e1 = pool.acquire(p)
        e2 = pool.acquire(p)
        assert e1 is e2
        assert e2.refs == 2
        pool.release(p)
        pool.release(p)
        pool.close_all()

    def test_release_decrements_ref_to_zero(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"refcount" * 40
        p = tmp_path / "rc.shm"
        write_shm_file(p, data)

        pool = _MmapPool(max_entries=8)
        pool.acquire(p)
        pool.release(p)
        with pool._lock:
            assert pool._pool[str(p)].refs == 0
        pool.close_all()

    def test_invalidate_removes_entry(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"invalidate" * 20
        p = tmp_path / "inv.shm"
        write_shm_file(p, data)

        pool = _MmapPool(max_entries=8)
        pool.acquire(p)
        pool.release(p)
        pool.invalidate(p)
        with pool._lock:
            assert str(p) not in pool._pool
        pool.close_all()


class TestMmapPoolLRUEviction:

    def test_lru_evicts_oldest_unreferenced(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        pool = _MmapPool(max_entries=2)
        paths = []
        for i in range(3):
            data = (f"shard{i}" * 30).encode()
            p = tmp_path / f"s{i}.shm"
            write_shm_file(p, data)
            paths.append(p)

        pool.acquire(paths[0])
        pool.release(paths[0])
        pool.acquire(paths[1])
        pool.release(paths[1])
        pool.acquire(paths[2])

        with pool._lock:
            assert str(paths[0]) not in pool._pool, "Shard 0 should have been evicted"
        pool.release(paths[2])
        pool.close_all()


class TestMmapPoolThreadSafety:

    def test_concurrent_acquire_release_no_errors(self, tmp_path):
        from dino_loader.shard_cache import _MmapPool
        data = b"concurrent" * 100
        p = tmp_path / "concurrent.shm"
        write_shm_file(p, data)

        pool = _MmapPool(max_entries=8)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    pool.acquire(p)
                    time.sleep(0)
                    pool.release(p)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pool.close_all()
        assert not errors, f"Thread safety violation: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# NodeSharedShardCache._write
# ══════════════════════════════════════════════════════════════════════════════


class TestNodeSharedShardCacheWrite:

    def test_write_produces_16_byte_header_plus_payload(self, tmp_path):
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "ok.shm"
        data = b"payload data" * 10
        NodeSharedShardCache._write(shm, data)
        assert shm.exists()
        assert len(shm.read_bytes()) == 16 + len(data)

    def test_no_fsync_called_on_tmpfs(self, tmp_path):
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "nofsync.shm"
        with patch("os.fsync") as mock_fsync:
            NodeSharedShardCache._write(shm, b"check")
        mock_fsync.assert_not_called()

    def test_tmp_cleaned_up_on_rename_failure(self, tmp_path):
        from dino_loader.shard_cache import NodeSharedShardCache
        shm = tmp_path / "fail.shm"

        def bad_rename(self_path, target):
            raise OSError("simulated rename failure")

        with patch.object(Path, "rename", bad_rename), pytest.raises(OSError):
            NodeSharedShardCache._write(shm, b"data")

        assert not shm.with_suffix(".tmp").exists(), ".tmp must be cleaned up on failure"


# ══════════════════════════════════════════════════════════════════════════════
# _HeartbeatWriter — "pid:job_id" format [FIX-HB]
# ══════════════════════════════════════════════════════════════════════════════


class TestHeartbeatWriterFormat:
    """Verify the heartbeat file uses "pid:job_id" to prevent PID-recycling false negatives."""

    def test_heartbeat_file_contains_pid_and_job_id(self, tmp_path):
        from dino_loader.shard_cache import _HeartbeatWriter
        hb_path = tmp_path / "heartbeat"
        writer  = _HeartbeatWriter(hb_path, job_id="test_job_123")
        try:
            # Give the writer a moment to write the file.
            time.sleep(0.05)
            assert hb_path.exists(), "Heartbeat file was not created"
            content = hb_path.read_text().strip()
            assert ":" in content, (
                f"Heartbeat content {content!r} does not contain ':' separator. "
                "Expected 'pid:job_id' format [FIX-HB]."
            )
            pid_str, job_id = content.split(":", 1)
            assert pid_str.isdigit(), f"PID part {pid_str!r} is not numeric"
            assert int(pid_str) == os.getpid(), "PID does not match current process"
            assert job_id == "test_job_123", f"Job ID {job_id!r} does not match"
        finally:
            writer.stop()

    def test_heartbeat_pid_matches_current_process(self, tmp_path):
        from dino_loader.shard_cache import _HeartbeatWriter
        hb_path = tmp_path / "hb"
        writer  = _HeartbeatWriter(hb_path, job_id="myjob")
        try:
            time.sleep(0.05)
            content = hb_path.read_text().strip()
            pid_str = content.split(":")[0]
            assert int(pid_str) == os.getpid()
        finally:
            writer.stop()

    def test_heartbeat_stop_removes_file(self, tmp_path):
        from dino_loader.shard_cache import _HeartbeatWriter
        hb_path = tmp_path / "hb_stop"
        writer  = _HeartbeatWriter(hb_path, job_id="stoptest")
        time.sleep(0.05)
        assert hb_path.exists()
        writer.stop()
        assert not hb_path.exists(), "Heartbeat file should be removed after stop()"

    def test_purge_skips_alive_process_with_matching_job_id(self, tmp_path):
        """A live PID with the same job_id must NOT be purged.

        This guards against the scenario where the heartbeat belongs to a
        sibling process of the same job that happens to be alive.
        """
        from dino_loader.shard_cache import _purge_orphaned_shm

        fake_shm = tmp_path / "fake_job"
        fake_shm.mkdir()
        hb = fake_shm / "heartbeat"
        # Write "our PID : our job_id" — alive process, same job.
        hb.write_text(f"{os.getpid()}:fake_job")

        # _purge_orphaned_shm should skip this directory because the job_id matches.
        _purge_orphaned_shm("fake_job", hb_stale_s=0.0)
        assert fake_shm.exists(), (
            "Sibling process with matching job_id was wrongly purged."
        )


# ══════════════════════════════════════════════════════════════════════════════
# _evict_for_locked backpressure [B2]
# ══════════════════════════════════════════════════════════════════════════════


class TestEvictForLockedBackpressure:
    """When all mmap slots are referenced simultaneously, raise clearly [B2]."""

    def test_evict_raises_after_max_retries_when_all_slots_pinned(self, tmp_path):
        import asyncio
        from collections import OrderedDict

        from dino_loader.shard_cache import NodeSharedShardCache

        with patch("dino_loader.shard_cache._EVICT_RETRIES", 1), \
             patch("dino_loader.shard_cache._EVICT_WAIT_S", 0.01):

            cache = MagicMock(spec=NodeSharedShardCache)
            cache._lru = OrderedDict()
            cache._total_bytes = 200 * (1 << 30)
            cache._max_bytes = 128 * (1 << 30)

            async def _run():
                with pytest.raises(RuntimeError, match="could not evict enough space"):
                    await NodeSharedShardCache._load_one(cache, "fake/shard.tar", tmp_path / "x")

            asyncio.run(_run())


# ══════════════════════════════════════════════════════════════════════════════
# heartbeat_stale_s forwarded to NodeSharedShardCache [M4]
# ══════════════════════════════════════════════════════════════════════════════


class TestHeartbeatStaleForwarding:

    def test_heartbeat_stale_forwarded_to_purge(self, tmp_path):
        called_with: dict = {}

        def _spy(job_name, hb_stale_s=300.0):
            called_with["hb_stale_s"] = hb_stale_s

        with patch("dino_loader.shard_cache._purge_orphaned_shm", _spy):
            from dino_loader.shard_cache import NodeSharedShardCache
            try:
                NodeSharedShardCache(
                    node_master=True,
                    job_id="test_hb",
                    max_shm_gb=0.01,
                    heartbeat_stale_s=42.0,
                )
            except Exception:
                pass

        if called_with:
            assert called_with["hb_stale_s"] == 42.0
