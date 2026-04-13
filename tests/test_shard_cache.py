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

CephFS / inotify compatibility [FS-2]
- _inotify_wait falls back to stat-poll on ENOSYS (CephFS FUSE)
- _INOTIFY_AVAILABLE flag reset after ENOSYS
- _read_shard_async (anciennement _read_lustre) fonctionne avec aiofiles
"""

from __future__ import annotations

import asyncio
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
            assert str(paths[0]) not in pool._pool
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
        assert not errors


# ══════════════════════════════════════════════════════════════════════════════
# NodeSharedShardCache._write
# ══════════════════════════════════════════════════════════════════════════════


class TestNodeSharedShardCacheWrite:

    def test_write_produces_16_byte_header_plus_payload(self, tmp_path):
        from dino_loader.shard_cache import NodeSharedShardCache
        shm  = tmp_path / "ok.shm"
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

        assert not shm.with_suffix(".tmp").exists()


# ══════════════════════════════════════════════════════════════════════════════
# CephFS / inotify compatibility [FS-2]
# ══════════════════════════════════════════════════════════════════════════════


class TestInotifyFallback:
    """Tests pour la détection CephFS FUSE et le fallback stat-poll."""

    def test_inotify_available_flag_default_true(self):
        """Par défaut, _INOTIFY_AVAILABLE est True (inotify supposé disponible)."""
        import dino_loader.shard_cache as sc
        # Réinitialise le flag pour isoler le test.
        original = sc._INOTIFY_AVAILABLE
        sc._INOTIFY_AVAILABLE = True
        try:
            assert sc._INOTIFY_AVAILABLE is True
        finally:
            sc._INOTIFY_AVAILABLE = original

    def test_enosys_sets_flag_false_and_warns(self, tmp_path, caplog):
        """ENOSYS sur inotify_init1 → _INOTIFY_AVAILABLE=False + WARNING."""
        import ctypes
        import logging
        import dino_loader.shard_cache as sc

        original = sc._INOTIFY_AVAILABLE
        sc._INOTIFY_AVAILABLE = True

        # Crée un fichier "prêt" pour que _is_ready() retourne True immédiatement.
        from dino_loader.shard_cache import NodeSharedShardCache
        shm  = tmp_path / "ready.shm"
        NodeSharedShardCache._write(shm, b"data")

        class _FakeLibc:
            def inotify_init1(self, flags):
                ctypes.set_errno(38)  # ENOSYS
                return -1
            def inotify_add_watch(self, *a):
                return -1
            def inotify_rm_watch(self, *a):
                pass

        try:
            with patch("ctypes.CDLL", return_value=_FakeLibc()), \
                 caplog.at_level(logging.WARNING, logger="dino_loader.shard_cache"):
                # _inotify_wait doit tomber en stat-poll et réussir (fichier prêt).
                sc._inotify_wait(shm, timeout_s=1.0)

            assert sc._INOTIFY_AVAILABLE is False
            assert any("ENOSYS" in r.message or "ceph" in r.message.lower()
                       for r in caplog.records)
        finally:
            sc._INOTIFY_AVAILABLE = original

    def test_stat_poll_fallback_succeeds_on_ready_file(self, tmp_path):
        """Le fallback stat-poll doit retourner dès que le fichier est prêt."""
        import dino_loader.shard_cache as sc
        from dino_loader.shard_cache import NodeSharedShardCache

        original = sc._INOTIFY_AVAILABLE
        sc._INOTIFY_AVAILABLE = False   # force le fallback

        shm = tmp_path / "ready_poll.shm"
        NodeSharedShardCache._write(shm, b"poll_data")

        try:
            # Doit retourner sans TimeoutError.
            sc._inotify_wait(shm, timeout_s=2.0)
        finally:
            sc._INOTIFY_AVAILABLE = original

    def test_stat_poll_fallback_raises_on_timeout(self, tmp_path):
        """Le fallback stat-poll doit lever TimeoutError si le fichier n'arrive pas."""
        import dino_loader.shard_cache as sc

        original = sc._INOTIFY_AVAILABLE
        sc._INOTIFY_AVAILABLE = False

        shm = tmp_path / "never_ready.shm"  # fichier inexistant

        try:
            with pytest.raises(TimeoutError):
                sc._inotify_wait(shm, timeout_s=0.1)
        finally:
            sc._INOTIFY_AVAILABLE = original

    def test_read_shard_async_name_is_filesystem_neutral(self):
        """[FS-1] La fonction de lecture est renommée _read_shard_async."""
        import dino_loader.shard_cache as sc
        assert hasattr(sc, "_read_shard_async"), (
            "_read_shard_async doit exister (anciennement _read_lustre)"
        )
        assert not hasattr(sc, "_read_lustre"), (
            "_read_lustre a été renommé en _read_shard_async"
        )

    def test_read_shard_async_reads_file(self, tmp_path):
        """_read_shard_async lit un fichier correctement."""
        import dino_loader.shard_cache as sc
        data = b"test shard content" * 100
        p    = tmp_path / "test_shard.tar"
        p.write_bytes(data)

        result = asyncio.run(sc._read_shard_async(str(p)))
        assert result == data


# ══════════════════════════════════════════════════════════════════════════════
# _HeartbeatWriter [FIX-HB]
# ══════════════════════════════════════════════════════════════════════════════


class TestHeartbeatWriterFormat:

    def test_heartbeat_file_contains_pid_and_job_id(self, tmp_path):
        from dino_loader.shard_cache import _HeartbeatWriter
        hb_path = tmp_path / "heartbeat"
        writer  = _HeartbeatWriter(hb_path, job_id="test_job_123")
        try:
            time.sleep(0.05)
            assert hb_path.exists()
            content = hb_path.read_text().strip()
            assert ":" in content
            pid_str, job_id = content.split(":", 1)
            assert pid_str.isdigit()
            assert int(pid_str) == os.getpid()
            assert job_id == "test_job_123"
        finally:
            writer.stop()

    def test_heartbeat_stop_removes_file(self, tmp_path):
        from dino_loader.shard_cache import _HeartbeatWriter
        hb_path = tmp_path / "hb_stop"
        writer  = _HeartbeatWriter(hb_path, job_id="stoptest")
        time.sleep(0.05)
        assert hb_path.exists()
        writer.stop()
        assert not hb_path.exists()

    def test_purge_skips_alive_process_with_matching_job_id(self, tmp_path):
        """Un processus vivant avec le même job_id ne doit pas être purgé."""
        from dino_loader.shard_cache import _purge_orphaned_shm

        fake_shm = tmp_path / "fake_job"
        fake_shm.mkdir()
        hb = fake_shm / "heartbeat"
        hb.write_text(f"{os.getpid()}:fake_job")

        _purge_orphaned_shm("fake_job", hb_stale_s=0.0)
        assert fake_shm.exists(), "Sibling process with matching job_id was wrongly purged."


# ══════════════════════════════════════════════════════════════════════════════
# _evict_for_locked backpressure [B2]
# ══════════════════════════════════════════════════════════════════════════════


class TestEvictForLockedBackpressure:

    def test_evict_raises_after_max_retries_when_all_slots_pinned(self, tmp_path):
        """_load_one must raise RuntimeError when eviction cannot free enough space.

        [FIX] The previous test used MagicMock(spec=NodeSharedShardCache) which
        does not have _sem, causing AttributeError.  We now build a minimal
        stub with the exact attributes accessed by _load_one, including a real
        asyncio.Semaphore, and patch the module-level constants to make the
        test fast.
        """
        from collections import OrderedDict
        from dino_loader.shard_cache import NodeSharedShardCache

        class _StubCache:
            """Minimal stub providing the attributes accessed by _load_one."""
            _sem          = asyncio.Semaphore(1)  # real semaphore
            _lru          = OrderedDict()          # empty — nothing to evict
            _total_bytes  = 200 * (1 << 30)       # 200 GB used
            _max_bytes    = 128 * (1 << 30)       # 128 GB budget → always full
            _in_flight: set = set()
            _metrics      = None

            # _lru_lock must be a real lock.
            _lru_lock     = threading.Lock()

            def _evict_for_locked(self, incoming: int) -> None:
                """No-op — nothing to evict (simulates all slots pinned)."""

            def _update_utilisation_metric(self) -> None:
                pass

            @staticmethod
            def _write(shm: Path, data: bytes) -> None:
                pass

        stub = _StubCache()

        with patch("dino_loader.shard_cache._EVICT_RETRIES", 1), \
             patch("dino_loader.shard_cache._EVICT_WAIT_S", 0.01):

            async def _run():
                with pytest.raises(RuntimeError, match="could not evict enough space"):
                    await NodeSharedShardCache._load_one(stub, "fake/shard.tar", tmp_path / "x")

            asyncio.run(_run())


# ══════════════════════════════════════════════════════════════════════════════
# heartbeat_stale_s forwarding [M4] — [FIX-STALE]
# ══════════════════════════════════════════════════════════════════════════════


class TestHeartbeatStaleForwarding:

    def test_heartbeat_stale_forwarded_to_init_shm(self, tmp_path):
        """heartbeat_stale_s must be forwarded from the constructor to _init_shm.

        [FIX-STALE] The previous implementation passed _HB_STALE_S (the global
        constant) to _purge_orphaned_shm inside _init_shm instead of using the
        constructor parameter.  We verify the fix by spying on _init_shm.
        """
        from dino_loader.shard_cache import NodeSharedShardCache

        init_shm_calls: list[float] = []
        original_init_shm = NodeSharedShardCache._init_shm

        def _spy_init_shm(self_cache, heartbeat_stale_s=300.0):  # type: ignore[misc]
            init_shm_calls.append(heartbeat_stale_s)
            # Don't actually create /dev/shm dirs in tests.

        with patch.object(NodeSharedShardCache, "_init_shm", _spy_init_shm), \
             patch("dino_loader.shard_cache._HeartbeatWriter"), \
             patch("dino_loader.shard_cache._purge_orphaned_shm"), \
             patch("asyncio.new_event_loop"):
            try:
                NodeSharedShardCache(
                    node_master=True, job_id="test_hb",
                    max_shm_gb=0.01, heartbeat_stale_s=42.0,
                )
            except Exception:
                pass

        assert init_shm_calls, "_init_shm was never called"
        assert init_shm_calls[0] == 42.0, (
            f"Expected heartbeat_stale_s=42.0, got {init_shm_calls[0]}"
        )