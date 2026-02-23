"""
dino_loader.io.shard_cache
==========================
Node-local shared-memory shard cache.

Design
------
- One backing file per shard per node in /dev/shm/<job_id>/.
- Local rank 0 ("node master") reads from Lustre/NVMe and writes.
- Other local ranks wait via inotify (Linux) or stat-poll fallback —
  NOT a busy-spin, so 71 waiting ranks on NVL72 cost near-zero CPU.
- LRU eviction at file granularity.
- Header encodes (data_len, ready_flag) so readers can detect partial writes.
- asyncio + aiofiles for concurrent shard prefetch (hides Lustre latency).
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import logging
import mmap
import os
import select
import signal
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence

log = logging.getLogger(__name__)

# Header: [data_len: u64, ready: u64]  — two 8-byte fields, 16 bytes total
_HDR_FMT  = "QQ"
_HDR_SIZE = struct.calcsize(_HDR_FMT)   # 16

_READY_MAGIC = 0xDEAD_BEEF_CAFE_F00D    # sentinel distinguishing ready from zero


# ══════════════════════════════════════════════════════════════════════════════
# inotify helper (Linux-only; degrades gracefully)
# ══════════════════════════════════════════════════════════════════════════════

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO    = 0x00000080

try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _HAS_INOTIFY = True
except Exception:
    _HAS_INOTIFY = False


def _inotify_wait(path: Path, timeout_s: float = 120.0) -> None:
    """
    Block until `path` appears and is ready, using inotify on its parent dir.
    Falls back to exponential-backoff stat() when inotify is unavailable.
    """
    parent = str(path.parent)
    name   = path.name.encode()

    if _HAS_INOTIFY:
        fd = _libc.inotify_init1(os.O_NONBLOCK)
        if fd < 0:
            _stat_poll(path, timeout_s)
            return
        wd = _libc.inotify_add_watch(
            fd, parent.encode(),
            ctypes.c_uint32(_IN_CLOSE_WRITE | _IN_MOVED_TO)
        )
        try:
            # Check if already ready before we start watching
            if _is_ready(path):
                return
            deadline = time.monotonic() + timeout_s
            buf = ctypes.create_string_buffer(4096)
            while time.monotonic() < deadline:
                r, _, _ = select.select([fd], [], [], min(1.0, deadline - time.monotonic()))
                if r:
                    n = _libc.read(fd, buf, len(buf))
                    if n > 0 and _is_ready(path):
                        return
            raise TimeoutError(f"Shard not ready after {timeout_s}s: {path}")
        finally:
            _libc.inotify_rm_watch(fd, wd)
            os.close(fd)
    else:
        _stat_poll(path, timeout_s)


def _stat_poll(path: Path, timeout_s: float) -> None:
    backoff = 0.005
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_ready(path):
            return
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 0.5)
    raise TimeoutError(f"Shard not ready after {timeout_s}s: {path}")


def _is_ready(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            raw = f.read(_HDR_SIZE)
        if len(raw) < _HDR_SIZE:
            return False
        _, magic = struct.unpack(_HDR_FMT, raw)
        return magic == _READY_MAGIC
    except OSError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Shared shard cache
# ══════════════════════════════════════════════════════════════════════════════

class NodeSharedShardCache:
    """
    Shared-memory shard cache.  One instance per process; all processes on
    the same node share the same /dev/shm directory.

    Parameters
    ----------
    node_master     : True for local rank 0 — this process fills the cache.
    job_id          : Namespace for /dev/shm files (use SLURM_JOB_ID).
    max_shm_gb      : RAM budget in /dev/shm for this node.
    prefetch_window : How many shards to load concurrently (node master only).
    """

    def __init__(
        self,
        node_master:      bool,
        job_id:           str   = "dino",
        max_shm_gb:       float = 128.0,
        prefetch_window:  int   = 64,
    ):
        self._node_master    = node_master
        self._max_bytes      = int(max_shm_gb * (1 << 30))
        self._base           = Path(f"/dev/shm/{job_id}")
        self._base.mkdir(parents=True, exist_ok=True)

        # LRU tracking (node master only; non-masters read-only)
        self._lru:         OrderedDict[str, int]  = OrderedDict()  # path → byte size
        self._total_bytes: int = 0
        self._lru_lock:    threading.Lock = threading.Lock()

        if node_master:
            self._loop   = asyncio.new_event_loop()
            self._sem    = asyncio.Semaphore(prefetch_window)
            self._thread = threading.Thread(
                target=self._loop.run_forever, name="shard-io", daemon=True
            )
            self._thread.start()
            atexit.register(self._cleanup)
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: self._cleanup())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefetch(self, shard_path: str) -> None:
        """Schedule a shard for background loading (node master only; no-op otherwise)."""
        if not self._node_master:
            return
        shm = self._shm_path(shard_path)
        if shm.exists():
            return
        asyncio.run_coroutine_threadsafe(
            self._load_one(shard_path, shm), self._loop
        )

    def get(self, shard_path: str) -> bytes:
        """
        Return raw shard bytes.
        Node master: loads synchronously if not cached.
        Other ranks: waits via inotify until node master has written it.
        """
        shm = self._shm_path(shard_path)

        if self._node_master:
            if not _is_ready(shm):
                # Synchronous load on cold miss (happens only at epoch start
                # before the prefetch window is warm)
                future = asyncio.run_coroutine_threadsafe(
                    self._load_one(shard_path, shm), self._loop
                )
                future.result()   # block until done
            return self._read(shm)
        else:
            _inotify_wait(shm)
            return self._read(shm)

    @property
    def utilisation(self) -> float:
        with self._lru_lock:
            return self._total_bytes / max(self._max_bytes, 1)

    # ------------------------------------------------------------------
    # Async I/O  (node master only)
    # ------------------------------------------------------------------

    async def _load_one(self, shard_path: str, shm: Path) -> None:
        if _is_ready(shm):
            return
        async with self._sem:   # bound concurrency to prefetch_window
            if _is_ready(shm):  # double-check after acquiring sem
                return
            t0 = time.perf_counter()
            data = await self._read_lustre(shard_path)
            elapsed = time.perf_counter() - t0
            log.debug("Shard %s: %.0f MB in %.2fs (%.0f MB/s)",
                      Path(shard_path).name,
                      len(data) / (1 << 20), elapsed,
                      len(data) / (1 << 20) / max(elapsed, 1e-9))
            self._evict_for(len(data))
            self._write(shm, data)
            with self._lru_lock:
                self._lru[str(shm)] = len(data)
                self._total_bytes  += len(data)

    @staticmethod
    async def _read_lustre(path: str) -> bytes:
        try:
            import aiofiles
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except ImportError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: open(path, "rb").read()
            )

    # ------------------------------------------------------------------
    # SHM read / write
    # ------------------------------------------------------------------

    def _shm_path(self, shard_path: str) -> Path:
        key = hashlib.blake2b(shard_path.encode(), digest_size=16).hexdigest()
        return self._base / key

    @staticmethod
    def _write(shm: Path, data: bytes) -> None:
        """
        Atomic write: write data to a .tmp file, set the ready magic last,
        then rename into place.  rename() is atomic on POSIX; inotify
        IN_MOVED_TO fires only after the file is fully visible.
        """
        total = _HDR_SIZE + len(data)
        tmp   = shm.with_suffix(".tmp")
        with open(tmp, "w+b") as f:
            f.truncate(total)
            f.flush()
            with mmap.mmap(f.fileno(), total) as mm:
                mm[_HDR_SIZE:] = data
                mm.flush()
                # Write ready magic last so readers cannot see partial data
                struct.pack_into(_HDR_FMT, mm, 0, len(data), _READY_MAGIC)
                mm.flush()
        tmp.rename(shm)   # atomic on POSIX

    @staticmethod
    def _read(shm: Path) -> bytes:
        with open(shm, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data_len, magic = struct.unpack_from(_HDR_FMT, mm, 0)
                if magic != _READY_MAGIC:
                    raise RuntimeError(f"Shard {shm} has corrupt header")
                return bytes(mm[_HDR_SIZE: _HDR_SIZE + data_len])

    # ------------------------------------------------------------------
    # LRU eviction
    # ------------------------------------------------------------------

    def _evict_for(self, incoming: int) -> None:
        with self._lru_lock:
            while self._total_bytes + incoming > self._max_bytes and self._lru:
                path_str, sz = self._lru.popitem(last=False)
                try:
                    Path(path_str).unlink()
                    self._total_bytes -= sz
                except FileNotFoundError:
                    self._total_bytes -= sz

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        if not self._node_master:
            return
        import shutil
        try:
            shutil.rmtree(self._base, ignore_errors=True)
            log.info("Cleaned up shard cache at %s", self._base)
        except Exception:
            pass
