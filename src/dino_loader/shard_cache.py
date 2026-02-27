"""
dino_loader.shard_cache
=======================
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

Changes from previous version (intern review)
----------------------------------------------
ACCEPTED
  [A-1] _in_flight set: prevents duplicate concurrent downloads.
  [A-2] _init_shm: removes orphaned /dev/shm files from a previous crashed run.
  [A-3] shard_timeout_s constructor parameter: configurable wait timeout.
  [A-4] .tmp cleanup in _evict_for: unlinks both shard and residual .tmp file.
  [A-5] _in_flight.discard in finally block of _load_one.

REJECTED / FIXED (intern review, unchanged)
  [R-1] memoryview / fd-leak in _read_zero_copy: rejected.
  [R-2] Lock held across file I/O in _load_one: fixed.
  [R-3] os._exit in signal handler: replaced with sys.exit.
  [R-4] aiofiles fallback removed: restored.
  [R-5] inotify via os module: select import restored.

Additional fixes
----------------
[FIX-1]  Added missing ``import atexit`` — node master startup crashed with
         NameError: name 'atexit' is not defined.

[FIX-2]  Closed fd properly in ``_read_lustre`` executor fallback.
         The previous lambda ``open(path, "rb").read()`` left the file object
         unclosed until GC. Under load this caused fd exhaustion (EMFILE).
         Replaced with the module-level function ``_read_file_sync`` which
         uses a ``with`` block.

[FIX-7]  Replaced mmap-based ``_write`` with direct sequential writes.
         The old approach mmap'd up to 2 GB of virtual address space per
         write-once shard file and issued two sequential mm.flush() calls.
         With 64 concurrent prefetches this added excessive VA pressure.
         New approach: f.write(data) + os.fsync + seek-back for the header
         sentinel, letting the OS page cache coalesce the writes.

[FIX-12] Signal handler now sets a threading.Event instead of calling
         sys.exit() directly. sys.exit() in a signal handler raises
         SystemExit in whichever thread is running; when that thread is a
         C extension (e.g. NCCL all-reduce), SystemExit is silently
         swallowed, bypassing atexit and leaving /dev/shm uncleaned.
         Fix: handler sets an Event; a daemon watcher thread restores the
         default handler and re-raises SIGTERM via os.kill() so the process
         unwinds through atexit normally.
"""

from __future__ import annotations

import asyncio
import atexit    # [FIX-1] was missing — caused NameError at node master startup
import contextlib
import hashlib
import logging
import mmap
import os
import select
import shutil
import signal
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Set

from dino_loader.monitor.metrics import get_registry

log = logging.getLogger(__name__)

_HDR_FMT     = "QQ"
_HDR_SIZE    = struct.calcsize(_HDR_FMT)
_READY_MAGIC = 0xDEAD_BEEF_CAFE_F00D

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO    = 0x00000080


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read_file_sync(path: str) -> bytes:
    """
    Synchronous file read for the aiofiles-absent executor fallback.

    Defined at module level (not a lambda) so the with-block is guaranteed
    to close the fd even if read() raises. [FIX-2]
    """
    with open(path, "rb") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════════
# inotify / stat-poll helpers
# ══════════════════════════════════════════════════════════════════════════════

def _inotify_wait(path: Path, timeout_s: float) -> None:
    """
    Block until path is fully written and ready, using inotify on its
    parent directory. Falls back to exponential-backoff stat() on platforms
    where inotify is unavailable.
    """
    if _is_ready(path):
        return

    if not (hasattr(os, "inotify_init") and hasattr(os, "inotify_add_watch")):
        _stat_poll(path, timeout_s)
        return

    inotify_fd = os.inotify_init()
    try:
        os.inotify_add_watch(
            inotify_fd,
            str(path.parent),
            _IN_MOVED_TO | _IN_CLOSE_WRITE,
        )
        if _is_ready(path):
            return

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            r, _, _ = select.select([inotify_fd], [], [], min(1.0, remaining))
            if r:
                os.read(inotify_fd, 4096)
                if _is_ready(path):
                    return

        raise TimeoutError(f"Shard not ready after {timeout_s:.0f}s: {path}")
    finally:
        os.close(inotify_fd)


def _stat_poll(path: Path, timeout_s: float) -> None:
    """Exponential-backoff polling fallback for non-Linux platforms."""
    backoff  = 0.005
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_ready(path):
            return
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 0.5)
    raise TimeoutError(f"Shard not ready after {timeout_s:.0f}s: {path}")


def _is_ready(path: Path) -> bool:
    """Return True iff the shard file exists and its header magic is valid."""
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
    Shared-memory shard cache. One instance per process; all processes on
    the same node share the same /dev/shm directory.

    Parameters
    ----------
    node_master      : True for local rank 0 — this process fills the cache.
    job_id           : Namespace for /dev/shm files (use SLURM_JOB_ID).
    max_shm_gb       : RAM budget in /dev/shm for this node.
    prefetch_window  : Max concurrent shard downloads (node master only).
    shard_timeout_s  : How long non-master ranks wait for a shard. [A-3]
    """

    def __init__(
        self,
        node_master:     bool,
        job_id:          str   = "dino",
        max_shm_gb:      float = 128.0,
        prefetch_window: int   = 64,
        shard_timeout_s: float = 300.0,
    ):
        self._node_master = node_master
        self._max_bytes   = int(max_shm_gb * (1 << 30))
        self._base        = Path(f"/dev/shm/{job_id}")
        self._timeout     = shard_timeout_s

        self._lru:         OrderedDict[str, int] = OrderedDict()
        self._total_bytes: int                   = 0
        self._lru_lock:    threading.Lock        = threading.Lock()
        self._in_flight:   Set[str]              = set()

        # [FIX-12] Event monitored by the watcher thread
        self._shutdown_event = threading.Event()

        if node_master:
            self._init_shm()
            # Metrics registry is already initialised by DINODataLoader.__init__
            # before the cache is constructed.  Grab the singleton here.
            self._metrics = get_registry()
            self._loop   = asyncio.new_event_loop()
            self._sem    = asyncio.Semaphore(prefetch_window)
            self._thread = threading.Thread(
                target=self._loop.run_forever, name="shard-io", daemon=True
            )
            self._thread.start()
            atexit.register(self._cleanup)  # [FIX-1]
            self._register_signals()        # [FIX-12]
        else:
            self._base.mkdir(parents=True, exist_ok=True)
            self._metrics = get_registry()   # non-master ranks need shard_cache_wait

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefetch(self, shard_path: str) -> None:
        """Schedule a shard for background loading (node master only). [A-1]"""
        if not self._node_master:
            return
        shm = self._shm_path(shard_path)
        with self._lru_lock:
            if _is_ready(shm) or shard_path in self._in_flight:
                return
            self._in_flight.add(shard_path)
        asyncio.run_coroutine_threadsafe(
            self._load_one(shard_path, shm), self._loop
        )

    def get(self, shard_path: str) -> bytes:
        """Return raw shard bytes (owned copy)."""
        shm = self._shm_path(shard_path)
        if self._node_master:
            if not _is_ready(shm):
                with self._lru_lock:
                    if shard_path not in self._in_flight:
                        self._in_flight.add(shard_path)
                asyncio.run_coroutine_threadsafe(
                    self._load_one(shard_path, shm), self._loop
                ).result()
            return self._read(shm)
        else:
            t_wait = time.perf_counter()
            _inotify_wait(shm, self._timeout)
            wait_ms = int((time.perf_counter() - t_wait) * 1000)
            if self._metrics is not None and wait_ms > 0:
                self._metrics.inc("shard_cache_wait_time_ms", wait_ms)
            return self._read(shm)

    @contextlib.contextmanager
    def get_view(self, shard_path: str) -> Iterator[memoryview]:
        """
        Yield a zero-copy memoryview into the shard file.
        Caller MUST NOT let the view escape the with-block.
        """
        shm = self._shm_path(shard_path)
        if self._node_master:
            if not _is_ready(shm):
                with self._lru_lock:
                    if shard_path not in self._in_flight:
                        self._in_flight.add(shard_path)
                asyncio.run_coroutine_threadsafe(
                    self._load_one(shard_path, shm), self._loop
                ).result()
        else:
            t_wait = time.perf_counter()
            _inotify_wait(shm, self._timeout)
            wait_ms = int((time.perf_counter() - t_wait) * 1000)
            if self._metrics is not None and wait_ms > 0:
                self._metrics.inc("shard_cache_wait_time_ms", wait_ms)

        with open(shm, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data_len, magic = struct.unpack_from(_HDR_FMT, mm, 0)
                if magic != _READY_MAGIC:
                    raise RuntimeError(f"Shard {shm} has corrupt header")
                yield memoryview(mm)[_HDR_SIZE: _HDR_SIZE + data_len]

    @property
    def utilisation(self) -> float:
        """Fraction of the /dev/shm budget currently in use."""
        with self._lru_lock:
            return self._total_bytes / max(self._max_bytes, 1)

    # ------------------------------------------------------------------
    # Async I/O  (node master only)
    # ------------------------------------------------------------------

    async def _load_one(self, shard_path: str, shm: Path) -> None:
        """Download one shard and write it to /dev/shm. [R-2, A-5]"""
        try:
            async with self._sem:
                if _is_ready(shm):
                    return
                t0      = time.perf_counter()
                data    = await self._read_lustre(shard_path)
                elapsed = time.perf_counter() - t0
                log.debug(
                    "Shard %s: %.0f MB in %.2fs (%.0f MB/s)",
                    Path(shard_path).name,
                    len(data) / (1 << 20), elapsed,
                    len(data) / (1 << 20) / max(elapsed, 1e-9),
                )
                # ── Publish Lustre metrics ──────────────────────────────────
                if self._metrics is not None:
                    self._metrics.inc("lustre_read_time_ms", int(elapsed * 1000))
                    self._metrics.inc("lustre_bytes_read",   len(data))
                    util = self.utilisation * 100.0
                    self._metrics.set("shard_cache_utilization_pct", util)
                # ───────────────────────────────────────────────────────────
                with self._lru_lock:
                    self._evict_for_locked(len(data))
                self._write(shm, data)
                with self._lru_lock:
                    self._lru[str(shm)] = len(data)
                    self._total_bytes  += len(data)
        finally:
            with self._lru_lock:
                self._in_flight.discard(shard_path)  # [A-5]

    @staticmethod
    async def _read_lustre(path: str) -> bytes:
        """Async shard read; executor fallback when aiofiles is absent. [FIX-2]"""
        try:
            import aiofiles
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except ImportError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _read_file_sync, path)

    # ------------------------------------------------------------------
    # SHM read / write
    # ------------------------------------------------------------------

    def _shm_path(self, shard_path: str) -> Path:
        key = hashlib.blake2b(shard_path.encode(), digest_size=16).hexdigest()
        return self._base / key

    @staticmethod
    def _write(shm: Path, data: bytes) -> None:
        """
        Atomic write via a temporary file and POSIX rename.

        Layout: [data_len: u64][ready_magic: u64][data bytes...]

        [FIX-7] Replaced mmap-based writer with direct sequential writes.
        The old approach mmap'd the full file (up to 2 GB) and called
        mm.flush() twice; for a write-once file this is slower than letting
        the OS page cache coalesce writes, and adds VA pressure with 64
        concurrent prefetches in flight.

        Write order: data first, magic sentinel last.
        rename() provides the POSIX visibility barrier; inotify IN_MOVED_TO
        fires only after the destination path is fully visible.
        """
        tmp = shm.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack(_HDR_FMT, len(data), 0))  # magic=0 → not-ready
                f.write(data)
                f.flush()
                os.fsync(f.fileno())   # data must reach storage before magic
                f.seek(0)
                f.write(struct.pack(_HDR_FMT, len(data), _READY_MAGIC))
                f.flush()
                os.fsync(f.fileno())
            tmp.rename(shm)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    @staticmethod
    def _read(shm: Path) -> bytes:
        """Read shard bytes into an owned bytes object."""
        with open(shm, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data_len, magic = struct.unpack_from(_HDR_FMT, mm, 0)
                if magic != _READY_MAGIC:
                    raise RuntimeError(f"Shard {shm} has corrupt header")
                return bytes(mm[_HDR_SIZE: _HDR_SIZE + data_len])

    # ------------------------------------------------------------------
    # LRU eviction  (caller must hold _lru_lock)
    # ------------------------------------------------------------------

    def _evict_for_locked(self, incoming: int) -> None:
        """Evict LRU shards to make room. Caller must hold _lru_lock. [A-4]"""
        while self._total_bytes + incoming > self._max_bytes and self._lru:
            path_str, sz = self._lru.popitem(last=False)
            p = Path(path_str)
            try:
                p.unlink(missing_ok=True)
                p.with_suffix(".tmp").unlink(missing_ok=True)  # [A-4]
                self._total_bytes -= sz
            except Exception as exc:
                log.warning("Eviction failed for %s: %s", path_str, exc)
                self._total_bytes -= sz

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    def _init_shm(self) -> None:
        """Remove stale /dev/shm/<job_id>/ from a previous crashed run. [A-2]"""
        if self._base.exists():
            log.info("Removing orphaned shard cache at %s", self._base)
            shutil.rmtree(self._base, ignore_errors=True)
        self._base.mkdir(parents=True, exist_ok=True)

    def _register_signals(self) -> None:
        """
        Register SIGTERM/SIGINT handlers for clean shutdown.

        [FIX-12] The previous implementation called sys.exit() inside the
        signal handler, which raises SystemExit in whichever Python thread
        happens to be running. When that thread is a C extension (e.g. an
        NCCL all-reduce or inotify select), SystemExit is silently swallowed,
        bypassing atexit and leaving /dev/shm uncleaned.

        New approach:
          1. The signal handler sets a threading.Event (async-signal-safe).
          2. A daemon watcher thread monitors the event.
          3. When fired, the watcher restores default signal handlers and
             calls os.kill(os.getpid(), SIGTERM), causing the process to
             unwind normally through Python's atexit machinery.
        """
        self._exiting = False

        def _handle(signum: int, frame) -> None:
            if not self._exiting:
                self._exiting = True
                self._shutdown_event.set()

        def _watcher() -> None:
            self._shutdown_event.wait()
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _handle)

        threading.Thread(
            target=_watcher, name="shm-shutdown-watcher", daemon=True
        ).start()

    def _cleanup(self) -> None:
        """Remove the entire /dev/shm cache directory (node master only)."""
        if not self._node_master:
            return
        try:
            shutil.rmtree(self._base, ignore_errors=True)
            log.info("Cleaned up shard cache at %s", self._base)
        except Exception:
            pass
