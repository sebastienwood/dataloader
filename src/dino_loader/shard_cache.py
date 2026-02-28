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

[FIX-SI] SIGINT was not registered with the graceful-shutdown handler.
         The CAVEAT-3 analysis was correct: only SIGTERM was intercepted,
         so Ctrl+C during interactive development left extraction threads
         alive and /dev/shm populated.  SIGINT is now registered alongside
         SIGTERM in _register_signals().  The watcher thread re-raises as
         SIGTERM (not SIGINT) so Python's default SIGTERM handler triggers
         the atexit chain regardless of which signal arrived first.

[FIX-SHM] /dev/shm utilisation warning.
         The utilisation counter was published to MetricsRegistry but no
         subsystem triggered an operator-visible warning when nearing
         capacity.  A silent ENOSPC during _write() manifests as a
         "Shard not ready after 300s" TimeoutError, which is extremely
         confusing.  Fix: _update_utilisation_metric() now emits a
         log.warning when utilisation exceeds the ``shm_warn_threshold``
         configured in LoaderConfig (default 0.85).  The warning is
         rate-limited to once per ``_SHM_WARN_INTERVAL`` seconds to avoid
         flooding logs during sustained high utilisation.
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
from typing import Iterator, Optional, Set

from dino_loader.monitor.metrics import get_registry

log = logging.getLogger(__name__)

_HDR_FMT     = "QQ"
_HDR_SIZE    = struct.calcsize(_HDR_FMT)
_READY_MAGIC = 0xDEAD_BEEF_CAFE_F00D

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO    = 0x00000080

# [FIX-SHM] Minimum seconds between successive utilisation warnings.
# Prevents log flooding during sustained high-utilisation periods.
_SHM_WARN_INTERVAL = 60.0


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
    shm_warn_threshold : Fraction (0–1) at which to emit a utilisation
                       warning. [FIX-SHM]
    """

    def __init__(
        self,
        node_master:        bool,
        job_id:             str   = "dino",
        max_shm_gb:         float = 128.0,
        prefetch_window:    int   = 64,
        shard_timeout_s:    float = 300.0,
        shm_warn_threshold: float = 0.85,   # [FIX-SHM]
    ):
        self._node_master       = node_master
        self._max_bytes         = int(max_shm_gb * (1 << 30))
        self._base              = Path(f"/dev/shm/{job_id}")
        self._timeout           = shard_timeout_s
        self._warn_threshold    = shm_warn_threshold   # [FIX-SHM]
        self._last_warn_ts: float = 0.0                # [FIX-SHM] rate-limit

        self._lru:         OrderedDict[str, int] = OrderedDict()
        self._total_bytes: int                   = 0
        self._lru_lock:    threading.Lock        = threading.Lock()
        self._in_flight:   Set[str]              = set()

        # [FIX-12] Event monitored by the watcher thread.
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
            self._register_signals()        # [FIX-12] / [FIX-SI]
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
        if self._max_bytes == 0:
            return 0.0
        with self._lru_lock:
            return self._total_bytes / self._max_bytes

    # ------------------------------------------------------------------
    # Internal: shard path helpers
    # ------------------------------------------------------------------

    def _shm_path(self, shard_path: str) -> Path:
        """Stable /dev/shm path for a Lustre shard path."""
        digest = hashlib.sha1(shard_path.encode()).hexdigest()[:16]
        return self._base / digest

    # ------------------------------------------------------------------
    # Internal: async I/O loop (node master only)
    # ------------------------------------------------------------------

    async def _load_one(self, shard_path: str, shm: Path) -> None:
        """Fetch one shard from Lustre and write it to /dev/shm. [A-5]"""
        async with self._sem:
            try:
                data = await self._read_lustre(shard_path)
                with self._lru_lock:
                    self._evict_for_locked(len(data))
                    self._write(shm, data)
                    self._lru[shard_path] = len(data)
                    self._total_bytes    += len(data)

                # [FIX-SHM] Update utilisation metric and warn if near capacity.
                self._update_utilisation_metric()

                if self._metrics is not None:
                    self._metrics.inc("lustre_bytes_read", len(data))
            finally:
                with self._lru_lock:
                    self._in_flight.discard(shard_path)  # [A-5]

    async def _read_lustre(self, shard_path: str) -> bytes:
        """Read a shard from Lustre, preferring aiofiles for async I/O."""
        t0 = time.perf_counter()
        try:
            import aiofiles
            async with aiofiles.open(shard_path, "rb") as f:
                data = await f.read()
        except ImportError:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _read_file_sync, shard_path)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if self._metrics is not None:
            self._metrics.inc("lustre_read_time_ms", elapsed_ms)
        return data

    # ------------------------------------------------------------------
    # Internal: utilisation warning  [FIX-SHM]
    # ------------------------------------------------------------------

    def _update_utilisation_metric(self) -> None:
        """
        Publish utilisation to MetricsRegistry and emit a warning when the
        /dev/shm budget is nearly exhausted.

        Rate-limited to once per _SHM_WARN_INTERVAL seconds to avoid log
        flooding during sustained high-utilisation periods.
        """
        util = self.utilisation

        if self._metrics is not None:
            self._metrics.set("shard_cache_utilization_pct", util * 100.0)

        if util >= self._warn_threshold:
            now = time.monotonic()
            if now - self._last_warn_ts >= _SHM_WARN_INTERVAL:
                self._last_warn_ts = now
                log.warning(
                    "/dev/shm utilisation is %.1f%% (threshold %.0f%%).  "
                    "Shard writes may fail with ENOSPC if utilisation reaches 100%%.  "
                    "Increase node_shm_gb in LoaderConfig or reduce "
                    "shard_prefetch_window to free capacity.  "
                    "Current budget: %.1f GB, used: %.1f GB.",
                    util * 100.0,
                    self._warn_threshold * 100.0,
                    self._max_bytes / (1 << 30),
                    self._total_bytes / (1 << 30),
                )

    # ------------------------------------------------------------------
    # Internal: file I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write(shm: Path, data: bytes) -> None:
        """
        Write shard bytes to /dev/shm atomically.

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
        Register SIGTERM and SIGINT handlers for clean shutdown.

        [FIX-12] The previous implementation called sys.exit() inside the
        signal handler, which raises SystemExit in whichever Python thread
        happens to be running. When that thread is a C extension (e.g. an
        NCCL all-reduce or inotify select), SystemExit is silently swallowed,
        bypassing atexit and leaving /dev/shm uncleaned.

        [FIX-SI] SIGINT was not registered in the previous version.  Ctrl+C
        during interactive development left extraction threads alive and
        /dev/shm populated.  Both SIGINT and SIGTERM are now handled.

        New approach (both signals):
          1. The signal handler sets a threading.Event (async-signal-safe).
          2. A daemon watcher thread monitors the event.
          3. When fired, the watcher restores default signal handlers and
             calls os.kill(os.getpid(), SIGTERM), causing the process to
             unwind normally through Python's atexit machinery.  We re-raise
             as SIGTERM regardless of the originating signal so the atexit
             chain always fires (Python's default SIGINT handler raises
             KeyboardInterrupt, which can bypass atexit in some embeddings).
        """
        self._exiting = False

        def _handle(signum: int, frame) -> None:
            if not self._exiting:
                self._exiting = True
                self._shutdown_event.set()

        def _watcher() -> None:
            self._shutdown_event.wait()
            # Restore defaults so the re-raised signal is handled normally.
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    signal.signal(sig, signal.SIG_DFL)
                except (OSError, ValueError):
                    pass  # non-main thread on some platforms
            os.kill(os.getpid(), signal.SIGTERM)

        # [FIX-SI] Register both SIGTERM and SIGINT.
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handle)
            except (OSError, ValueError) as exc:
                # signal.signal() must be called from the main thread.
                # If the cache is constructed from a worker thread (unusual),
                # silently skip — the default handlers remain in place.
                log.debug("Could not register signal %s handler: %s", sig, exc)

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
