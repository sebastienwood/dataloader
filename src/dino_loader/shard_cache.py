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

Changes from previous version (intern review)
----------------------------------------------
ACCEPTED
  [A-1] _in_flight set: prevents duplicate concurrent downloads for the
        same shard when prefetch() races with itself.
  [A-2] _init_shm: removes orphaned /dev/shm files from a previous crashed
        run with the same SLURM_JOB_ID before the new job starts.
  [A-3] shard_timeout_s constructor parameter: configurable wait timeout;
        120 s can be too short on a heavily loaded Lustre filesystem serving
        hundreds of ranks simultaneously.
  [A-4] .tmp cleanup in _evict_for: unlinks both the shard file and any
        residual .tmp file left by a crash mid-write.
  [A-5] _in_flight.discard in finally block of _load_one: ensures the path
        is removed from the in-flight set even if _read_lustre raises, so
        the shard stays fetchable for future calls.

REJECTED / FIXED
  [R-1] memoryview / fd-leak in _read_zero_copy: rejected entirely.
        The intern's zero-copy read left the file descriptor open ("mmap
        needs the fd" — incorrect on Linux; the fd can be closed once mmap()
        returns).  More critically, the returned memoryview was backed by an
        mmap with no surviving Python reference, so the GC could unmap the
        memory while the caller still held the view — silent data corruption.
        We keep returning bytes (a safe owned copy).
  [R-2] Lock held across file I/O in _load_one: the intern's version called
        _write() while holding self._lock, serialising all prefetch workers
        for the duration of a potentially 500 ms write.  Lock scope restored
        to cover only the metadata update after the write completes.
  [R-3] os._exit in signal handler: skips atexit handlers, prevents the
        asyncio loop from shutting down cleanly, and can leave orphaned .tmp
        files.  Replaced with sys.exit() which raises SystemExit and unwinds
        normally (atexit runs, including _cleanup).
  [R-4] aiofiles fallback removed in intern version: hard ImportError at
        runtime.  Fallback to run_in_executor restored.
  [R-5] inotify via os module (neutral improvement): accepted for cleanliness,
        but the missing `import select` from the intern's file is restored.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import mmap
import os
import select
import shutil
import signal
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Set

log = logging.getLogger(__name__)

# Header layout: [data_len: u64, ready_magic: u64] — 16 bytes total
_HDR_FMT  = "QQ"
_HDR_SIZE = struct.calcsize(_HDR_FMT)   # 16

# Sentinel written last so readers can distinguish a complete write from a
# partial one.  Chosen to be astronomically unlikely as an uninitialised value.
_READY_MAGIC = 0xDEAD_BEEF_CAFE_F00D

# inotify event masks
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO    = 0x00000080


# ══════════════════════════════════════════════════════════════════════════════
# inotify / stat-poll helpers
# ══════════════════════════════════════════════════════════════════════════════

def _inotify_wait(path: Path, timeout_s: float) -> None:
    """
    Block until `path` is fully written and ready, using inotify on its
    parent directory.  Falls back to exponential-backoff stat() on platforms
    where inotify is unavailable.

    Uses os.inotify_init / os.inotify_add_watch (Python 3.9+, Linux).
    Falls back gracefully if those symbols are absent.
    """
    if _is_ready(path):
        return

    if not (hasattr(os, "inotify_init") and hasattr(os, "inotify_add_watch")):
        _stat_poll(path, timeout_s)
        return

    inotify_fd = os.inotify_init()
    try:
        # Watch the parent dir for IN_MOVED_TO (triggered by our atomic rename)
        # and IN_CLOSE_WRITE as a belt-and-suspenders fallback.
        os.inotify_add_watch(
            inotify_fd,
            str(path.parent),
            _IN_MOVED_TO | _IN_CLOSE_WRITE,
        )

        # Re-check after registering the watch to close the TOCTOU window:
        # the file may have appeared between our initial check and watch setup.
        if _is_ready(path):
            return

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            r, _, _ = select.select([inotify_fd], [], [], min(1.0, remaining))
            if r:
                os.read(inotify_fd, 4096)   # drain the event queue
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
    Shared-memory shard cache.  One instance per process; all processes on
    the same node share the same /dev/shm directory.

    Parameters
    ----------
    node_master      : True for local rank 0 — this process fills the cache.
    job_id           : Namespace for /dev/shm files (use SLURM_JOB_ID).
    max_shm_gb       : RAM budget in /dev/shm for this node.
    prefetch_window  : Max concurrent shard downloads (node master only).
    shard_timeout_s  : How long non-master ranks wait for a shard before
                       raising TimeoutError.  300 s is safe for busy Lustre.
    """

    def __init__(
        self,
        node_master:     bool,
        job_id:          str   = "dino",
        max_shm_gb:      float = 128.0,
        prefetch_window: int   = 64,
        shard_timeout_s: float = 300.0,   # [A-3]
    ):
        self._node_master    = node_master
        self._max_bytes      = int(max_shm_gb * (1 << 30))
        self._base           = Path(f"/dev/shm/{job_id}")
        self._timeout        = shard_timeout_s

        # LRU metadata — guarded by _lru_lock.
        # Writes: only in _load_one (node master async loop).
        # Reads:  utilisation property (any thread).
        self._lru:         OrderedDict[str, int] = OrderedDict()  # shm_path → bytes
        self._total_bytes: int = 0
        self._lru_lock:    threading.Lock = threading.Lock()

        # In-flight set — prevents duplicate concurrent downloads. [A-1]
        # Shares _lru_lock (same critical section, avoids a second lock).
        self._in_flight: Set[str] = set()

        if node_master:
            self._init_shm()   # [A-2] clean orphans before starting
            self._loop   = asyncio.new_event_loop()
            self._sem    = asyncio.Semaphore(prefetch_window)
            self._thread = threading.Thread(
                target=self._loop.run_forever, name="shard-io", daemon=True
            )
            self._thread.start()
            atexit.register(self._cleanup)
            self._register_signals()
        else:
            # Non-master ranks only read; ensure the directory exists so
            # inotify_add_watch on the parent does not fail before the master
            # has had a chance to create it.
            self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefetch(self, shard_path: str) -> None:
        """
        Schedule a shard for background loading (node master only; no-op
        on other ranks).  Safe to call redundantly — duplicate submissions
        are suppressed by the _in_flight set. [A-1]
        """
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
        """
        Return raw shard bytes.

        Node master  : loads synchronously on cold miss (rare after pre-warm).
        Other ranks  : waits via inotify until node master has written it.

        Returns bytes — a safe owned copy.  Zero-copy memoryview was
        considered and rejected: it requires the caller to hold the backing
        mmap alive, creating an implicit lifetime contract that is unsafe
        across the DALI ExternalSource boundary. [R-1]
        """
        shm = self._shm_path(shard_path)

        if self._node_master:
            if not _is_ready(shm):
                # Synchronous fallback for cold miss.
                # Guard with _in_flight to avoid a duplicate submission if
                # prefetch() already queued this shard.
                with self._lru_lock:
                    if shard_path not in self._in_flight:
                        self._in_flight.add(shard_path)
                future = asyncio.run_coroutine_threadsafe(
                    self._load_one(shard_path, shm), self._loop
                )
                future.result()   # block until done
            return self._read(shm)
        else:
            import time
            from dino_loader.monitor.tracing import trace
            from dino_loader.monitor.metrics import get_registry
            
            t0 = time.time()
            with trace("shard_wait", "io"):
                _inotify_wait(shm, self._timeout)
            
            reg = get_registry()
            if reg:
                reg.inc("shard_wait_time_ms", int((time.time() - t0) * 1000.0))
            
            return self._read(shm)

    @contextlib.contextmanager
    def get_view(self, shard_path: str) -> Iterator[memoryview]:
        """
        Yield a zero-copy memoryview into the shard file.
        The caller MUST NOT let the memoryview escape the `with` block,
        otherwise it will reference a closed mmap.
        """
        shm = self._shm_path(shard_path)

        if self._node_master:
            if not _is_ready(shm):
                with self._lru_lock:
                    if shard_path not in self._in_flight:
                        self._in_flight.add(shard_path)
                future = asyncio.run_coroutine_threadsafe(
                    self._load_one(shard_path, shm), self._loop
                )
                future.result()
        else:
            import time
            from dino_loader.monitor.tracing import trace
            from dino_loader.monitor.metrics import get_registry
            
            t0 = time.time()
            with trace("shard_wait", "io"):
                _inotify_wait(shm, self._timeout)
                
            reg = get_registry()
            if reg:
                reg.inc("shard_wait_time_ms", int((time.time() - t0) * 1000.0))

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
        """
        Download one shard and write it to /dev/shm.

        The _in_flight entry is always removed in the finally block so that
        any exception (network error, corrupt shard, Lustre ENOENT) does not
        permanently blacklist the path. [A-5]

        The lock is held only for LRU metadata updates, never across file
        I/O, so concurrent prefetch coroutines are not serialised. [R-2]
        """
        try:
            async with self._sem:
                if _is_ready(shm):
                    return   # arrived while waiting for the semaphore

                t0   = time.perf_counter()
                data = await self._read_lustre(shard_path)
                elapsed = time.perf_counter() - t0
                log.debug(
                    "Shard %s: %.0f MB in %.2fs (%.0f MB/s)",
                    Path(shard_path).name,
                    len(data) / (1 << 20),
                    elapsed,
                    len(data) / (1 << 20) / max(elapsed, 1e-9),
                )

                # Reserve space under lock, then write outside it. [R-2]
                with self._lru_lock:
                    self._evict_for_locked(len(data))

                self._write(shm, data)   # atomic rename; no lock needed

                with self._lru_lock:
                    self._lru[str(shm)] = len(data)
                    self._total_bytes  += len(data)
                    
                from dino_loader.monitor.metrics import get_registry
                reg = get_registry()
                if reg:
                    reg.inc("lustre_bytes_read", len(data))
                    reg.inc("lustre_read_time_ms", int(elapsed * 1000.0))
                    reg.set("shard_cache_utilization_pct", self.utilisation * 100.0)
        finally:
            with self._lru_lock:
                self._in_flight.discard(shard_path)   # [A-5]

    @staticmethod
    async def _read_lustre(path: str) -> bytes:
        """Async shard read; executor fallback when aiofiles is absent. [R-4]"""
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
        Atomic write via a temporary file and POSIX rename.

        Layout: [data_len: u64][ready_magic: u64][data bytes...]

        Data is written before the magic sentinel, so a reader that sees a
        valid magic is guaranteed to see complete data.  rename() provides
        the POSIX visibility barrier; inotify IN_MOVED_TO fires only after
        the destination path is fully visible to all processes.
        """
        total = _HDR_SIZE + len(data)
        tmp   = shm.with_suffix(".tmp")
        with open(tmp, "w+b") as f:
            f.truncate(total)
            f.flush()
            with mmap.mmap(f.fileno(), total) as mm:
                mm[_HDR_SIZE:] = data
                mm.flush()
                struct.pack_into(_HDR_FMT, mm, 0, len(data), _READY_MAGIC)
                mm.flush()
        tmp.rename(shm)   # atomic on POSIX

    @staticmethod
    def _read(shm: Path) -> bytes:
        """
        Read shard bytes into an owned bytes object.

        The file descriptor is closed immediately after mmap() returns;
        on Linux the kernel holds its own reference to the underlying file
        via the mapping, so closing the fd does not invalidate the mmap.
        The final bytes() copy ensures the caller owns the data and the
        mmap can be released promptly.
        """
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
        """
        Evict LRU shards until there is room for `incoming` bytes.
        Caller MUST hold self._lru_lock.
        Cleans up residual .tmp files from crashed mid-writes. [A-4]
        """
        while self._total_bytes + incoming > self._max_bytes and self._lru:
            path_str, sz = self._lru.popitem(last=False)
            p = Path(path_str)
            try:
                p.unlink(missing_ok=True)
                p.with_suffix(".tmp").unlink(missing_ok=True)   # [A-4]
                self._total_bytes -= sz
            except Exception as exc:
                log.warning("Eviction failed for %s: %s", path_str, exc)
                self._total_bytes -= sz   # deduct anyway; avoid infinite loop

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    def _init_shm(self) -> None:
        """
        Remove any stale /dev/shm/<job_id>/ directory left by a previous
        crashed run before creating a fresh one. [A-2]
        """
        if self._base.exists():
            log.info("Removing orphaned shard cache at %s", self._base)
            shutil.rmtree(self._base, ignore_errors=True)
        self._base.mkdir(parents=True, exist_ok=True)

    def _register_signals(self) -> None:
        """
        Register SIGTERM / SIGINT handlers that trigger a clean shutdown.

        Uses sys.exit() — not os._exit() — so that SystemExit propagates
        normally, atexit handlers run, and _cleanup removes /dev/shm. [R-3]
        An _exiting flag prevents re-entrant cleanup if a second signal
        arrives before the first has finished unwinding.
        """
        self._exiting = False

        def _handle(signum, frame):
            if not self._exiting:
                self._exiting = True
                sys.exit(0)   # raises SystemExit → atexit → _cleanup

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _handle)

    def _cleanup(self) -> None:
        """Remove the entire /dev/shm cache directory (node master only)."""
        if not self._node_master:
            return
        try:
            shutil.rmtree(self._base, ignore_errors=True)
            log.info("Cleaned up shard cache at %s", self._base)
        except Exception:
            pass
