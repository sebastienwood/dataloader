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

[FIX-ORPHAN] Application-level cleanup of orphaned /dev/shm directories.
         On SIGKILL or OOM-kill, atexit handlers do not run, leaving
         /dev/shm/<job_id>/ populated.  On busy clusters with frequent OOM
         crashes, these directories accumulate and can silently saturate
         the tmpfs (shared at node scope), causing ENOSPC for unrelated
         jobs.
         Previously, the recommendation was a SLURM prolog script deployed
         by cluster admins.  This fix internalises the cleanup:
         _purge_orphaned_shm() is called at node-master startup (inside
         _init_shm) and removes any /dev/shm/<N>/ directory (integer name
         = SLURM job ID) whose job is no longer alive according to squeue.
         Conservative fallback: if squeue is unavailable or times out (2s),
         no directories are removed so as not to risk destroying a live
         cache.

[FIX-PERM] /dev/shm/<job_id>/ created with mode 0o700 (user-only).
         The previous code used the process umask default, typically 0o755
         or 0o777, which allows other users on a shared node to read
         cached shard data.  mode=0o700 restricts access to the owning
         user only.

[FIX-HEADROOM] Real available space in /dev/shm is checked before each
         shard write via shutil.disk_usage("/dev/shm").free.
         The LRU budget (_max_bytes) only tracks bytes written by this
         process; other processes (e.g. NCCL, other jobs on the same node)
         can consume the shared tmpfs independently.  Without this check,
         _evict_for_locked() may successfully free LRU entries yet still
         hit ENOSPC on the actual write — manifesting as a confusing
         TimeoutError 300 s later.  A warning is emitted when headroom
         drops below 20 % of the incoming shard size (minimum 512 MB),
         and an IOError is raised immediately when free space is less than
         the shard size, providing a clear, actionable error message.
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
import subprocess   # [FIX-ORPHAN]
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

# [FIX-ORPHAN] Maximum seconds to wait for squeue before giving up.
_SQUEUE_TIMEOUT_S = 2.0

# [FIX-HEADROOM] Minimum free-space margin as a fraction of the incoming
# shard size.  Below this, a warning is emitted.  Below 1.0 (i.e. less
# free space than the shard itself), an IOError is raised immediately.
_SHM_HEADROOM_WARN_FACTOR  = 1.2   # warn if free < incoming * 1.2
_SHM_HEADROOM_MIN_WARN_MB  = 512   # always warn when < 512 MB free


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


def _is_slurm_job_alive(job_id: str, timeout_s: float = _SQUEUE_TIMEOUT_S) -> bool:
    """
    Return True if SLURM job *job_id* is still active (RUNNING or PENDING).

    Strategy
    --------
    1. Run ``squeue --jobs <id> --noheader`` with a hard timeout.
       - Exit 0 + non-empty stdout  → job is alive.
       - Exit 0 + empty stdout      → job has finished (COMPLETED / CANCELLED).
       - Exit 1                     → job ID unknown to SLURM → definitively dead.
    2. If squeue is not on PATH, or the SLURM controller does not respond
       within *timeout_s*, return True (conservative: assume alive so we
       never accidentally destroy a live cache).

    [FIX-ORPHAN]
    """
    try:
        result = subprocess.run(
            ["squeue", "--jobs", job_id, "--noheader"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
        # squeue exits 1 when the job ID is not found → definitively dead.
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # squeue not on PATH, or SLURM controller unreachable → be conservative.
        return True


def _purge_orphaned_shm(current_job_id: str) -> None:
    """
    Scan /dev/shm/ for directories left by dead SLURM jobs and remove them.

    Only directories whose name is a pure integer (i.e. a SLURM job ID) and
    that do NOT correspond to a live job are removed.  The current job's
    directory is always skipped — it is handled separately by _init_shm.

    This function is called once at node-master startup, before _init_shm
    creates the cache directory for the current job.

    Failure modes are handled gracefully:
    - /dev/shm not listable             → logged at DEBUG, no action.
    - squeue unavailable / timed out    → conservative skip (see _is_slurm_job_alive).
    - rmtree fails (permissions, race)  → logged at DEBUG, no action.

    [FIX-ORPHAN]
    """
    shm_root = Path("/dev/shm")
    try:
        candidates = [
            p for p in shm_root.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
    except OSError as exc:
        log.debug("Could not scan /dev/shm for orphaned job directories: %s", exc)
        return

    for candidate in candidates:
        if candidate.name == current_job_id:
            continue  # _init_shm handles the current job's own stale directory.

        if not _is_slurm_job_alive(candidate.name):
            try:
                shutil.rmtree(candidate)
                log.info(
                    "[FIX-ORPHAN] Removed orphaned /dev/shm/%s "
                    "(SLURM job no longer alive)",
                    candidate.name,
                )
            except Exception as exc:
                # Possible causes: directory belongs to another user (EPERM),
                # concurrent cleanup by another node's node-master (ENOENT),
                # or tmpfs still being written by a process we did not account
                # for.  All are safe to ignore — the directory will be cleaned
                # up on the next job startup, or by the OS when tmpfs pressure
                # eventually triggers eviction.
                log.debug(
                    "Could not remove orphaned /dev/shm/%s: %s",
                    candidate.name, exc,
                )


def _check_shm_headroom(incoming: int) -> None:
    """
    Verify that /dev/shm has enough real free space to absorb *incoming* bytes.

    /dev/shm is a node-scoped tmpfs shared by all processes (NCCL, other
    jobs, the OS itself).  The LRU budget (_max_bytes) only accounts for
    bytes written by this process; external consumers reduce available space
    independently.  Without this guard, _evict_for_locked() may free entries
    from our LRU yet still hit ENOSPC on the actual write — surfacing as a
    completely opaque "Shard not ready after 300s" TimeoutError.

    Behaviour
    ---------
    - Emits log.warning  when free space < incoming * _SHM_HEADROOM_WARN_FACTOR
      (or < _SHM_HEADROOM_MIN_WARN_MB, whichever is larger).
    - Raises IOError     when free space < incoming (write would certainly fail).

    This function is called by _load_one after _evict_for_locked() and
    before _write().  The shutil.disk_usage() call is cheap (single statfs
    syscall) and non-blocking.

    [FIX-HEADROOM]
    """
    try:
        free = shutil.disk_usage("/dev/shm").free
    except OSError as exc:
        # Should never happen on a well-configured Linux host, but don't
        # crash the prefetch coroutine over a monitoring call.
        log.debug("Could not stat /dev/shm usage: %s", exc)
        return

    warn_threshold = max(
        int(incoming * _SHM_HEADROOM_WARN_FACTOR),
        _SHM_HEADROOM_MIN_WARN_MB * (1 << 20),
    )

    if free < incoming:
        raise IOError(
            f"/dev/shm has only {free >> 20} MB free but needs {incoming >> 20} MB "
            f"for this shard.  Increase node_shm_gb in LoaderConfig, reduce "
            f"shard_prefetch_window, or free space from other processes."
        )

    if free < warn_threshold:
        log.warning(
            "[FIX-HEADROOM] /dev/shm headroom low: %d MB free, "
            "incoming shard is %d MB (warn threshold %d MB).  "
            "ENOSPC may occur soon.",
            free >> 20,
            incoming >> 20,
            warn_threshold >> 20,
        )


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
                    _check_shm_headroom(len(data))   # [FIX-HEADROOM]
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
        """
        Prepare the /dev/shm cache directory for this job.

        Steps
        -----
        1. [FIX-ORPHAN] Purge /dev/shm directories from dead SLURM jobs to
           prevent gradual tmpfs saturation across successive OOM / SIGKILL
           crashes.  Uses _purge_orphaned_shm() which queries squeue with a
           conservative 2-second timeout.
        2. [A-2] Remove this job's own stale directory if it exists from a
           previous run with the same job ID (can happen with SLURM job
           arrays or during development restarts).
        3. [FIX-PERM] Create the directory with mode 0o700 (user-only) so
           other users on a shared node cannot read cached shard data.
        """
        # Step 1 — evict dead jobs' leftovers.
        _purge_orphaned_shm(self._base.name)   # [FIX-ORPHAN]

        # Step 2 — remove this job's own stale directory.
        if self._base.exists():
            log.info("Removing stale shard cache at %s", self._base)
            shutil.rmtree(self._base, ignore_errors=True)

        # Step 3 — create with restricted permissions.
        self._base.mkdir(parents=True, exist_ok=True, mode=0o700)  # [FIX-PERM]

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
            shutil.rmtree(self._base)
            log.info("Cleaned up shard cache at %s", self._base)
        except FileNotFoundError:
            pass  # already cleaned up (e.g. double atexit call)
        except Exception as exc:
            log.warning("Partial cleanup of %s: %s", self._base, exc)
