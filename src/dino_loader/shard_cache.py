"""dino_loader.shard_cache

=======================
Node-local shared-memory shard cache.

Design
------
- One backing file per shard per node in /dev/shm/<job_id>/.
- Local rank 0 ("node master") reads from the parallel filesystem and writes.
- Other local ranks wait via inotify (Linux) or stat-poll fallback —
  not a busy-spin, so waiting ranks cost near-zero CPU.
- LRU eviction at file granularity.
- Header encodes (data_len, ready_flag) so readers can detect partial writes.
- asyncio + aiofiles for concurrent shard prefetch (hides filesystem latency).

Filesystem compatibility
------------------------
This cache is filesystem-agnostic: the shards can live on Lustre, CephFS
(kernel client), NFS, or any other POSIX-compatible parallel filesystem.

inotify support:
  - Lustre (lnet/llite)        : supported
  - CephFS kernel client        : supported (kernel ≥ 4.14)
  - CephFS FUSE (ceph-fuse)     : NOT supported (ENOSYS on inotify_init1)
  - NFS (kernel client)         : supported on recent kernels with CONFIG_NFS_V4_2

When inotify is unavailable (e.g. FUSE mount), _inotify_wait() detects the
failure and falls back to stat-polling automatically.  A WARNING is logged
once per process to alert operators that the efficient path is unavailable.

Corrections
-----------
[B2-FIX]    _evict_for_locked: backpressure instead of grow-and-proceed.
[FIX-EVICT] _evict_for_locked accounting drift fix.
[FIX-EVICT-EARLY] Early exit dans _load_one si le shard dépasse le budget
            total, évitant 20 s d'attente inutile.
[FIX-HB]    Heartbeat file now stores "pid:job_id".
[FIX-READ]  _read_file_sync utilise pathlib.Path.read_bytes() pour une
            lecture correcte sur tous les filesystems (évite l'os.read()
            partiel sur NFS/Lustre).
[FIX-SIGNAL] _register_signals chaîne les handlers SIGTERM/SIGINT existants
             (installés par PyTorch distributed / NCCL) plutôt que de les
             écraser, évitant les deadlocks NCCL à la terminaison.
[FIX-STALE] heartbeat_stale_s transmis à _init_shm/_purge_orphaned_shm
            (était ignoré : _init_shm utilisait la constante _HB_STALE_S).
[PERF-1]    No fsync on tmpfs.
[PERF-2]    Persistent mmap pool.
[LOG-1]     INFO→DEBUG demotion for per-shard logs.
[FS-1]      _read_lustre renamed to _read_shard (filesystem-neutral).
[FS-2]      _inotify_wait detects FUSE mounts and falls back gracefully.
"""

import asyncio
import atexit
import contextlib
import hashlib
import logging
import mmap
import os
import select
import shutil
import signal
import struct
import subprocess
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

from dino_loader.monitor.metrics import MetricField, get_registry

log = logging.getLogger(__name__)

_HDR_FMT     = "QQ"
_HDR_SIZE    = struct.calcsize(_HDR_FMT)
_READY_MAGIC = 0xDEAD_BEEF_CAFE_F00D

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO    = 0x00000080

_SHM_WARN_INTERVAL        = 60.0
_SQUEUE_TIMEOUT_S         = 2.0
_SHM_HEADROOM_WARN_FACTOR = 1.2
_SHM_HEADROOM_MIN_WARN_MB = 512
_MMAP_POOL_MAX            = 256

_HB_INTERVAL_S = 10.0
_HB_STALE_S    = 300.0
_HB_FILENAME   = "heartbeat"

_EVICT_WAIT_S  = 2.0
_EVICT_RETRIES = 10

# Whether inotify is available on this process's filesystem.
_INOTIFY_AVAILABLE: bool = True


class _MmapEntry:
    __slots__ = ("data_len", "fd", "mm", "refs")

    def __init__(self, fd: int, mm: mmap.mmap, data_len: int) -> None:
        self.fd       = fd
        self.mm       = mm
        self.data_len = data_len
        self.refs     = 0


class _MmapPool:
    """Thread-safe pool of persistent memory-mapped shard files."""

    def __init__(self, max_entries: int = _MMAP_POOL_MAX) -> None:
        self._max   = max_entries
        self._pool: OrderedDict[str, _MmapEntry] = OrderedDict()
        self._lock  = threading.Lock()

    def acquire(self, path: Path) -> _MmapEntry:
        """Acquire a memory-mapped entry for *path*, opening it if necessary."""
        key = str(path)
        with self._lock:
            if key in self._pool:
                entry = self._pool[key]
                self._pool.move_to_end(key)
                entry.refs += 1
                return entry
            self._evict_unreferenced()
            fd = os.open(key, os.O_RDONLY)
            try:
                mm              = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                data_len, magic = struct.unpack_from(_HDR_FMT, mm, 0)
                if magic != _READY_MAGIC:
                    mm.close()
                    os.close(fd)
                    raise RuntimeError(
                        f"Shard {path} has corrupt header (magic={magic:#x})",
                    )
                entry      = _MmapEntry(fd, mm, data_len)
                entry.refs = 1
                self._pool[key] = entry
                return entry
            except Exception:
                os.close(fd)
                raise

    def release(self, path: Path) -> None:
        """Decrement the reference count for *path*."""
        key = str(path)
        with self._lock:
            if key in self._pool:
                self._pool[key].refs = max(0, self._pool[key].refs - 1)

    def invalidate(self, path: Path) -> None:
        """Remove *path* from the pool and close its file descriptors."""
        key = str(path)
        with self._lock:
            entry = self._pool.pop(key, None)
        if entry is not None:
            self._close_entry(entry)

    def close_all(self) -> None:
        """Close all entries in the pool."""
        with self._lock:
            entries = list(self._pool.values())
            self._pool.clear()
        for entry in entries:
            self._close_entry(entry)

    def _evict_unreferenced(self) -> None:
        """Evict LRU entries with ref==0. Caller holds lock."""
        while len(self._pool) >= self._max:
            evicted = False
            for key, entry in self._pool.items():
                if entry.refs == 0:
                    del self._pool[key]
                    self._close_entry(entry)
                    evicted = True
                    break
            if not evicted:
                break

    @staticmethod
    def _close_entry(entry: _MmapEntry) -> None:
        """Close a single pool entry, suppressing all errors."""
        with contextlib.suppress(Exception):
            entry.mm.close()
        with contextlib.suppress(Exception):
            os.close(entry.fd)


class _HeartbeatWriter:
    """Background daemon that refreshes the heartbeat file mtime.

    [FIX-HB] File content is "pid:job_id" rather than just PID. OS PID
    recycling cannot cause a live unrelated process to be mistaken for a
    live dataloader heartbeat when both PID and job_id must match.
    """

    def __init__(self, hb_path: Path, job_id: str) -> None:
        self._path   = hb_path
        self._job_id = job_id
        self._stop   = threading.Event()
        self._write()
        self._thread = threading.Thread(
            target=self._run, name="shm-heartbeat", daemon=True,
        )
        self._thread.start()
        log.debug("HeartbeatWriter started: %s (pid=%d)", hb_path, os.getpid())

    def _write(self) -> None:
        """Write the heartbeat file atomically."""
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(f"{os.getpid()}:{self._job_id}", encoding="utf-8")
            tmp.rename(self._path)
        except Exception as exc:
            log.warning("HeartbeatWriter: could not write %s: %s", self._path, exc)

    def _run(self) -> None:
        """Heartbeat loop — runs until stop() is called."""
        while not self._stop.wait(timeout=_HB_INTERVAL_S):
            self._write()

    def stop(self) -> None:
        """Stop the heartbeat thread and remove the heartbeat file."""
        self._stop.set()
        self._thread.join(timeout=5)
        with contextlib.suppress(Exception):
            self._path.unlink(missing_ok=True)


def _purge_orphaned_shm(job_name: str, hb_stale_s: float = _HB_STALE_S) -> None:
    """Remove /dev/shm directories from dead jobs.

    [FIX-HB] Validates both PID liveness and job_id.
    """
    base = Path("/dev/shm")
    for d in base.iterdir():
        if not d.is_dir() or d.name == job_name:
            continue
        hb = d / _HB_FILENAME
        if hb.exists():
            try:
                mtime = hb.stat().st_mtime
                age   = time.time() - mtime
                if age < hb_stale_s:
                    continue
                content = hb.read_text(encoding="utf-8").strip()
                if ":" in content:
                    pid_str, hb_job_id = content.split(":", 1)
                    if hb_job_id == job_name:
                        continue
                else:
                    pid_str = content

                pid = int(pid_str)
                try:
                    os.kill(pid, 0)
                    continue
                except ProcessLookupError:
                    pass
            except Exception:
                pass

            log.info("Purging orphaned /dev/shm dir (stale heartbeat): %s", d)
            shutil.rmtree(d, ignore_errors=True)
        else:
            try:
                result = subprocess.run(
                    ["squeue", "--job", d.name, "--noheader"],
                    capture_output=True, timeout=_SQUEUE_TIMEOUT_S,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    log.info("Purging orphaned /dev/shm dir (no squeue entry): %s", d)
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


def _is_ready(shm: Path) -> bool:
    """Return True if the shard file is fully written and ready to read."""
    if not shm.exists():
        return False
    try:
        with open(shm, "rb") as f, mmap.mmap(f.fileno(), _HDR_SIZE, access=mmap.ACCESS_READ) as mm:
            _, magic = struct.unpack_from(_HDR_FMT, mm, 0)
            return magic == _READY_MAGIC
    except Exception:
        return False


def _check_shm_headroom(incoming: int) -> None:
    """Raise if the OS-reported free tmpfs space is dangerously low."""
    try:
        st   = os.statvfs("/dev/shm")
        free = st.f_bsize * st.f_bavail
    except Exception:
        return
    needed = incoming * _SHM_HEADROOM_WARN_FACTOR
    if free < max(needed, _SHM_HEADROOM_MIN_WARN_MB * (1 << 20)):
        msg = (
            f"/dev/shm has only {free >> 20} MB free; shard write of "
            f"{incoming >> 20} MB would exceed available space. "
            "Reduce node_shm_gb or shard_prefetch_window."
        )
        raise RuntimeError(msg)


def _read_file_sync(path: str) -> bytes:
    """Synchronous fallback for shard reads when aiofiles is unavailable.

    [FIX-READ] Utilise pathlib.Path.read_bytes() pour une lecture correcte sur
    tous les filesystems.  L'ancienne version utilisait os.read(fd, st_size)
    qui n'est pas garanti de lire tous les octets en un seul appel sur NFS/Lustre
    avec certaines options de montage.
    """
    try:
        return Path(path).read_bytes()
    except Exception as exc:
        raise RuntimeError(f"Failed to read shard {path}: {exc}") from exc


def _inotify_wait(shm: Path, timeout_s: float) -> None:
    """Block until shm is ready, using inotify on Linux or stat-poll elsewhere.

    [FS-2] Détection automatique de la disponibilité d'inotify.
    """
    global _INOTIFY_AVAILABLE  # noqa: PLW0603

    deadline = time.monotonic() + timeout_s

    if _INOTIFY_AVAILABLE:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            ifd  = libc.inotify_init1(0o4000)  # IN_NONBLOCK
            if ifd < 0:
                errno = ctypes.get_errno()
                if errno == 38:  # ENOSYS — inotify not supported (e.g. CephFS FUSE)
                    _INOTIFY_AVAILABLE = False
                    log.warning(
                        "inotify_init1 returned ENOSYS — filesystem does not support "
                        "inotify (possibly CephFS FUSE mount). "
                        "Falling back to stat-polling. "
                        "For optimal performance, use the CephFS kernel client (ceph.ko) "
                        "instead of ceph-fuse for the shard data path.",
                    )
                else:
                    raise OSError(f"inotify_init1 failed (errno={errno})")
            else:
                wd = libc.inotify_add_watch(
                    ifd,
                    str(shm.parent).encode(),
                    _IN_CLOSE_WRITE | _IN_MOVED_TO,
                )
                if wd < 0:
                    os.close(ifd)
                    raise OSError("inotify_add_watch failed")
                try:
                    while not _is_ready(shm):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                f"Timed out ({timeout_s:.0f}s) waiting for shard: {shm}",
                            )
                        r, _, _ = select.select([ifd], [], [], min(remaining, 1.0))
                        if r:
                            os.read(ifd, 4096)
                finally:
                    libc.inotify_rm_watch(ifd, wd)
                    os.close(ifd)
                return
        except (TimeoutError, OSError):
            raise
        except Exception:
            pass

    # Stat-poll fallback (CephFS FUSE, NFS sans inotify, autres).
    while not _is_ready(shm):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out ({timeout_s:.0f}s) waiting for shard: {shm}",
            )
        time.sleep(0.05)


async def _read_shard_async(shard_path: str) -> bytes:
    """Lire un shard depuis le filesystem de manière asynchrone.

    Supporte Lustre, CephFS (kernel ou FUSE), et tout filesystem POSIX.

    Args:
        shard_path: Chemin absolu vers le fichier .tar du shard.

    Returns:
        Contenu brut du shard en bytes.

    Raises:
        RuntimeError: En cas d'échec de lecture.

    """
    try:
        import aiofiles  # noqa: PLC0415
        async with aiofiles.open(shard_path, "rb") as f:
            return await f.read()
    except ImportError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read_file_sync, shard_path)


class NodeSharedShardCache:
    """Node-local /dev/shm shard cache.

    Corrections
    -----------
    [FIX-EVICT-EARLY] _load_one lève immédiatement si le shard dépasse le
        budget total, sans attendre _EVICT_RETRIES itérations.
    [FIX-SIGNAL] _register_signals chaîne les handlers existants (PyTorch
        distributed / NCCL) pour éviter les deadlocks NCCL à la terminaison.
    [FIX-STALE] heartbeat_stale_s est maintenant transmis à _init_shm, qui
        l'utilise lors de l'appel à _purge_orphaned_shm au lieu d'utiliser
        la constante globale _HB_STALE_S.

    Args:
        node_master: True for local rank 0 — this process fills the cache.
        job_id: Namespace for /dev/shm files (use SLURM_JOB_ID).
        max_shm_gb: RAM budget in /dev/shm for this node.
        prefetch_window: Max concurrent filesystem → /dev/shm downloads.
        shard_timeout_s: How long non-master ranks wait for a shard.
        shm_warn_threshold: Fraction (0–1) at which to emit a utilisation warning.
        heartbeat_stale_s: Seconds of no heartbeat before a dir is orphaned.

    """

    def __init__(
        self,
        node_master:        bool,
        job_id:             str   = "dino",
        max_shm_gb:         float = 128.0,
        prefetch_window:    int   = 64,
        shard_timeout_s:    float = 300.0,
        shm_warn_threshold: float = 0.85,
        heartbeat_stale_s:  float = _HB_STALE_S,
    ) -> None:
        self._node_master      = node_master
        self._job_id           = job_id
        self._max_bytes        = int(max_shm_gb * (1 << 30))
        self._base             = Path(f"/dev/shm/{job_id}")
        self._timeout          = shard_timeout_s
        self._warn_threshold   = shm_warn_threshold
        self._heartbeat_stale  = heartbeat_stale_s
        self._last_warn_ts:    float = 0.0

        self._lru:         OrderedDict[str, int] = OrderedDict()
        self._total_bytes: int                   = 0
        self._lru_lock:    threading.Lock        = threading.Lock()
        self._in_flight:   set[str]              = set()
        self._shutdown_event = threading.Event()

        self._mmap_pool = _MmapPool(max_entries=_MMAP_POOL_MAX)

        if node_master:
            # [FIX-STALE] Pass heartbeat_stale_s through to _init_shm.
            self._init_shm(heartbeat_stale_s)
            self._metrics = get_registry()
            self._loop    = asyncio.new_event_loop()
            self._sem     = asyncio.Semaphore(prefetch_window)
            self._thread  = threading.Thread(
                target=self._loop.run_forever, name="shard-io", daemon=True,
            )
            self._thread.start()
            self._heartbeat: _HeartbeatWriter | None = _HeartbeatWriter(
                self._base / _HB_FILENAME, job_id=job_id,
            )
            atexit.register(self._cleanup)
            self._register_signals()
        else:
            self._base.mkdir(parents=True, exist_ok=True)
            self._metrics   = get_registry()
            self._heartbeat = None

    def prefetch(self, shard_path: str) -> None:
        """Schedule a shard for background loading (node master only)."""
        if not self._node_master:
            return
        shm = self._shm_path(shard_path)
        with self._lru_lock:
            if _is_ready(shm) or shard_path in self._in_flight:
                return
            self._in_flight.add(shard_path)
        asyncio.run_coroutine_threadsafe(
            self._load_one(shard_path, shm), self._loop,
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
                    self._load_one(shard_path, shm), self._loop,
                ).result()
            return self._read(shm)
        t_wait  = time.perf_counter()
        _inotify_wait(shm, self._timeout)
        wait_ms = int((time.perf_counter() - t_wait) * 1000)
        if self._metrics is not None and wait_ms > 0:
            self._metrics.inc(MetricField.SHARD_CACHE_WAIT_MS, wait_ms)
        return self._read(shm)

    @contextlib.contextmanager
    def get_view(self, shard_path: str) -> Iterator[memoryview]:
        """Yield a zero-copy memoryview into the shard."""
        shm = self._shm_path(shard_path)
        if self._node_master:
            if not _is_ready(shm):
                with self._lru_lock:
                    if shard_path not in self._in_flight:
                        self._in_flight.add(shard_path)
                asyncio.run_coroutine_threadsafe(
                    self._load_one(shard_path, shm), self._loop,
                ).result()
        else:
            t_wait  = time.perf_counter()
            _inotify_wait(shm, self._timeout)
            wait_ms = int((time.perf_counter() - t_wait) * 1000)
            if self._metrics is not None and wait_ms > 0:
                self._metrics.inc(MetricField.SHARD_CACHE_WAIT_MS, wait_ms)

        entry = self._mmap_pool.acquire(shm)
        try:
            yield memoryview(entry.mm)[_HDR_SIZE: _HDR_SIZE + entry.data_len]
        finally:
            self._mmap_pool.release(shm)

    @property
    def utilisation(self) -> float:
        """Current cache utilisation as a fraction in [0, 1]."""
        if self._max_bytes == 0:
            return 0.0
        with self._lru_lock:
            return self._total_bytes / self._max_bytes

    def _shm_path(self, shard_path: str) -> Path:
        """Derive the /dev/shm path for a given shard path."""
        digest = hashlib.sha1(shard_path.encode()).hexdigest()[:16]
        return self._base / digest

    async def _load_one(self, shard_path: str, shm: Path) -> None:
        """Fetch one shard from the filesystem, write to /dev/shm.

        [FIX-EVICT-EARLY] Si le shard dépasse structurellement le budget total,
        on lève immédiatement sans attendre les _EVICT_RETRIES itérations.
        """
        async with self._sem:
            try:
                t0   = time.perf_counter()
                data = await _read_shard_async(shard_path)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)

                if self._metrics is not None:
                    self._metrics.inc(MetricField.LUSTRE_READ_TIME_MS, elapsed_ms)
                    self._metrics.inc(MetricField.LUSTRE_BYTES_READ, len(data))

                # [FIX-EVICT-EARLY] Early exit si le shard lui-même dépasse
                # le budget total — aucune éviction ne pourrait faire de la place.
                if len(data) > self._max_bytes:
                    msg = (
                        f"NodeSharedShardCache: shard {shard_path!r} "
                        f"({len(data) >> 20} MB) exceeds the entire shm budget "
                        f"({self._max_bytes >> 30} GB). Increase node_shm_gb."
                    )
                    raise RuntimeError(msg)

                for attempt in range(_EVICT_RETRIES):
                    with self._lru_lock:
                        if self._total_bytes + len(data) <= self._max_bytes:
                            break
                        self._evict_for_locked(len(data))
                        if self._total_bytes + len(data) <= self._max_bytes:
                            break
                    if attempt < _EVICT_RETRIES - 1:
                        await asyncio.sleep(_EVICT_WAIT_S)
                    else:
                        msg = (
                            f"NodeSharedShardCache: could not evict enough space "
                            f"for shard {shard_path!r} after {_EVICT_RETRIES} retries "
                            f"({_EVICT_RETRIES * _EVICT_WAIT_S:.0f}s). "
                            "All mmap slots are referenced simultaneously — "
                            "reduce shard_prefetch_window or increase node_shm_gb."
                        )
                        raise RuntimeError(msg)

                with self._lru_lock:
                    _check_shm_headroom(len(data))
                    self._write(shm, data)
                    self._lru[shard_path]  = len(data)
                    self._total_bytes     += len(data)

                self._update_utilisation_metric()
                log.debug("Shard cached: %s (%d MB)", shard_path, len(data) >> 20)
            finally:
                with self._lru_lock:
                    self._in_flight.discard(shard_path)

    @staticmethod
    def _write(shm: Path, data: bytes) -> None:
        """Write shard bytes to /dev/shm atomically (no fsync — PERF-1)."""
        tmp = shm.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack(_HDR_FMT, len(data), 0))
                f.write(data)
                f.seek(0)
                f.write(struct.pack(_HDR_FMT, len(data), _READY_MAGIC))
            tmp.rename(shm)
        except Exception:
            with contextlib.suppress(Exception):
                tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read(shm: Path) -> bytes:
        """Read shard bytes from a /dev/shm backing file."""
        with open(shm, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data_len, magic = struct.unpack_from(_HDR_FMT, mm, 0)
            if magic != _READY_MAGIC:
                raise RuntimeError(f"Shard {shm} has corrupt header")
            return bytes(mm[_HDR_SIZE: _HDR_SIZE + data_len])

    def _evict_for_locked(self, incoming: int) -> None:
        """Evict LRU shards to make room. Caller must hold _lru_lock.

        [FIX-EVICT] _total_bytes is only decremented after a successful unlink.
        """
        while self._total_bytes + incoming > self._max_bytes and self._lru:
            path_str, sz = self._lru.popitem(last=False)
            p = Path(path_str)
            self._mmap_pool.invalidate(p)
            try:
                p.unlink(missing_ok=True)
                p.with_suffix(".tmp").unlink(missing_ok=True)
                self._total_bytes -= sz
            except Exception as exc:
                log.warning(
                    "Eviction failed for %s: %s — bytes NOT decremented to "
                    "preserve accounting integrity.",
                    path_str, exc,
                )

    def _update_utilisation_metric(self) -> None:
        """Update the shard cache utilisation metric and emit a warning if high."""
        util = self.utilisation
        if self._metrics is not None:
            self._metrics.set_float(MetricField.SHARD_CACHE_UTIL_PCT, util * 100.0)
        if util >= self._warn_threshold:
            now = time.monotonic()
            if now - self._last_warn_ts >= _SHM_WARN_INTERVAL:
                self._last_warn_ts = now
                log.warning(
                    "/dev/shm utilisation is %.1f%% (threshold %.0f%%). "
                    "Increase node_shm_gb or reduce shard_prefetch_window. "
                    "Budget: %.1f GB, used: %.1f GB.",
                    util * 100.0, self._warn_threshold * 100.0,
                    self._max_bytes / (1 << 30), self._total_bytes / (1 << 30),
                )

    def _init_shm(self, heartbeat_stale_s: float = _HB_STALE_S) -> None:
        """Initialise the /dev/shm directory for this job.

        [FIX-STALE] Uses the provided heartbeat_stale_s instead of the
        global constant, so the constructor parameter is actually honoured.
        """
        _purge_orphaned_shm(self._base.name, hb_stale_s=heartbeat_stale_s)
        if self._base.exists():
            log.info("Removing stale shard cache at %s", self._base)
            shutil.rmtree(self._base, ignore_errors=True)
        self._base.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _register_signals(self) -> None:
        """Register SIGTERM/SIGINT handlers to trigger a clean shutdown.

        [FIX-SIGNAL] Chaîne les handlers existants (PyTorch distributed / NCCL)
        plutôt que de les écraser.  Écraser SIGTERM quand torch.distributed est
        déjà initialisé provoque des deadlocks NCCL lors de la terminaison du job.

        Ce handler n'est enregistré que sur le node master (local_rank == 0),
        le seul processus qui possède un cache /dev/shm à nettoyer.
        """
        def _make_handler(
            old_handler: Any,
        ) -> Any:
            """Retourne un handler qui appelle l'ancien handler après nettoyage."""
            def _handler(signum: int, frame: Any) -> None:
                self._shutdown_event.set()
                # Chaîner l'ancien handler s'il était un callable Python.
                if callable(old_handler):
                    old_handler(signum, frame)
                elif old_handler == signal.SIG_DFL:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)
                # SIG_IGN : on ignore (rare pour SIGTERM, impossible pour SIGINT).
            return _handler

        for sig in (signal.SIGTERM, signal.SIGINT):
            old = signal.getsignal(sig)
            try:
                signal.signal(sig, _make_handler(old))
            except (OSError, ValueError) as exc:
                # Peut échouer si on n'est pas dans le thread principal.
                log.debug("Could not register signal %s handler: %s", sig, exc)

    def _cleanup(self) -> None:
        """atexit: stop heartbeat, remove /dev/shm cache, close pools."""
        if self._heartbeat is not None:
            self._heartbeat.stop()
        self._mmap_pool.close_all()
        with contextlib.suppress(Exception):
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._base.exists():
            shutil.rmtree(self._base, ignore_errors=True)
        log.info("NodeSharedShardCache cleaned up")