"""dino_loader.monitor.otel
========================
Unified observability layer for dino_loader.

Goals
-----
1. **Structured logging** — rank / epoch / step are propagated automatically
   through Python's ``contextvars`` so that every log line emitted from any
   thread (extraction workers, H2D threads, …) carries the right labels
   without explicit parameter threading.

2. **OpenTelemetry spans** — each of the 5 pipeline stages is wrapped in an
   OTEL span so that wall-clock time and causal relationships are captured.
   Traces can be exported to any OTEL-compatible backend (Jaeger, Tempo,
   Prometheus OTLP, …).  When the ``opentelemetry-sdk`` package is absent the
   entire layer degrades gracefully to a zero-cost no-op — not a single extra
   branch on the critical path.

3. **Chrome trace compatibility** — ``tracing.py`` is preserved for backward
   compatibility, but ``StageTracer`` (below) also records to it when both
   systems are active, so users need only one code path.

Architecture
------------
::

    ┌─────────────────────────────────────────────────────┐
    │  LoaderContext (module-level ContextVar)             │
    │  rank, epoch, step — propagated to all threads       │
    └──────────┬──────────────────────────────────────────┘
               │ read by
    ┌──────────▼──────────────────────────────────────────┐
    │  DINOLoggingFilter                                   │
    │  Injects rank/epoch/step into every LogRecord        │
    └──────────┬──────────────────────────────────────────┘
               │ used by
    ┌──────────▼──────────────────────────────────────────┐
    │  StageTracer                                         │
    │  Context manager — wraps a pipeline stage name       │
    │  • Starts / ends an OTEL span                        │
    │  • Records to tracing.py Chrome events               │
    │  • Updates MetricField counters                       │
    └─────────────────────────────────────────────────────┘

Public API
----------
::

    from dino_loader.monitor.otel import (
        LoaderContext,           # ContextVar holder
        set_loader_context,      # call once per epoch iteration
        install_logging_filter,  # call once at loader init
        StageTracer,             # context manager for pipeline stages
        stage,                   # convenience: stage("lustre_io")
    )

Usage in loader.py
------------------
::

    from dino_loader.monitor.otel import set_loader_context, stage

    set_loader_context(rank=env.rank, epoch=0, step=0)

    for step, dali_out in enumerate(self._dali_iter):
        set_loader_context(rank=env.rank, epoch=self._epoch, step=step)

        with stage("h2d_transfer"):
            batch = self._h2d.transfer(views)

        with stage("fp8_quant"):
            batch = self._fp8.quantise(batch)

        yield batch

OTEL exporter configuration
----------------------------
Set the standard OTEL env vars — dino_loader passes them through unchanged:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
    OTEL_SERVICE_NAME=dino_loader
    OTEL_RESOURCE_ATTRIBUTES=slurm_job_id=123456,node=node01

If these vars are absent the tracer runs in "noop" mode (no network I/O).

Implementation notes
--------------------
* ``contextvars.ContextVar`` propagation is automatic across
  ``ThreadPoolExecutor`` threads in Python ≥ 3.7 — extraction workers
  inherit the context of the thread that submitted them, so no explicit
  passing is needed.
* ``contextvars.copy_context()`` is used when spawning long-lived threads
  that must track context changes made after the thread starts (e.g. the
  epoch counter updated by ``set_epoch``).
* All public functions are safe to call before ``init_registry()`` —
  metrics updates are silently skipped if the registry is unavailable.
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LoaderContext — ContextVar holder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _LoaderCtx:
    """Immutable snapshot of the current loader context."""

    rank:  int = 0
    epoch: int = 0
    step:  int = 0


# Module-level ContextVar.  Frozen dataclasses are set atomically (replace the
# whole object), so there is no risk of a reader observing a half-updated state.
_ctx_var: contextvars.ContextVar[_LoaderCtx] = contextvars.ContextVar(
    "dino_loader_context",
    default=_LoaderCtx(),
)


class LoaderContext:
    """Read-only view of the current rank / epoch / step.

    Thread-safe by construction: ``ContextVar`` reads are always consistent
    within a given execution context, and writes from one thread do not
    bleed into another thread's context unless explicitly copied.

    Example::

        ctx = LoaderContext.get()
        print(f"rank={ctx.rank} epoch={ctx.epoch} step={ctx.step}")
    """

    @staticmethod
    def get() -> _LoaderCtx:
        """Return the current context snapshot (never raises)."""
        return _ctx_var.get()

    @staticmethod
    def set(rank: int, epoch: int, step: int) -> contextvars.Token:
        """Update the context in the current thread / coroutine.

        Returns a ``Token`` that can be passed to ``reset()`` to undo the
        change (useful in tests that need deterministic isolation).
        """
        return _ctx_var.set(_LoaderCtx(rank=rank, epoch=epoch, step=step))

    @staticmethod
    def reset(token: contextvars.Token) -> None:
        """Undo a previous ``set()`` call (test teardown helper)."""
        _ctx_var.reset(token)


def set_loader_context(rank: int, epoch: int, step: int) -> None:
    """Convenience wrapper — update the loader context from the training loop.

    Call this at the start of each step (inside ``_raw_iter``) so that every
    log line emitted anywhere in that step's call graph carries the correct
    rank / epoch / step labels — including lines from background threads that
    inherited this context when their futures were submitted.

    Example::

        for step, batch in enumerate(loader):
            set_loader_context(rank=env.rank, epoch=epoch, step=step)
            train_step(batch)
    """
    LoaderContext.set(rank=rank, epoch=epoch, step=step)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DINOLoggingFilter — automatic label injection
# ══════════════════════════════════════════════════════════════════════════════

class DINOLoggingFilter(logging.Filter):
    """Logging filter that injects ``rank``, ``epoch``, and ``step`` into every
    ``LogRecord`` produced by a logger that has this filter installed.

    After installing, use ``%(rank)s``, ``%(epoch)s``, ``%(step)s`` in your
    format string::

        %(asctime)s %(levelname)-8s [rank=%(rank)s ep=%(epoch)s st=%(step)s] %(name)s %(message)s

    Thread safety
    -------------
    Reading from a ContextVar is always consistent within the current
    execution context — no locking required.

    Installation
    ------------
    Use ``install_logging_filter()`` rather than instantiating directly.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = LoaderContext.get()
        record.rank  = ctx.rank
        record.epoch = ctx.epoch
        record.step  = ctx.step
        return True  # always pass the record through


def install_logging_filter(
    logger: logging.Logger | None = None,
    fmt: str = (
        "%(asctime)s %(levelname)-8s "
        "[rank=%(rank)s ep=%(epoch)s st=%(step)s] "
        "%(name)s %(message)s"
    ),
) -> logging.Filter:
    """Attach a :class:`DINOLoggingFilter` to *logger* (default: root logger).

    Also replaces the first ``StreamHandler`` found on the logger with one
    using *fmt* so that the rank/epoch/step fields are visible immediately.
    Existing non-stream handlers (file, syslog, …) are left untouched.

    Parameters
    ----------
    logger
        Logger to patch.  Defaults to the root logger so that all loggers in
        the process tree benefit automatically.
    fmt
        Log format string.  Must contain ``%(rank)s``, ``%(epoch)s``,
        ``%(step)s`` to show the injected fields.

    Returns
    -------
    DINOLoggingFilter
        The installed filter instance (useful for testing / removal).

    Example::

        install_logging_filter()
        logging.basicConfig(level=logging.INFO)
        log.info("starting")
        # → 14:32:01 INFO     [rank=0 ep=3 st=512] dino_loader.loader starting

    """
    if logger is None:
        logger = logging.getLogger()

    f = DINOLoggingFilter()
    logger.addFilter(f)

    # Patch the first StreamHandler we find so that the new fields show up.
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%H:%M:%S"))
            break

    return f


# ══════════════════════════════════════════════════════════════════════════════
# 3. OTEL tracer — graceful no-op when opentelemetry-sdk is absent
# ══════════════════════════════════════════════════════════════════════════════

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        _HAS_OTLP = True
    except ImportError:
        _HAS_OTLP = False

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    _HAS_OTLP = False


def _build_tracer(service_name: str = "dino_loader") -> object:
    """Build and register an OTEL ``TracerProvider`` from environment variables.

    Priority
    --------
    1. ``OTEL_EXPORTER_OTLP_ENDPOINT`` → gRPC OTLP exporter (Jaeger / Tempo).
    2. ``OTEL_CONSOLE_EXPORTER=1``      → stdout JSON (debugging).
    3. No env vars set               → SDK noop tracer (zero overhead).

    The provider is registered as the global OTEL provider so that any
    downstream code using ``opentelemetry.trace.get_tracer()`` participates
    in the same trace automatically.
    """
    if not _HAS_OTEL:
        return None

    import os
    provider = TracerProvider()

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint and _HAS_OTLP:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        log.info("OTEL: gRPC OTLP exporter → %s", otlp_endpoint)
    elif os.environ.get("OTEL_CONSOLE_EXPORTER"):
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        log.info("OTEL: console exporter enabled")
    else:
        log.debug("OTEL: no exporter configured — running in noop mode")

    _otel_trace.set_tracer_provider(provider)
    return _otel_trace.get_tracer(service_name)


# Lazy singleton — created on first call to ``init_otel()`` or ``stage()``.
_TRACER: object = None
_TRACER_INITIALISED = False


def init_otel(service_name: str = "dino_loader") -> None:
    """Initialise the OTEL tracer.  Safe to call multiple times (idempotent).

    Call once at loader construction time, before the first batch.  If
    ``opentelemetry-sdk`` is not installed, this is a silent no-op.

    Parameters
    ----------
    service_name
        OTEL ``service.name`` resource attribute (overrides the default
        ``"dino_loader"`` value).

    """
    global _TRACER, _TRACER_INITIALISED
    if _TRACER_INITIALISED:
        return
    _TRACER = _build_tracer(service_name)
    _TRACER_INITIALISED = True


# ══════════════════════════════════════════════════════════════════════════════
# 4. StageTracer — unified span + Chrome trace + metrics
# ══════════════════════════════════════════════════════════════════════════════

# Map of stage name → MetricField to increment (ms elapsed).
# Imported lazily to avoid a hard dependency cycle at module level.
_STAGE_METRIC: dict[str, str] = {
    "lustre_io":       "lustre_read_time_ms",
    "shard_wait":      "shard_cache_wait_time_ms",
    "dali_pipeline":   "pipeline_yield_time_ms",
    "h2d_transfer":    "h2d_transfer_time_ms",
    "fp8_quant":       "pipeline_yield_time_ms",   # folded into pipeline time
    "network_stall":   "network_stall_time_ms",
    "multinode_stall": "multinode_stall_time_ms",
}


@contextmanager
def stage(
    name: str,
    *,
    attributes: dict | None = None,
) -> Iterator[None]:
    """Context manager that instruments a named pipeline stage.

    What it does (all in one call)
    -------------------------------
    1. Starts an OTEL span with ``name`` and the current
       ``rank`` / ``epoch`` / ``step`` as span attributes.
    2. Records a Chrome trace event via ``tracing.py`` (zero-cost if tracing
       is not started).
    3. On exit, updates the corresponding ``MetricField`` counter with the
       elapsed milliseconds.

    Zero overhead guarantee
    -----------------------
    When OTEL is unavailable **and** Chrome tracing is not started, the entire
    body reduces to::

        t0 = time.perf_counter_ns()
        yield
        elapsed_ns = time.perf_counter_ns() - t0
        # → 2 syscalls total, no allocation, no lock

    Parameters
    ----------
    name
        Stage identifier.  Must be one of the keys in ``_STAGE_METRIC``
        for metrics to be updated; unknown names are silently ignored for
        metrics (but still traced).
    attributes
        Optional dict of extra OTEL span attributes (e.g. ``{"shard": path}``).

    Example::

        with stage("h2d_transfer"):
            gpu_batch = h2d.transfer(cpu_batch)

        with stage("lustre_io", attributes={"shard": shard_path}):
            data = lustre_read(shard_path)

    """
    # ── Retrieve context ──────────────────────────────────────────────────────
    ctx = LoaderContext.get()

    # ── OTEL span ─────────────────────────────────────────────────────────────
    otel_span = None
    if _HAS_OTEL and _TRACER is not None:
        span_attrs = {
            "rank":  ctx.rank,
            "epoch": ctx.epoch,
            "step":  ctx.step,
        }
        if attributes:
            span_attrs.update(attributes)
        otel_span = _TRACER.start_span(name, attributes=span_attrs)  # type: ignore[attr-defined]

    # ── Chrome tracing ────────────────────────────────────────────────────────
    # Import here to avoid module-level circular dependency.
    try:
        from dino_loader.monitor import tracing as _tracing
        _chrome_enabled = _tracing._GLOBAL_TRACER.enabled
    except Exception:
        _chrome_enabled = False

    t_start_ns = time.perf_counter_ns()

    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - t_start_ns
        elapsed_ms = elapsed_ns // 1_000_000

        # ── Finalise OTEL span ────────────────────────────────────────────────
        if otel_span is not None:
            try:
                otel_span.set_attribute("elapsed_ms", elapsed_ms)
                otel_span.end()
            except Exception:
                pass  # never let observability crash the training loop

        # ── Chrome trace event ────────────────────────────────────────────────
        if _chrome_enabled:
            try:
                _tracing._GLOBAL_TRACER.record(
                    name     = f"{name}[rank={ctx.rank}]",
                    cat      = "dino_loader",
                    start_us = t_start_ns // 1_000,
                    dur_us   = elapsed_ns  // 1_000,
                )
            except Exception:
                pass

        # ── Metrics update ────────────────────────────────────────────────────
        metric_field = _STAGE_METRIC.get(name)
        if metric_field:
            try:
                from dino_loader.monitor.metrics import get_registry
                reg = get_registry()
                if reg is not None:
                    reg.inc(metric_field, int(elapsed_ms))
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# 5. Convenience: stage_timer — a non-context-manager version for async paths
# ══════════════════════════════════════════════════════════════════════════════

class StageTimer:
    """Manual start/stop timer for code paths where a context manager is
    inconvenient (e.g. async callbacks, or stages split across methods).

    Example::

        timer = StageTimer("shard_wait")
        timer.start()
        shard = await shard_cache.get_async(path)
        elapsed_ms = timer.stop()   # records metrics + Chrome trace
    """

    __slots__ = ("_attributes", "_name", "_t_start_ns")

    def __init__(self, name: str, attributes: dict | None = None) -> None:
        self._name       = name
        self._attributes = attributes
        self._t_start_ns: int | None = None

    def start(self) -> StageTimer:
        self._t_start_ns = time.perf_counter_ns()
        return self

    def stop(self) -> int:
        """Stop the timer, record metrics / traces, and return elapsed ms.

        Calling ``stop()`` without a prior ``start()`` logs a warning and
        returns 0 — it never raises.
        """
        if self._t_start_ns is None:
            log.warning("StageTimer('%s').stop() called without start()", self._name)
            return 0

        elapsed_ns = time.perf_counter_ns() - self._t_start_ns
        elapsed_ms = elapsed_ns // 1_000_000
        self._t_start_ns = None

        ctx = LoaderContext.get()

        # Chrome trace
        try:
            from dino_loader.monitor import tracing as _tracing
            if _tracing._GLOBAL_TRACER.enabled:
                _tracing._GLOBAL_TRACER.record(
                    name     = f"{self._name}[rank={ctx.rank}]",
                    cat      = "dino_loader",
                    start_us = (time.perf_counter_ns() - elapsed_ns) // 1_000,
                    dur_us   = elapsed_ns // 1_000,
                )
        except Exception:
            pass

        # Metrics
        metric_field = _STAGE_METRIC.get(self._name)
        if metric_field:
            try:
                from dino_loader.monitor.metrics import get_registry
                reg = get_registry()
                if reg is not None:
                    reg.inc(metric_field, int(elapsed_ms))
            except Exception:
                pass

        return int(elapsed_ms)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Public re-export surface
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Context
    "LoaderContext",
    "set_loader_context",
    # Logging
    "DINOLoggingFilter",
    "install_logging_filter",
    # OTEL lifecycle
    "init_otel",
    # Tracing
    "stage",
    "StageTimer",
]
