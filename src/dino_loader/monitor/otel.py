"""dino_loader.monitor.otel
========================
Unified observability layer for dino_loader.

Goals
-----
1. **Structured logging** — rank / epoch / step are propagated automatically
   through Python's ``contextvars`` so that every log line emitted from any
   thread carries the right labels without explicit parameter threading.

2. **OpenTelemetry spans** — each of the 5 pipeline stages is wrapped in an
   OTEL span. Traces can be exported to any OTEL-compatible backend.
   When ``opentelemetry-sdk`` is absent the layer degrades gracefully to a
   zero-cost no-op.

3. **Chrome trace compatibility** — ``tracing.py`` is preserved for backward
   compatibility, but ``StageTracer`` also records to it when both systems
   are active.

Corrections
-----------
[FIX-OTEL-LOCK] _TRACER et _TRACER_INITIALISED sont maintenant protégés par
    un threading.Lock pour éviter la double-initialisation si deux threads
    appellent init_otel() simultanément.
[FIX-FUTURE] from __future__ import annotations supprimé (Python ≥ 3.12 natif).
"""

import contextvars
import logging
import threading
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


_ctx_var: contextvars.ContextVar[_LoaderCtx] = contextvars.ContextVar(
    "dino_loader_context",
    default=_LoaderCtx(),
)


class LoaderContext:
    """Read-only view of the current rank / epoch / step.

    Thread-safe by construction: ``ContextVar`` reads are always consistent
    within a given execution context.

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
        """Update the context in the current thread / coroutine."""
        return _ctx_var.set(_LoaderCtx(rank=rank, epoch=epoch, step=step))

    @staticmethod
    def reset(token: contextvars.Token) -> None:
        """Undo a previous ``set()`` call (test teardown helper)."""
        _ctx_var.reset(token)


def set_loader_context(rank: int, epoch: int, step: int) -> None:
    """Convenience wrapper — update the loader context from the training loop."""
    LoaderContext.set(rank=rank, epoch=epoch, step=step)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DINOLoggingFilter — automatic label injection
# ══════════════════════════════════════════════════════════════════════════════

class DINOLoggingFilter(logging.Filter):
    """Logging filter that injects ``rank``, ``epoch``, and ``step`` into every
    ``LogRecord`` produced by a logger that has this filter installed.

    After installing, use ``%(rank)s``, ``%(epoch)s``, ``%(step)s`` in your
    format string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject context fields into the log record."""
        ctx = LoaderContext.get()
        record.rank  = ctx.rank
        record.epoch = ctx.epoch
        record.step  = ctx.step
        return True


def install_logging_filter(
    logger: logging.Logger | None = None,
    fmt: str = (
        "%(asctime)s %(levelname)-8s "
        "[rank=%(rank)s ep=%(epoch)s st=%(step)s] "
        "%(name)s %(message)s"
    ),
) -> logging.Filter:
    """Attach a :class:`DINOLoggingFilter` to *logger* (default: root logger).

    Parameters
    ----------
    logger
        Logger to patch.  Defaults to the root logger.
    fmt
        Log format string.

    Returns
    -------
    DINOLoggingFilter
        The installed filter instance (useful for testing / removal).

    """
    if logger is None:
        logger = logging.getLogger()

    f = DINOLoggingFilter()
    logger.addFilter(f)

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
    """Build and register an OTEL ``TracerProvider`` from environment variables."""
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


# [FIX-OTEL-LOCK] Lock protégeant le singleton OTEL contre la double-init.
# Sans ce lock, deux threads appelant init_otel() simultanément pourraient
# créer deux TracerProvider distincts, dont seul le second serait enregistré.
_TRACER_LOCK: threading.Lock = threading.Lock()
_TRACER: object = None
_TRACER_INITIALISED: bool = False


def init_otel(service_name: str = "dino_loader") -> None:
    """Initialise the OTEL tracer.  Safe to call multiple times (idempotent).

    Thread-safe: utilise un lock pour éviter la double-initialisation si
    deux threads appellent init_otel() simultanément (ex. ranks différents
    dans un job multi-GPU partagé).

    Parameters
    ----------
    service_name
        OTEL ``service.name`` resource attribute.

    """
    global _TRACER, _TRACER_INITIALISED  # noqa: PLW0603
    # Double-checked locking pour minimiser le coût sur le chemin rapide.
    if _TRACER_INITIALISED:
        return
    with _TRACER_LOCK:
        if _TRACER_INITIALISED:
            return
        _TRACER = _build_tracer(service_name)
        _TRACER_INITIALISED = True


# ══════════════════════════════════════════════════════════════════════════════
# 4. StageTracer — unified span + Chrome trace + metrics
# ══════════════════════════════════════════════════════════════════════════════

_STAGE_METRIC: dict[str, str] = {
    "lustre_io":       "lustre_read_time_ms",
    "shard_wait":      "shard_cache_wait_time_ms",
    "dali_pipeline":   "pipeline_yield_time_ms",
    "h2d_transfer":    "h2d_transfer_time_ms",
    "fp8_quant":       "pipeline_yield_time_ms",
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
    1. Starts an OTEL span with ``name`` and the current context.
    2. Records a Chrome trace event via ``tracing.py``.
    3. On exit, updates the corresponding ``MetricField`` counter.

    Parameters
    ----------
    name
        Stage identifier.  Must be one of the keys in ``_STAGE_METRIC``
        for metrics to be updated.
    attributes
        Optional dict of extra OTEL span attributes.

    """
    ctx = LoaderContext.get()

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

    try:
        from dino_loader.monitor import tracing as _tracing  # noqa: PLC0415
        _chrome_enabled = _tracing._GLOBAL_TRACER.enabled
    except Exception:
        _chrome_enabled = False

    t_start_ns = time.perf_counter_ns()

    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - t_start_ns
        elapsed_ms = elapsed_ns // 1_000_000

        if otel_span is not None:
            try:
                otel_span.set_attribute("elapsed_ms", elapsed_ms)
                otel_span.end()
            except Exception:
                pass

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

        metric_field = _STAGE_METRIC.get(name)
        if metric_field:
            try:
                from dino_loader.monitor.metrics import get_registry  # noqa: PLC0415
                reg = get_registry()
                if reg is not None:
                    reg.inc(metric_field, int(elapsed_ms))
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# 5. StageTimer — non-context-manager version for async paths
# ══════════════════════════════════════════════════════════════════════════════

class StageTimer:
    """Manual start/stop timer for code paths where a context manager is
    inconvenient (e.g. async callbacks, or stages split across methods).

    Example::

        timer = StageTimer("shard_wait")
        timer.start()
        shard = await shard_cache.get_async(path)
        elapsed_ms = timer.stop()
    """

    __slots__ = ("_attributes", "_name", "_t_start_ns")

    def __init__(self, name: str, attributes: dict | None = None) -> None:
        self._name       = name
        self._attributes = attributes
        self._t_start_ns: int | None = None

    def start(self) -> "StageTimer":
        """Start the timer."""
        self._t_start_ns = time.perf_counter_ns()
        return self

    def stop(self) -> int:
        """Stop the timer, record metrics / traces, and return elapsed ms."""
        if self._t_start_ns is None:
            log.warning("StageTimer('%s').stop() called without start()", self._name)
            return 0

        elapsed_ns = time.perf_counter_ns() - self._t_start_ns
        elapsed_ms = elapsed_ns // 1_000_000
        self._t_start_ns = None

        ctx = LoaderContext.get()

        try:
            from dino_loader.monitor import tracing as _tracing  # noqa: PLC0415
            if _tracing._GLOBAL_TRACER.enabled:
                _tracing._GLOBAL_TRACER.record(
                    name     = f"{self._name}[rank={ctx.rank}]",
                    cat      = "dino_loader",
                    start_us = (time.perf_counter_ns() - elapsed_ns) // 1_000,
                    dur_us   = elapsed_ns // 1_000,
                )
        except Exception:
            pass

        metric_field = _STAGE_METRIC.get(self._name)
        if metric_field:
            try:
                from dino_loader.monitor.metrics import get_registry  # noqa: PLC0415
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
    "DINOLoggingFilter",
    "LoaderContext",
    "StageTimer",
    "init_otel",
    "install_logging_filter",
    "set_loader_context",
    "stage",
]
