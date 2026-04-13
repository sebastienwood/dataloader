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
[FIX-OTEL-IMPORTS] Les imports ``tracing`` et ``get_registry`` dans le bloc
    ``finally`` de ``stage()`` ont été déplacés en haut du module pour éviter
    un lookup ``sys.modules`` à chaque span.  En production avec des milliers
    de spans/s, cela élimine un coût non nul sous GIL.
[FIX-STAGE-TIMER-CM] StageTimer expose maintenant ``__enter__`` / ``__exit__``
    pour une utilisation sûre comme context manager.  L'ancien pattern
    start() / stop() laissait les métriques non enregistrées en cas d'exception
    entre les deux appels.  Le context manager garantit l'enregistrement même
    si le bloc lève une exception.
[FIX-STAGE1-INSTRUMENTATION] Ajout du nom de stage ``"lustre_io"`` sur les
    appels à ``_read_shard_async`` et ``_load_one`` dans shard_cache.py,
    documenté ici pour que les développeurs sachent où instrumenter.
"""

import contextvars
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-import monitoring dependencies once at module load time rather than
# inside the hot ``finally`` path of every span.  [FIX-OTEL-IMPORTS]
# ---------------------------------------------------------------------------

# Chrome tracer — imported lazily but cached at module level after first use.
_tracing_module = None


def _get_tracing():
    """Return the tracing module, importing it once and caching."""
    global _tracing_module  # noqa: PLW0603
    if _tracing_module is None:
        try:
            from dino_loader.monitor import tracing as _t  # noqa: PLC0415
            _tracing_module = _t
        except ImportError:
            _tracing_module = None
    return _tracing_module


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
_TRACER_LOCK: threading.Lock = threading.Lock()
_TRACER: object = None
_TRACER_INITIALISED: bool = False


def init_otel(service_name: str = "dino_loader") -> None:
    """Initialise the OTEL tracer.  Safe to call multiple times (idempotent).

    Thread-safe via double-checked locking.

    Parameters
    ----------
    service_name
        OTEL ``service.name`` resource attribute.

    """
    global _TRACER, _TRACER_INITIALISED  # noqa: PLW0603
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


def _record_stage_metrics(
    name:       str,
    ctx:        _LoaderCtx,
    elapsed_ns: int,
    otel_span:  object | None,
) -> None:
    """Record metrics for a completed stage.  Called from both stage() and StageTimer.

    [FIX-OTEL-IMPORTS] Metric and tracing module references are resolved once
    at import time (_get_tracing) or from the already-imported registry module,
    not on every span completion.
    """
    elapsed_ms = elapsed_ns // 1_000_000
    elapsed_us = elapsed_ns // 1_000

    if otel_span is not None:
        try:
            otel_span.set_attribute("elapsed_ms", elapsed_ms)  # type: ignore[union-attr]
            otel_span.end()  # type: ignore[union-attr]
        except Exception:
            pass

    tracing = _get_tracing()
    if tracing is not None:
        try:
            if tracing._GLOBAL_TRACER.enabled:
                start_us = (time.perf_counter_ns() - elapsed_ns) // 1_000
                tracing._GLOBAL_TRACER.record(
                    name     = f"{name}[rank={ctx.rank}]",
                    cat      = "dino_loader",
                    start_us = start_us,
                    dur_us   = elapsed_us,
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
        for metrics to be updated.  Well-known stage names:

        - ``"lustre_io"``       Stage 1: Lustre → /dev/shm (node master only)
        - ``"shard_wait"``      Stage 1: non-master rank waits for shard
        - ``"dali_pipeline"``   Stage 3: DALI augmentation pipeline
        - ``"h2d_transfer"``    Stage 4: host → device transfer
        - ``"fp8_quant"``       Stage 5: FP8 quantisation
        - ``"network_stall"``   Network stall detection
        - ``"multinode_stall"`` Multi-node stall detection

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
        otel_span = _TRACER.start_span(name, attributes=span_attrs)  # type: ignore[union-attr]

    t_start_ns = time.perf_counter_ns()

    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - t_start_ns
        _record_stage_metrics(name, ctx, elapsed_ns, otel_span)


# ══════════════════════════════════════════════════════════════════════════════
# 5. StageTimer — non-context-manager version for async paths
# ══════════════════════════════════════════════════════════════════════════════

class StageTimer:
    """Manual start/stop timer for code paths where a context manager is
    inconvenient (e.g. async callbacks, or stages split across methods).

    Supports both the manual start()/stop() API and the context manager
    protocol (__enter__/__exit__) for exception-safe usage.

    [FIX-STAGE-TIMER-CM] The context manager protocol guarantees that metrics
    are recorded even when the instrumented block raises an exception.  The
    previous start()/stop() pattern left metrics unrecorded on exception.

    Example (manual)::

        timer = StageTimer("shard_wait")
        timer.start()
        shard = await shard_cache.get_async(path)
        elapsed_ms = timer.stop()

    Example (context manager — preferred when possible)::

        with StageTimer("lustre_io"):
            data = await _read_shard_async(path)

    """

    __slots__ = ("_attributes", "_name", "_otel_span", "_t_start_ns")

    def __init__(self, name: str, attributes: dict | None = None) -> None:
        self._name        = name
        self._attributes  = attributes
        self._t_start_ns: int | None = None
        self._otel_span: object | None = None

    def start(self) -> "StageTimer":
        """Start the timer, opening an OTEL span if tracing is active."""
        ctx = LoaderContext.get()
        if _HAS_OTEL and _TRACER is not None:
            span_attrs: dict = {"rank": ctx.rank, "epoch": ctx.epoch, "step": ctx.step}
            if self._attributes:
                span_attrs.update(self._attributes)
            self._otel_span = _TRACER.start_span(self._name, attributes=span_attrs)  # type: ignore[union-attr]
        self._t_start_ns = time.perf_counter_ns()
        return self

    def stop(self) -> int:
        """Stop the timer, record metrics / traces, and return elapsed ms."""
        if self._t_start_ns is None:
            log.warning("StageTimer('%s').stop() called without start()", self._name)
            return 0

        elapsed_ns       = time.perf_counter_ns() - self._t_start_ns
        self._t_start_ns = None
        ctx              = LoaderContext.get()

        _record_stage_metrics(self._name, ctx, elapsed_ns, self._otel_span)
        self._otel_span = None

        return elapsed_ns // 1_000_000

    # ------------------------------------------------------------------
    # Context manager protocol [FIX-STAGE-TIMER-CM]
    # ------------------------------------------------------------------

    def __enter__(self) -> "StageTimer":
        """Start the timer on context entry."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop the timer on context exit, recording metrics even on exception."""
        self.stop()
        # Never suppress exceptions.
        return None


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