"""
dino_loader.monitor.cli
=======================
Real-time terminal UI for Dataloader monitoring (htop style).

Usage
-----
::

    python -m dino_loader.monitor.cli --job $SLURM_JOB_ID

The monitor connects read-only to the ``/dev/shm`` metrics block written by
the dataloader and refreshes at ~4 Hz.  It never acquires any lock and
tolerates torn reads (display jitter is acceptable).

Fixes applied
-------------
[FIX-CLI-1] BUG-D: "Net Stall (ms)" column was reading ``lustre_read_time_ms``
            instead of ``network_stall_time_ms``.  Now correct.
[FIX-CLI-2] ``queue_table`` was built but never inserted into the layout;
            the panel was instead filled from an f-string.  Unified to use
            the Rich Table so all globals display consistently.
[FIX-CLI-3] ``time.time()`` replaced with ``time.monotonic()`` for rate
            computation.  ``time.time()`` can jump backward (NTP slew) and
            yield negative or infinite rates.
[FIX-CLI-4] Added staleness detection via ``heartbeat_ts``.  If a rank's
            last heartbeat is older than STALE_THRESHOLD_S, its row is
            dimmed and marked "[stale]" so operators can distinguish a
            healthy idle rank from a hung/dead process.
[FIX-CLI-5] Renamed column header "Mutex Wait (ms)" → "Cache Wait (ms)" to
            match the renamed ``shard_cache_wait_time_ms`` field.
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    from rich.columns import Columns
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .metrics import MetricsRegistry, MAX_LOCAL_RANKS

# A rank whose heartbeat is older than this is considered stale / dead.
STALE_THRESHOLD_S: int = 10


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_bytes(b: float) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"


def _bar(value: float, total: float, width: int = 24) -> str:
    """ASCII progress bar.  ``value`` and ``total`` must be in the same unit."""
    if total <= 0:
        return "▒" * width
    ratio  = min(max(value / total, 0.0), 1.0)
    filled = int(ratio * width)
    return "█" * filled + "▒" * (width - filled)


def _is_stale(heartbeat_ts: int, now: float) -> bool:
    return heartbeat_ts > 0 and (now - heartbeat_ts) > STALE_THRESHOLD_S


def _is_empty(m) -> bool:
    """True if this rank slot has never been written."""
    return m.loader_batches_yielded == 0 and m.lustre_bytes_read == 0 and m.heartbeat_ts == 0


# ── Main monitor loop ─────────────────────────────────────────────────────────

def run_monitor(job_id: str) -> None:
    if not HAS_RICH:
        print("The monitor requires the `rich` library.  Install with:  pip install rich")
        sys.exit(1)

    registry = MetricsRegistry(job_id=job_id, create=False)
    if registry.shm is None:
        print(
            f"[ERROR] Could not connect to shared memory for job '{job_id}'.\n"
            "        Is the dataloader running on this node?"
        )
        sys.exit(1)

    # Per-rank rate-tracking state (monotonic timestamps + last counters).
    last_mono        = time.monotonic()
    last_lustre_bytes = [0] * MAX_LOCAL_RANKS
    last_batches      = [0] * MAX_LOCAL_RANKS

    try:
        with Live(refresh_per_second=4, screen=True) as live:
            while True:
                now_mono = time.monotonic()                        # [FIX-CLI-3]
                now_wall = time.time()
                dt       = max(now_mono - last_mono, 1e-3)        # guard div-by-zero

                data = registry.read_all_ranks()
                if data is None:
                    break

                # ── Layout skeleton ──────────────────────────────────────────
                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="globals", size=7),
                    Layout(name="ranks"),
                )

                # ── Header ───────────────────────────────────────────────────
                layout["header"].update(
                    Panel(
                        Text(
                            f"DINOv3 Dataloader Monitor  ·  Job: {job_id}  ·  "
                            f"{time.strftime('%H:%M:%S')}",
                            style="bold white on blue",
                            justify="center",
                        )
                    )
                )

                # ── Global view (rank 0 aggregates Lustre + cache) ───────────
                # [FIX-CLI-2] Use a proper Rich Table, not a dead local variable.
                m0            = data.ranks[0]
                lustre_rate   = (m0.lustre_bytes_read - last_lustre_bytes[0]) / dt
                batches_rate  = (m0.loader_batches_yielded - last_batches[0]) / dt
                shm_util_pct  = m0.shard_cache_utilization_pct

                globals_table = Table.grid(padding=(0, 4))
                globals_table.add_column(style="bold cyan",  justify="right")
                globals_table.add_column(style="white",      justify="left")
                globals_table.add_row("Lustre I/O",  f"{_fmt_bytes(lustre_rate)}/s")
                globals_table.add_row(
                    "Shard Cache",
                    f"{_bar(shm_util_pct, 100.0)}  {shm_util_pct:.1f}%",
                )
                globals_table.add_row(
                    "Throughput",
                    f"{batches_rate:.2f} batches/s",
                )
                layout["globals"].update(
                    Panel(globals_table, title="[bold]Node Pipeline  (Rank 0 view)[/bold]")
                )

                # ── Per-rank table ───────────────────────────────────────────
                ranks_table = Table(expand=True, show_lines=False)
                ranks_table.add_column("Rank",            justify="center",  style="cyan",    width=6)
                ranks_table.add_column("Batches",         justify="right",   style="blue")
                ranks_table.add_column("Net Stall (ms)",  justify="right",   style="red")    # [FIX-CLI-1]
                ranks_table.add_column("Cache Wait (ms)", justify="right",   style="yellow") # [FIX-CLI-5]
                ranks_table.add_column("Pipe Yield (ms)", justify="right",   style="magenta")
                ranks_table.add_column("H2D (ms)",        justify="right",   style="green")
                ranks_table.add_column("Status",          justify="center",  style="white")

                for i in range(MAX_LOCAL_RANKS):
                    m = data.ranks[i]
                    if _is_empty(m):
                        continue

                    stale  = _is_stale(m.heartbeat_ts, now_wall)   # [FIX-CLI-4]
                    status = "[dim]stale[/dim]" if stale else "[green]●[/green]"
                    style  = "dim" if stale else ""

                    ranks_table.add_row(
                        f"[{style}]{i}[/{style}]"         if style else str(i),
                        f"[{style}]{m.loader_batches_yielded}[/{style}]"    if style else str(m.loader_batches_yielded),
                        f"[{style}]{m.network_stall_time_ms}[/{style}]"     if style else str(m.network_stall_time_ms),   # [FIX-CLI-1]
                        f"[{style}]{m.shard_cache_wait_time_ms}[/{style}]"  if style else str(m.shard_cache_wait_time_ms),
                        f"[{style}]{m.pipeline_yield_time_ms}[/{style}]"    if style else str(m.pipeline_yield_time_ms),
                        f"[{style}]{m.h2d_transfer_time_ms}[/{style}]"      if style else str(m.h2d_transfer_time_ms),
                        status,
                    )

                layout["ranks"].update(
                    Panel(ranks_table, title="[bold]Per-GPU Workers[/bold]")
                )

                live.update(layout)

                # ── Update rate-tracking state ───────────────────────────────
                for i in range(MAX_LOCAL_RANKS):
                    last_lustre_bytes[i] = data.ranks[i].lustre_bytes_read
                    last_batches[i]      = data.ranks[i].loader_batches_yielded
                last_mono = now_mono

                time.sleep(0.25)

    except KeyboardInterrupt:
        pass


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time DINOv3 Dataloader Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--job",
        type=str,
        default="dino",
        help="Job ID to monitor (same value passed as job_id= to DINODataLoader)",
    )
    args = parser.parse_args()
    run_monitor(args.job)


if __name__ == "__main__":
    main()
