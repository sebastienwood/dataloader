"""dino_loader.monitor.cli
=======================
Real-time terminal UI for Dataloader monitoring (htop style).

Usage::

    python -m dino_loader.monitor.cli --job $SLURM_JOB_ID

The monitor connects read-only to the ``/dev/shm`` metrics block written by
the dataloader and refreshes at ~4 Hz.  It never acquires any lock and
tolerates torn reads (display jitter is acceptable).

Fixes applied
-------------
[FIX-CLI-1] "Net Stall (ms)" column now reads ``network_stall_time_ms``.
[FIX-CLI-2] ``queue_table`` now uses Rich Table consistently.
[FIX-CLI-3] ``time.monotonic()`` replaces ``time.time()`` for rate computation.
[FIX-CLI-4] Staleness detection via ``heartbeat_ts``.
[FIX-CLI-5] Column header renamed "Mutex Wait" → "Cache Wait".
"""

import argparse
import sys
import time

try:
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .metrics import MAX_LOCAL_RANKS, MetricsRegistry

STALE_THRESHOLD_S: int = 10


def _fmt_bytes(b: float) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"


def _bar(value: float, total: float, width: int = 24) -> str:
    """ASCII progress bar."""
    if total <= 0:
        return "▒" * width
    ratio  = min(max(value / total, 0.0), 1.0)
    filled = int(ratio * width)
    return "█" * filled + "▒" * (width - filled)


def _is_stale(heartbeat_ts: int, now: float) -> bool:
    return heartbeat_ts > 0 and (now - heartbeat_ts) > STALE_THRESHOLD_S


def _is_empty(m) -> bool:
    """True if this rank slot has never been written."""
    return (
        m.loader_batches_yielded == 0
        and m.lustre_bytes_read == 0
        and m.heartbeat_ts == 0
    )


def run_monitor(job_id: str) -> None:
    if not HAS_RICH:
        print("The monitor requires the `rich` library.  Install with:  pip install rich")
        sys.exit(1)

    registry = MetricsRegistry(job_id=job_id, create=False)
    if registry.shm is None:
        print(
            f"[ERROR] Could not connect to shared memory for job '{job_id}'.\n"
            "        Is the dataloader running on this node?",
        )
        sys.exit(1)

    last_mono         = time.monotonic()
    last_lustre_bytes = [0] * MAX_LOCAL_RANKS
    last_batches      = [0] * MAX_LOCAL_RANKS

    try:
        with Live(refresh_per_second=4, screen=True) as live:
            while True:
                now_mono = time.monotonic()
                now_wall = time.time()
                dt       = max(now_mono - last_mono, 1e-3)

                data = registry.read_all_ranks()
                if data is None:
                    break

                layout = Layout()
                layout.split_column(
                    Layout(name="header",  size=3),
                    Layout(name="globals", size=7),
                    Layout(name="ranks"),
                )

                layout["header"].update(
                    Panel(
                        Text(
                            f"DINOv3 Dataloader Monitor  ·  Job: {job_id}  ·  "
                            f"{time.strftime('%H:%M:%S')}",
                            style="bold white on blue",
                            justify="center",
                        ),
                    ),
                )

                m0           = data.ranks[0]
                lustre_rate  = (m0.lustre_bytes_read - last_lustre_bytes[0]) / dt
                batches_rate = (m0.loader_batches_yielded - last_batches[0]) / dt
                shm_util_pct = m0.shard_cache_utilization_pct

                globals_table = Table.grid(padding=(0, 4))
                globals_table.add_column(style="bold cyan",  justify="right")
                globals_table.add_column(style="white",      justify="left")
                globals_table.add_row("Lustre I/O",  f"{_fmt_bytes(lustre_rate)}/s")
                globals_table.add_row(
                    "Shard Cache",
                    f"{_bar(shm_util_pct, 100.0)}  {shm_util_pct:.1f}%",
                )
                globals_table.add_row("Throughput", f"{batches_rate:.2f} batches/s")
                layout["globals"].update(
                    Panel(globals_table, title="[bold]Node Pipeline  (Rank 0 view)[/bold]"),
                )

                ranks_table = Table(expand=True, show_lines=False)
                ranks_table.add_column("Rank",            justify="center",  style="cyan",    width=6)
                ranks_table.add_column("Batches",         justify="right",   style="blue")
                ranks_table.add_column("Net Stall (ms)",  justify="right",   style="red")
                ranks_table.add_column("Cache Wait (ms)", justify="right",   style="yellow")
                ranks_table.add_column("Pipe Yield (ms)", justify="right",   style="magenta")
                ranks_table.add_column("H2D (ms)",        justify="right",   style="green")
                ranks_table.add_column("Status",          justify="center",  style="white")

                for i in range(MAX_LOCAL_RANKS):
                    m = data.ranks[i]
                    if _is_empty(m):
                        continue

                    stale  = _is_stale(m.heartbeat_ts, now_wall)
                    status = "[dim]stale[/dim]" if stale else "[green]●[/green]"
                    style  = "dim" if stale else ""

                    def _cell(val: object) -> str:
                        return f"[{style}]{val}[/{style}]" if style else str(val)

                    ranks_table.add_row(
                        _cell(i),
                        _cell(m.loader_batches_yielded),
                        _cell(m.network_stall_time_ms),
                        _cell(m.shard_cache_wait_time_ms),
                        _cell(m.pipeline_yield_time_ms),
                        _cell(m.h2d_transfer_time_ms),
                        status,
                    )

                layout["ranks"].update(
                    Panel(ranks_table, title="[bold]Per-GPU Workers[/bold]"),
                )

                live.update(layout)

                # Only track rank-0 Lustre bytes (only rank 0 writes to Lustre).
                last_lustre_bytes[0] = data.ranks[0].lustre_bytes_read
                for i in range(MAX_LOCAL_RANKS):
                    last_batches[i] = data.ranks[i].loader_batches_yielded
                last_mono = now_mono

                time.sleep(0.25)

    except KeyboardInterrupt:
        pass


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
