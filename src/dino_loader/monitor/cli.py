"""
dino_loader.monitor.cli
=======================
Real-time terminal UI for Dataloader monitoring (htop style).
"""

import argparse
import time
import sys

try:
    from rich.live import Live
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .metrics import MetricsRegistry, MAX_LOCAL_RANKS

def format_bytes(b: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if b < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"

def _bars(current: int, target: int, length: int = 20) -> str:
    if target <= 0:
        return " " * length
    ratio = min(max(current / target, 0.0), 1.0)
    filled = int(ratio * length)
    return "█" * filled + "▒" * (length - filled)

def run_monitor(job_id: str):
    if not HAS_RICH:
        print("The monitor requires the `rich` library. Install with `pip install rich`.")
        sys.exit(1)

    registry = MetricsRegistry(job_id, create=False)
    if registry.shm is None:
        print(f"Could not connect to shared memory for job '{job_id}'. Is the dataloader running?")
        print("Note: Fast-path metrics require the dataloader to be currently active.")
        sys.exit(1)

    # For rate tracking
    last_time = time.time()
    last_lustre_bytes = [0] * MAX_LOCAL_RANKS
    last_batches = [0] * MAX_LOCAL_RANKS

    try:
        with Live(refresh_per_second=4, screen=True) as live:
            while True:
                current_time = time.time()
                dt = current_time - last_time
                if dt == 0: dt = 0.001

                data = registry.read_all_ranks()
                if data is None:
                    break

                # We build the layout dynamically here to be simple
                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="queues", size=9),
                    Layout(name="ranks")
                )

                header_text = Text(f"DINOv3 Dataloader Monitor (Job: {job_id})", style="bold white on blue", justify="center")
                layout["header"].update(Panel(header_text))

                queue_table = Table.grid(padding=(0, 2))
                queue_table.add_column(justify="left")
                queue_table.add_column(justify="left")
                
                # Only use rank 0 for Lustre & Pipeline globals
                m0 = data.ranks[0]
                
                lustre_speed = (m0.lustre_bytes_read - last_lustre_bytes[0]) / dt
                batches_speed = (m0.loader_batches_yielded - last_batches[0]) / dt
                
                queue_text = f"""
[bold]Metrics (Rank 0 Global View):[/bold]
  Lustre Read : {format_bytes(lustre_speed)}/s
  Throughput  : {batches_speed:.2f} batches/s
  Shm Util    : {_bars(int(m0.shard_cache_utilization_pct), 100, 30)} {m0.shard_cache_utilization_pct:.1f}%
                """
                layout["queues"].update(Panel(queue_text, title="Node Pipeline Status"))

                workers_table = Table(title="Local Worker Threads (Ranks)", expand=True)
                workers_table.add_column("Rank", justify="center", style="cyan")
                workers_table.add_column("Batches", justify="right", style="blue")
                workers_table.add_column("Net Stall (ms)", justify="right", style="red")
                workers_table.add_column("Mutex Wait (ms)", justify="right", style="yellow")
                workers_table.add_column("Pipe Yield (ms)", justify="right", style="magenta")

                for i in range(MAX_LOCAL_RANKS):
                    m = data.ranks[i]
                    if m.loader_batches_yielded == 0 and m.lustre_read_time_ms == 0:
                        continue
                    
                    workers_table.add_row(
                        str(i),
                        f"{m.loader_batches_yielded}",
                        f"{m.lustre_read_time_ms}",
                        f"{m.shard_wait_time_ms}",
                        f"{m.pipeline_yield_time_ms}"
                    )

                layout["ranks"].update(Panel(workers_table, title="Workers"))

                live.update(layout)

                # Store history
                for i in range(MAX_LOCAL_RANKS):
                    last_lustre_bytes[i] = data.ranks[i].lustre_bytes_read
                    last_batches[i] = data.ranks[i].loader_batches_yielded
                last_time = current_time

                time.sleep(0.25)
                
    except KeyboardInterrupt:
        pass

def main():
    parser = argparse.ArgumentParser(description="Monitor Dataloader Performance")
    parser.add_argument("--job", type=str, default="dino", help="Job ID to monitor (same as loader job_id)")
    args = parser.parse_args()
    run_monitor(args.job)

if __name__ == "__main__":
    main()
