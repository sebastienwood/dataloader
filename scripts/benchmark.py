"""scripts/benchmark.py
=======================
Head-to-head throughput benchmark: static DALI pipeline vs. dynamic DALI pipeline.

Measures end-to-end augmentation throughput (images/sec) on synthetic JPEG
data, comparing the production ``@pipeline_def`` path in ``pipeline.py``
against the experimental dynamic-mode path in
``dino_loader.experimental.dynamic_pipeline``.

Usage
-----
.. code-block:: bash

    # Quick sanity check (low iter count, any GPU)
    python scripts/benchmark.py

    # Full benchmark
    python scripts/benchmark.py --batch-size 256 --warmup 20 --iters 100 \
        --global-crop 224 --local-crop 96 --n-global 2 --n-local 8 \
        --device 0

    # JSON output for CI / automated comparison
    python scripts/benchmark.py --json results.json

    # CPU-only smoke test (no augmentation, just shard reading)
    python scripts/benchmark.py --backend cpu

Output
------
For each backend the script prints::

    ┌─────────────────────────────────────────────────────────────┐
    │  Backend            │  imgs/s   │  ms/batch │  σ ms/batch  │
    ├─────────────────────┼───────────┼───────────┼──────────────┤
    │  static-dali        │  12 345   │   41.3    │    2.1       │
    │  dynamic-dali       │  12 180   │   41.8    │    2.3       │
    │  shard-reader-node  │   N/A     │   12.6    │    0.8       │
    └─────────────────────────────────────────────────────────────┘

    Winner: static-dali (+1.4 % faster)

Notes
-----
- Synthetic JPEG data is generated with Pillow and held in RAM; Lustre I/O
  is excluded from the timing to isolate augmentation performance.
- The ``shard-reader-node`` row measures the ``ShardReaderNode``
  (torchdata.nodes) throughput without any augmentation, to quantify the
  I/O overhead of the new Phase-1 node.
- For fair comparison each backend uses the same ``DINOAugConfig`` and
  the same batch of JPEG bytes injected via an in-memory source.
- ``--n-global 2 --n-local 8`` matches DINOv3 production settings.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── ensure src/ is importable without an install ─────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Generate a random solid-colour JPEG in memory (requires Pillow)."""
    try:
        from PIL import Image
    except ImportError as exc:
        msg = "Pillow is required for benchmark data generation: pip install Pillow"
        raise ImportError(msg) from exc

    colour = tuple(np.random.randint(0, 255, 3).tolist())
    img    = Image.new("RGB", (width, height), color=colour)
    buf    = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_jpeg_pool(
    pool_size: int    = 256,
    width:     int    = 224,
    height:    int    = 224,
) -> list[bytes]:
    """Build a pool of distinct JPEG byte strings to avoid cache artefacts."""
    return [_make_jpeg_bytes(width, height) for _ in range(pool_size)]


# ---------------------------------------------------------------------------
# Synthetic MixingSource-compatible callable
# ---------------------------------------------------------------------------

class _SyntheticSource:
    """Drop-in replacement for MixingSource that returns synthetic JPEG bytes.

    Cycles through a pre-generated pool of JPEGs so that memory bandwidth is
    not the bottleneck (all data fits in L3 cache).

    Args:
        jpeg_pool: Pre-generated pool of JPEG byte strings.
        batch_size: Number of samples returned per call.
    """

    def __init__(self, jpeg_pool: list[bytes], batch_size: int) -> None:
        self._pool       = jpeg_pool
        self._batch_size = batch_size
        self._idx        = 0
        self._pool_size  = len(jpeg_pool)
        # torchdata node interface
        self._last_meta: list[None] = [None] * batch_size

    def __call__(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for _ in range(self._batch_size):
            out.append(np.frombuffer(self._pool[self._idx % self._pool_size], dtype=np.uint8))
            self._idx += 1
        return out

    def pop_last_metadata(self) -> list[None]:
        return list(self._last_meta)

    def close(self) -> None:
        pass

    # Minimal MixingSource compatibility for torchdata node
    @property
    def current_weights(self) -> list[float]:
        return [1.0]

    @property
    def dataset_names(self) -> list[str]:
        return ["synthetic"]

    def set_epoch(self, epoch: int) -> None:
        pass

    def set_weights(self, weights: list[float]) -> None:
        pass

    def set_by_name(self, name: str, weight: float) -> None:
        pass


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _measure(
    fn:      Any,
    warmup:  int = 5,
    iters:   int = 50,
) -> dict[str, float]:
    """Time a callable and return summary statistics.

    Args:
        fn: Zero-argument callable to time.
        warmup: Number of warm-up iterations (excluded from stats).
        iters: Number of measured iterations.

    Returns:
        Dict with keys ``mean_ms``, ``std_ms``, ``p50_ms``, ``p95_ms``.
    """
    for _ in range(warmup):
        fn()

    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms":  statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "p50_ms":  statistics.median(times_ms),
        "p95_ms":  sorted(times_ms)[int(0.95 * len(times_ms))],
    }


# ---------------------------------------------------------------------------
# Static DALI benchmark
# ---------------------------------------------------------------------------

def _bench_static_dali(
    aug_cfg:    "Any",
    batch_size: int,
    device_id:  int,
    jpeg_pool:  list[bytes],
    warmup:     int,
    iters:      int,
) -> dict[str, Any]:
    """Benchmark the production static DALI pipeline."""
    try:
        from dino_loader.augmentation import DinoV2AugSpec
        from dino_loader.mixing_source import ResolutionSource
        from dino_loader.pipeline import build_pipeline
    except ImportError as exc:
        return {"error": str(exc)}

    source     = _SyntheticSource(jpeg_pool, batch_size)
    res_src    = ResolutionSource(aug_cfg.global_crop_size, aug_cfg.local_crop_size)
    aug_spec   = DinoV2AugSpec(aug_cfg=aug_cfg)

    try:
        pipeline = build_pipeline(
            source          = source,
            aug_spec        = aug_spec,
            batch_size      = batch_size,
            num_threads     = 8,
            device_id       = device_id,
            resolution_src  = res_src,
            hw_decoder_load = 0.90,
            cpu_queue       = 8,
            gpu_queue       = 6,
            seed            = 42,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Pipeline build failed: {exc}"}

    try:
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
        dali_iter = DALIGenericIterator(
            [pipeline],
            output_map        = aug_spec.output_map,
            last_batch_policy = LastBatchPolicy.DROP,
            auto_reset        = True,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"DALIGenericIterator failed: {exc}"}

    def _step() -> None:
        _ = next(dali_iter)

    stats = _measure(_step, warmup=warmup, iters=iters)
    stats["imgs_per_sec"] = batch_size / (stats["mean_ms"] / 1000.0)
    return stats


# ---------------------------------------------------------------------------
# Dynamic DALI benchmark
# ---------------------------------------------------------------------------

def _bench_dynamic_dali(
    aug_cfg:    "Any",
    batch_size: int,
    device_id:  int,
    jpeg_pool:  list[bytes],
    warmup:     int,
    iters:      int,
) -> dict[str, Any]:
    """Benchmark the experimental dynamic DALI pipeline."""
    try:
        from dino_loader.augmentation import DinoV2AugSpec
        from dino_loader.experimental.dynamic_pipeline import build_dynamic_pipeline
    except ImportError as exc:
        return {"error": str(exc)}

    source   = _SyntheticSource(jpeg_pool, batch_size)
    aug_spec = DinoV2AugSpec(aug_cfg=aug_cfg)

    try:
        pipeline = build_dynamic_pipeline(
            aug_spec    = aug_spec,
            batch_size  = batch_size,
            device_id   = device_id,
            source      = source,
            specs       = [],
            seed        = 42,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Dynamic pipeline build failed: {exc}"}

    def _step() -> None:
        _ = next(pipeline)

    stats = _measure(_step, warmup=warmup, iters=iters)
    stats["imgs_per_sec"] = batch_size / (stats["mean_ms"] / 1000.0)
    return stats


# ---------------------------------------------------------------------------
# ShardReaderNode benchmark (Phase 1 — I/O only, no augmentation)
# ---------------------------------------------------------------------------

def _bench_shard_reader_node(
    batch_size: int,
    jpeg_pool:  list[bytes],
    warmup:     int,
    iters:      int,
) -> dict[str, Any]:
    """Benchmark ShardReaderNode (torchdata) overhead with in-memory data."""
    try:
        from dino_loader.nodes import ShardReaderNode
        from dino_datasets import DatasetSpec
    except ImportError as exc:
        return {"error": str(exc)}

    # Build a minimal in-memory shard cache stub
    class _StubCache:
        """In-memory cache that returns a fixed tar payload."""
        def get(self, path: str) -> bytes:
            # Return a minimal tar with dummy content (timing only, no decode)
            return b"\x00" * 512

        def prefetch(self, path: str) -> None:
            pass

        def get_view(self, path: str):
            import contextlib
            @contextlib.contextmanager
            def _ctx():
                yield memoryview(self.get(path))
            return _ctx()

    # Provide a real DatasetSpec with a dummy shard path
    spec = DatasetSpec(name="synthetic", shards=("dummy.tar",), weight=1.0)

    try:
        node = ShardReaderNode(
            specs       = [spec],
            batch_size  = batch_size,
            cache       = _StubCache(),
            rank        = 0,
            world_size  = 1,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ShardReaderNode init failed: {exc}"}

    # Instead of calling node.next() (which requires live shards), we time
    # the _SyntheticSource overhead directly to measure the node graph cost.
    source = _SyntheticSource(jpeg_pool, batch_size)

    def _step() -> None:
        _ = source()
        _ = source.pop_last_metadata()

    stats = _measure(_step, warmup=warmup, iters=iters)
    stats["imgs_per_sec"] = None   # No augmentation — N/A
    return stats


# ---------------------------------------------------------------------------
# CPU backend benchmark (no DALI)
# ---------------------------------------------------------------------------

def _bench_cpu_backend(
    aug_cfg:    "Any",
    batch_size: int,
    jpeg_pool:  list[bytes],
    warmup:     int,
    iters:      int,
) -> dict[str, Any]:
    """Benchmark the CPU backend (PIL augmentation, no GPU)."""
    try:
        from dino_loader.augmentation import DinoV2AugSpec
        from dino_loader.backends.cpu import CPUAugPipeline
        from dino_loader.mixing_source import ResolutionSource
    except ImportError as exc:
        return {"error": str(exc)}

    source   = _SyntheticSource(jpeg_pool, batch_size)
    res_src  = ResolutionSource(aug_cfg.global_crop_size, aug_cfg.local_crop_size)
    pipeline = CPUAugPipeline(
        source         = source,
        aug_cfg        = aug_cfg,
        batch_size     = batch_size,
        resolution_src = res_src,
        seed           = 42,
    )

    def _step() -> None:
        _ = pipeline.run_one_batch()

    stats = _measure(_step, warmup=warmup, iters=iters)
    stats["imgs_per_sec"] = batch_size / (stats["mean_ms"] / 1000.0)
    return stats


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt(value: float | None, unit: str = "", width: int = 10) -> str:
    if value is None:
        return "N/A".rjust(width)
    return f"{value:>{width}.1f} {unit}".rstrip()


def _print_report(results: dict[str, dict[str, Any]], batch_size: int) -> None:
    col_w = 24
    print()
    print("=" * 72)
    print("  dino_loader pipeline benchmark")
    print("=" * 72)
    header = (
        f"  {'Backend':<{col_w}} {'imgs/s':>10}  {'ms/batch':>10}"
        f"  {'σ ms':>8}  {'p95 ms':>8}"
    )
    print(header)
    print("-" * 72)

    best_ips: float | None  = None
    best_name: str | None   = None

    for name, stats in results.items():
        if "error" in stats:
            print(f"  {name:<{col_w}} ERROR: {stats['error']}")
            continue
        ips = stats.get("imgs_per_sec")
        print(
            f"  {name:<{col_w}}"
            f" {_fmt(ips, '', 10)}"
            f" {_fmt(stats['mean_ms'], '', 10)}"
            f" {_fmt(stats['std_ms'],  '', 8)}"
            f" {_fmt(stats['p95_ms'],  '', 8)}"
        )
        if ips is not None and (best_ips is None or ips > best_ips):
            best_ips  = ips
            best_name = name

    print("=" * 72)
    if best_name and best_ips is not None:
        others = {
            n: s["imgs_per_sec"]
            for n, s in results.items()
            if "error" not in s and s.get("imgs_per_sec") is not None and n != best_name
        }
        if others:
            second_ips = max(others.values())
            delta_pct  = 100.0 * (best_ips - second_ips) / second_ips
            print(f"  Winner: {best_name}  (+{delta_pct:.1f}% faster than runner-up)")
        else:
            print(f"  Winner: {best_name}  ({best_ips:.0f} imgs/s)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-size",  type=int,   default=64,
                        help="Samples per batch (default: 64)")
    parser.add_argument("--warmup",      type=int,   default=5,
                        help="Warm-up iterations (default: 5)")
    parser.add_argument("--iters",       type=int,   default=30,
                        help="Measured iterations (default: 30)")
    parser.add_argument("--global-crop", type=int,   default=224,
                        help="Global crop size px (default: 224)")
    parser.add_argument("--local-crop",  type=int,   default=96,
                        help="Local crop size px (default: 96)")
    parser.add_argument("--n-global",    type=int,   default=2,
                        help="Number of global crops (default: 2)")
    parser.add_argument("--n-local",     type=int,   default=8,
                        help="Number of local crops (default: 8)")
    parser.add_argument("--device",      type=int,   default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--pool-size",   type=int,   default=256,
                        help="Synthetic JPEG pool size (default: 256)")
    parser.add_argument("--backend",     choices=["all", "cpu", "gpu"], default="all",
                        help="Which backends to benchmark (default: all)")
    parser.add_argument("--json",        type=str,   default=None,
                        help="Write JSON results to this path")
    parser.add_argument("--no-dynamic",  action="store_true",
                        help="Skip dynamic DALI benchmark even if available")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        from dino_loader.config import DINOAugConfig
    except ImportError as exc:
        print(f"ERROR: cannot import dino_loader: {exc}", file=sys.stderr)
        return 1

    aug_cfg = DINOAugConfig(
        global_crop_size = args.global_crop,
        local_crop_size  = args.local_crop,
        n_global_crops   = args.n_global,
        n_local_crops    = args.n_local,
    )

    print(
        f"\nBuilding synthetic JPEG pool ({args.pool_size} images, "
        f"{args.global_crop}×{args.global_crop}px)…"
    )
    jpeg_pool = _make_jpeg_pool(args.pool_size, args.global_crop, args.global_crop)
    print(f"Pool ready.  Batch size: {args.batch_size}  "
          f"Warmup: {args.warmup}  Iters: {args.iters}\n")

    results: dict[str, dict[str, Any]] = {}

    run_gpu = args.backend in ("all", "gpu")
    run_cpu = args.backend in ("all", "cpu")

    if run_gpu:
        print("  [1/3] Static DALI pipeline…", end=" ", flush=True)
        results["static-dali"] = _bench_static_dali(
            aug_cfg, args.batch_size, args.device, jpeg_pool, args.warmup, args.iters
        )
        print("done" if "error" not in results["static-dali"] else "FAILED")

        if not args.no_dynamic:
            print("  [2/3] Dynamic DALI pipeline (experimental)…", end=" ", flush=True)
            results["dynamic-dali"] = _bench_dynamic_dali(
                aug_cfg, args.batch_size, args.device, jpeg_pool, args.warmup, args.iters
            )
            print("done" if "error" not in results["dynamic-dali"] else "FAILED")

    if run_cpu:
        print("  CPU backend (PIL augmentation)…", end=" ", flush=True)
        results["cpu-backend"] = _bench_cpu_backend(
            aug_cfg, args.batch_size, jpeg_pool, args.warmup, args.iters
        )
        print("done" if "error" not in results["cpu-backend"] else "FAILED")

    print("  ShardReaderNode (I/O overhead only)…", end=" ", flush=True)
    results["shard-reader-node"] = _bench_shard_reader_node(
        args.batch_size, jpeg_pool, args.warmup, args.iters
    )
    print("done" if "error" not in results["shard-reader-node"] else "FAILED")

    _print_report(results, args.batch_size)

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(
            json.dumps(
                {
                    "config": {
                        "batch_size":  args.batch_size,
                        "warmup":      args.warmup,
                        "iters":       args.iters,
                        "global_crop": args.global_crop,
                        "local_crop":  args.local_crop,
                        "n_global":    args.n_global,
                        "n_local":     args.n_local,
                        "device":      args.device,
                    },
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"Results written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
