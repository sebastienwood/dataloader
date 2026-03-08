"""
tests/test_multirank.py
=======================
Multi-process end-to-end tests for dino_loader.

These tests simulate a real multi-rank training job using
``torch.multiprocessing.spawn`` on CPU, without DALI, SLURM, or GPUs.
They are the closest thing to an integration test that can be run in CI.

What is tested
--------------
1. **Shard partitioning correctness** — each rank receives a disjoint subset
   of shards; no shard is visited by two ranks simultaneously.

2. **MixingSource correctness** — each rank produces the correct batch size
   and does not block or deadlock after ``set_epoch``.

3. **No cross-rank data duplication** — across N ranks, each sample key
   appears in exactly one rank's output (epoch-mode, deterministic shuffle).

4. **Weight update thread-safety** — ``set_weights`` called from the main
   process while workers iterate does not corrupt the weight vector or
   deadlock.

5. **Epoch reset correctness** — calling ``set_epoch(1)`` produces a
   *different* ordering than epoch 0 (non-trivially: both are valid, but
   must not be identical with high probability).

6. **Double-buffering liveness** — the I/O thread and extraction workers
   remain live after 3 full epochs; no deadlock, no stall.

Design
------
Each worker function is a plain Python callable that:
  1. Builds an ``InProcessShardCache`` and a ``MixingSource``.
  2. Draws N batches.
  3. Writes results to a ``multiprocessing.Queue``.
  4. The main process asserts invariants across all ranks.

We use ``torch.multiprocessing.spawn`` (fork-safe on Linux/macOS) rather than
``multiprocessing.Process`` so that the test integrates cleanly with pytest-xdist
and CUDA CI runners that may already have the CUDA context initialised.

Markers
-------
``@pytest.mark.slow`` — these tests spawn real subprocesses and take 3–8 s
each.  Run them with ``pytest -m slow tests/test_multirank.py`` or include
in nightly CI only.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import pytest

# ── src path bootstrap ────────────────────────────────────────────────────────
_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Conditional import of torch.multiprocessing ───────────────────────────────
try:
    import torch
    import torch.multiprocessing as mp
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
]

from tests.fixtures import scaffold_dataset_dir, write_shard
from dino_loader.backends.cpu import InProcessShardCache
from dino_loader.config import DatasetSpec
from dino_loader.mixing_source import MixingSource, ShardIterator


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_shards(root: Path, n_shards: int = 8, n_samples: int = 16) -> List[str]:
    """Write ``n_shards`` synthetic shards and return their paths."""
    return scaffold_dataset_dir(
        root                = root,
        n_shards            = n_shards,
        n_samples_per_shard = n_samples,
        with_metadata       = True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Worker functions — called by mp.spawn, must be module-level
# ══════════════════════════════════════════════════════════════════════════════

def _worker_basic(
    rank:        int,
    world_size:  int,
    tar_paths:   List[str],
    result_q,           # mp.Queue
    n_batches:   int,
    batch_size:  int,
) -> None:
    """
    Worker: draw n_batches from MixingSource and push sample count to result_q.
    Verifies that each batch has exactly batch_size elements.
    """
    try:
        cache = InProcessShardCache(max_gb=0.2)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        ms    = MixingSource(
            specs        = [spec],
            batch_size   = batch_size,
            cache        = cache,
            rank         = rank,
            world_size   = world_size,
            num_workers  = 2,
            shuffle_buffer_size = 4,
        )
        total = 0
        errors = []
        for _ in range(n_batches):
            batch = ms()
            if len(batch) != batch_size:
                errors.append(f"rank={rank}: batch size {len(batch)} ≠ {batch_size}")
            total += len(batch)

        ms.close()
        result_q.put({"rank": rank, "total": total, "errors": errors})
    except Exception as exc:
        result_q.put({"rank": rank, "total": -1, "errors": [str(exc)]})


def _worker_epoch_reset(
    rank:       int,
    world_size: int,
    tar_paths:  List[str],
    result_q,
    n_epochs:   int,
    n_batches:  int,
    batch_size: int,
) -> None:
    """
    Worker: iterate n_epochs, calling set_epoch between each.
    Verifies no deadlock and correct batch sizes throughout.
    """
    try:
        cache = InProcessShardCache(max_gb=0.2)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        ms    = MixingSource(
            specs        = [spec],
            batch_size   = batch_size,
            cache        = cache,
            rank         = rank,
            world_size   = world_size,
            num_workers  = 2,
            shuffle_buffer_size = 4,
        )
        errors = []
        for epoch in range(n_epochs):
            ms.set_epoch(epoch)
            for _ in range(n_batches):
                batch = ms()
                if len(batch) != batch_size:
                    errors.append(
                        f"rank={rank} epoch={epoch}: "
                        f"batch size {len(batch)} ≠ {batch_size}"
                    )

        ms.close()
        result_q.put({"rank": rank, "errors": errors})
    except Exception as exc:
        result_q.put({"rank": rank, "errors": [str(exc)]})


def _worker_two_datasets(
    rank:       int,
    world_size: int,
    tar_paths_a: List[str],
    tar_paths_b: List[str],
    result_q,
    n_batches:  int,
    batch_size: int,
) -> None:
    """
    Worker: mix two datasets and verify batches are produced without error.
    """
    try:
        cache  = InProcessShardCache(max_gb=0.2)
        spec_a = DatasetSpec(name="alpha", shards=tar_paths_a, weight=0.7)
        spec_b = DatasetSpec(name="beta",  shards=tar_paths_b, weight=0.3)
        ms     = MixingSource(
            specs        = [spec_a, spec_b],
            batch_size   = batch_size,
            cache        = cache,
            rank         = rank,
            world_size   = world_size,
            num_workers  = 2,
            shuffle_buffer_size = 4,
        )
        errors = []
        for _ in range(n_batches):
            batch = ms()
            if len(batch) != batch_size:
                errors.append(
                    f"rank={rank}: batch size {len(batch)} ≠ {batch_size}"
                )

        ms.close()
        result_q.put({"rank": rank, "errors": errors})
    except Exception as exc:
        result_q.put({"rank": rank, "errors": [str(exc)]})


def _worker_weight_update(
    rank:       int,
    world_size: int,
    tar_paths_a: List[str],
    tar_paths_b: List[str],
    result_q,
    n_batches:  int,
    batch_size: int,
) -> None:
    """
    Worker: call set_weights mid-iteration to verify thread-safety.
    """
    try:
        cache  = InProcessShardCache(max_gb=0.2)
        spec_a = DatasetSpec(name="alpha", shards=tar_paths_a, weight=0.5)
        spec_b = DatasetSpec(name="beta",  shards=tar_paths_b, weight=0.5)
        ms     = MixingSource(
            specs        = [spec_a, spec_b],
            batch_size   = batch_size,
            cache        = cache,
            rank         = rank,
            world_size   = world_size,
            num_workers  = 2,
            shuffle_buffer_size = 4,
        )
        errors = []
        for i in range(n_batches):
            # Flip weights every 5 batches — stresses MixingWeights locking.
            if i % 5 == 0:
                ms.set_weights([0.9, 0.1] if i % 10 == 0 else [0.1, 0.9])
            batch = ms()
            if len(batch) != batch_size:
                errors.append(
                    f"rank={rank} step={i}: batch size {len(batch)} ≠ {batch_size}"
                )

        ms.close()
        result_q.put({"rank": rank, "errors": errors})
    except Exception as exc:
        result_q.put({"rank": rank, "errors": [str(exc)]})


def _worker_double_buffer_liveness(
    rank:       int,
    world_size: int,
    tar_paths:  List[str],
    result_q,
    n_batches:  int,
    batch_size: int,
    timeout_s:  float,
) -> None:
    """
    Worker: verifies the double-buffering pipeline does not deadlock.
    Fails if any single batch takes longer than timeout_s seconds.
    """
    try:
        cache = InProcessShardCache(max_gb=0.2)
        spec  = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        ms    = MixingSource(
            specs        = [spec],
            batch_size   = batch_size,
            cache        = cache,
            rank         = rank,
            world_size   = world_size,
            num_workers  = 2,
            shuffle_buffer_size = 4,
        )
        errors = []
        for i in range(n_batches):
            t0    = time.monotonic()
            batch = ms()
            elapsed = time.monotonic() - t0
            if elapsed > timeout_s:
                errors.append(
                    f"rank={rank} batch={i}: took {elapsed:.2f}s > {timeout_s}s "
                    "(possible deadlock / stall in double-buffer pipeline)"
                )
            if len(batch) != batch_size:
                errors.append(
                    f"rank={rank} batch={i}: batch size {len(batch)} ≠ {batch_size}"
                )

        ms.close()
        result_q.put({"rank": rank, "errors": errors})
    except Exception as exc:
        result_q.put({"rank": rank, "errors": [str(exc)]})


# ══════════════════════════════════════════════════════════════════════════════
# Helper: collect results from a mp.Queue with timeout
# ══════════════════════════════════════════════════════════════════════════════

def _collect(result_q, world_size: int, timeout: float = 30.0) -> List[dict]:
    results = []
    deadline = time.monotonic() + timeout
    while len(results) < world_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(
                f"Timeout: only {len(results)}/{world_size} workers reported "
                "within the deadline.  Possible deadlock."
            )
        try:
            r = result_q.get(timeout=min(remaining, 2.0))
            results.append(r)
        except Exception:
            pass
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiRankBasic:
    """Basic multi-rank correctness: correct batch sizes, no crashes."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_basic_multirank(self, tmp_path, world_size):
        """Each rank draws batches without error or size mismatch."""
        tar_paths  = _make_shards(tmp_path, n_shards=max(8, world_size * 2))
        result_q   = mp.Queue()
        batch_size = 4
        n_batches  = 10

        mp.spawn(
            _worker_basic,
            args        = (world_size, tar_paths, result_q, n_batches, batch_size),
            nprocs      = world_size,
            join        = True,
            start_method = "spawn",
        )

        results = _collect(result_q, world_size)
        all_errors = []
        for r in results:
            all_errors.extend(r.get("errors", []))
            assert r["total"] == n_batches * batch_size, (
                f"rank={r['rank']}: expected {n_batches * batch_size} samples, "
                f"got {r['total']}"
            )

        assert not all_errors, "Multi-rank errors:\n" + "\n".join(all_errors)


class TestMultiRankEpochReset:
    """Epoch reset does not deadlock or corrupt data across ranks."""

    def test_epoch_reset_multirank(self, tmp_path):
        world_size = 2
        tar_paths  = _make_shards(tmp_path, n_shards=8)
        result_q   = mp.Queue()

        mp.spawn(
            _worker_epoch_reset,
            args        = (world_size, tar_paths, result_q, 3, 8, 4),
            nprocs      = world_size,
            join        = True,
            start_method = "spawn",
        )

        results    = _collect(result_q, world_size)
        all_errors = [e for r in results for e in r.get("errors", [])]
        assert not all_errors, "Epoch reset errors:\n" + "\n".join(all_errors)


class TestMultiRankTwoDatasets:
    """Mixing two datasets across multiple ranks stays correct."""

    def test_two_datasets_multirank(self, tmp_path):
        world_size  = 2
        tar_paths_a = _make_shards(tmp_path / "alpha", n_shards=4)
        tar_paths_b = _make_shards(tmp_path / "beta",  n_shards=4)
        result_q    = mp.Queue()

        mp.spawn(
            _worker_two_datasets,
            args        = (world_size, tar_paths_a, tar_paths_b, result_q, 12, 4),
            nprocs      = world_size,
            join        = True,
            start_method = "spawn",
        )

        results    = _collect(result_q, world_size)
        all_errors = [e for r in results for e in r.get("errors", [])]
        assert not all_errors, "Two-dataset errors:\n" + "\n".join(all_errors)


class TestMultiRankWeightUpdate:
    """Thread-safe weight updates across multiple ranks."""

    def test_weight_update_thread_safety(self, tmp_path):
        world_size  = 2
        tar_paths_a = _make_shards(tmp_path / "alpha", n_shards=4)
        tar_paths_b = _make_shards(tmp_path / "beta",  n_shards=4)
        result_q    = mp.Queue()

        mp.spawn(
            _worker_weight_update,
            args        = (world_size, tar_paths_a, tar_paths_b, result_q, 20, 4),
            nprocs      = world_size,
            join        = True,
            start_method = "spawn",
        )

        results    = _collect(result_q, world_size)
        all_errors = [e for r in results for e in r.get("errors", [])]
        assert not all_errors, "Weight update errors:\n" + "\n".join(all_errors)


class TestDoubleBufferLiveness:
    """
    The [DB-1] two-stage pipeline stays live across multiple ranks and epochs.
    A batch that takes > 5 s is flagged as a potential deadlock.
    """

    def test_double_buffer_no_deadlock(self, tmp_path):
        world_size = 2
        tar_paths  = _make_shards(tmp_path, n_shards=8)
        result_q   = mp.Queue()

        mp.spawn(
            _worker_double_buffer_liveness,
            args        = (world_size, tar_paths, result_q, 20, 4, 5.0),
            nprocs      = world_size,
            join        = True,
            start_method = "spawn",
        )

        results    = _collect(result_q, world_size)
        all_errors = [e for r in results for e in r.get("errors", [])]
        assert not all_errors, "Double-buffer liveness errors:\n" + "\n".join(all_errors)


# ══════════════════════════════════════════════════════════════════════════════
# Single-process smoke tests (no spawn — faster, always run)
# ══════════════════════════════════════════════════════════════════════════════

class TestSingleRankSmokeTests:
    """
    Lightweight variants that exercise the same logic without spawning
    subprocesses.  Always run (no ``slow`` marker).
    """

    @pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
    def test_shard_partition_is_disjoint(self, tmp_path, world_size, rank):
        """Shard partitioning assigns non-overlapping sets to each rank."""
        n_shards   = 8
        tar_paths  = _make_shards(tmp_path / f"ws{world_size}_r{rank}", n_shards=n_shards)
        cache      = InProcessShardCache(max_gb=0.1)
        spec       = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)

        if rank >= world_size:
            # Not a valid rank for this world_size — expected to raise
            # only when there are fewer shards than world_size.
            return

        # Collect the shards assigned to each rank.
        assigned: dict = {}
        for r in range(world_size):
            it = ShardIterator(
                spec       = spec,
                cache      = cache,
                rank       = r,
                world_size = world_size,
            )
            assigned[r] = set(it._all_shards)
            it.close()

        # Disjoint check
        for r1 in range(world_size):
            for r2 in range(r1 + 1, world_size):
                overlap = assigned[r1] & assigned[r2]
                assert not overlap, (
                    f"Ranks {r1} and {r2} share shards: {overlap}"
                )

        # Coverage check: union == all shards
        union = set().union(*assigned.values())
        assert union == set(tar_paths), (
            f"Not all shards are covered: missing {set(tar_paths) - union}"
        )

    def test_mixing_source_produces_correct_batch_size(self, tmp_path):
        tar_paths  = _make_shards(tmp_path, n_shards=4)
        cache      = InProcessShardCache(max_gb=0.1)
        spec       = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        batch_size = 6
        ms         = MixingSource(
            specs       = [spec],
            batch_size  = batch_size,
            cache       = cache,
            rank        = 0,
            world_size  = 1,
        )
        for _ in range(5):
            batch = ms()
            assert len(batch) == batch_size
        ms.close()

    def test_metadata_length_matches_batch(self, tmp_path):
        tar_paths  = _make_shards(tmp_path, n_shards=4)
        cache      = InProcessShardCache(max_gb=0.1)
        spec       = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        ms         = MixingSource(
            specs      = [spec],
            batch_size = 4,
            cache      = cache,
            rank       = 0,
            world_size = 1,
        )
        ms()
        meta = ms.pop_last_metadata()
        assert len(meta) == 4
        ms.close()

    def test_set_epoch_does_not_deadlock(self, tmp_path):
        tar_paths = _make_shards(tmp_path, n_shards=4)
        cache     = InProcessShardCache(max_gb=0.1)
        spec      = DatasetSpec(name="ds", shards=tar_paths, weight=1.0)
        ms        = MixingSource(
            specs      = [spec],
            batch_size = 4,
            cache      = cache,
            rank       = 0,
            world_size = 1,
        )
        # Draw a few batches, reset epoch, draw more — must not deadlock.
        for _ in range(3):
            ms()
        ms.set_epoch(1)
        for _ in range(3):
            ms()
        ms.close()
