"""tests/test_reader_adapter.py
================================
Unit tests for ``dino_loader.shard_reader._ReaderAdapter``.

Coverage
--------
Metadata FIFO alignment invariant
- pop_last_metadata returns metadata in exact call order (FIFO)
- metadata aligns 1:1 with __call__ invocations
- pop_last_metadata on empty queue returns []

Queue overflow — [FIX-META-QUEUE-OVERFLOW]
- put_nowait raises RuntimeError when meta_queue_size is exhausted
- RuntimeError message is descriptive (mentions meta_queue_size, DALI queues)
- No silent drop: queue state is consistent after overflow attempt

The previous implementation silently dropped the *oldest* metadata entry
when the queue was full, causing a permanent off-by-one misalignment between
DALI batches and their metadata for the remainder of the training run.
This file directly regression-tests that the silent-drop path is gone.
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.shard_reader import _ReaderAdapter
from dino_loader.sources.resolution import ResolutionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    batches: list[tuple[list[np.ndarray], list[dict | None]]],
    *,
    meta_queue_size: int = 64,
) -> _ReaderAdapter:
    """Build a _ReaderAdapter backed by a fake ShardReaderNode.

    The fake reader cycles through ``batches`` in order, each call to
    ``next()`` popping the next entry.  Unlike a real ShardReaderNode,
    metadata is returned directly from ``next()`` — pop_last_metadata()
    on the MixingSource always returns [] (which is the real behaviour;
    alignment is managed by _ReaderAdapter._meta_queue).
    """
    call_idx = [0]

    class _FakeSource:
        def __call__(self) -> list[np.ndarray]:
            idx = call_idx[0] % len(batches)
            call_idx[0] += 1
            return batches[idx][0]

        def pop_last_metadata(self) -> list[dict | None]:
            # MixingSource always returns [] — metadata travels via _meta_queue.
            return []

        register_dataset_index_callback = MagicMock()

    class _FakeReader:
        _source = _FakeSource()
        _idx    = [0]

        def next(self) -> tuple[list[np.ndarray], list[dict | None]]:
            idx = self._idx[0] % len(batches)
            self._idx[0] += 1
            return batches[idx]

    res_src = ResolutionSource(224, 96)
    return _ReaderAdapter(
        reader          = _FakeReader(),  # type: ignore[arg-type]
        resolution_src  = res_src,
        batch_size      = 2,
        meta_queue_size = meta_queue_size,
    )


def _jpeg(tag: int) -> np.ndarray:
    return np.array([tag], dtype=np.uint8)


def _batch(tag: int) -> tuple[list[np.ndarray], list[dict | None]]:
    return [_jpeg(tag)], [{"tag": tag}]


# ---------------------------------------------------------------------------
# FIFO alignment
# ---------------------------------------------------------------------------


class TestReaderAdapterFIFO:

    def test_metadata_returned_in_call_order(self) -> None:
        """Metadata must come back in the same order as __call__ invocations."""
        batches = [_batch(i) for i in range(5)]
        adapter = _make_adapter(batches)

        for i in range(5):
            adapter()  # enqueues metadata for batch i

        for i in range(5):
            meta = adapter.pop_last_metadata()
            assert meta == [{"tag": i}], (
                f"Expected metadata for batch {i}, got {meta}"
            )

    def test_one_to_one_alignment(self) -> None:
        """Each __call__ enqueues exactly one metadata entry."""
        batches = [_batch(i) for i in range(3)]
        adapter = _make_adapter(batches)

        # Interleave calls and pops to verify 1:1 mapping.
        adapter()
        assert adapter.pop_last_metadata() == [{"tag": 0}]

        adapter()
        adapter()
        assert adapter.pop_last_metadata() == [{"tag": 1}]
        assert adapter.pop_last_metadata() == [{"tag": 2}]

    def test_pop_on_empty_queue_returns_empty_list(self) -> None:
        """pop_last_metadata on an empty queue must return [] without raising."""
        adapter = _make_adapter([_batch(0)])
        result = adapter.pop_last_metadata()
        assert result == []

    def test_pop_after_all_consumed_returns_empty_list(self) -> None:
        adapter = _make_adapter([_batch(7)])
        adapter()
        adapter.pop_last_metadata()       # consume the one entry
        assert adapter.pop_last_metadata() == []


# ---------------------------------------------------------------------------
# Queue overflow — [FIX-META-QUEUE-OVERFLOW]
# ---------------------------------------------------------------------------


class TestReaderAdapterQueueOverflow:
    """Regression tests for the silent drop-oldest bug.

    The old implementation caught queue.Full and silently dropped the oldest
    entry to make room, creating a permanent 1-batch metadata misalignment.
    The fix raises RuntimeError instead, failing loudly and early.
    """

    def test_overflow_raises_runtime_error(self) -> None:
        """Filling the queue beyond capacity must raise RuntimeError."""
        size    = 3
        batches = [_batch(i) for i in range(size + 1)]
        adapter = _make_adapter(batches, meta_queue_size=size)

        for _ in range(size):
            adapter()  # fill the queue to capacity

        with pytest.raises(RuntimeError, match="metadata queue full"):
            adapter()  # one more call must raise, not silently drop

    def test_overflow_error_message_is_descriptive(self) -> None:
        """RuntimeError message must reference meta_queue_size and DALI queues."""
        adapter = _make_adapter([_batch(0)], meta_queue_size=1)
        adapter()  # fill the single slot

        with pytest.raises(RuntimeError) as exc_info:
            adapter()

        msg = str(exc_info.value)
        assert "meta_queue_size" in msg
        assert "DALI" in msg.upper() or "dali" in msg

    def test_no_silent_drop_queue_state_after_overflow(self) -> None:
        """After an overflow RuntimeError, existing queue entries are untouched.

        This directly tests the absence of the old drop-oldest behaviour:
        if any entry had been silently removed, the subsequent pop would
        return the wrong metadata.
        """
        size    = 2
        batches = [_batch(i) for i in range(size + 1)]
        adapter = _make_adapter(batches, meta_queue_size=size)

        adapter()  # enqueue tag=0
        adapter()  # enqueue tag=1  (queue now full)

        with pytest.raises(RuntimeError):
            adapter()  # must NOT drop tag=0 to make room for tag=2

        # Both original entries must still be intact and in order.
        assert adapter.pop_last_metadata() == [{"tag": 0}]
        assert adapter.pop_last_metadata() == [{"tag": 1}]
        assert adapter.pop_last_metadata() == []

    def test_default_queue_size_accommodates_dali_defaults(self) -> None:
        """Default meta_queue_size (64) must never overflow under typical DALI config.

        With dali_cpu_queue=16, dali_gpu_queue=6, prefetch_factor=2, the max
        simultaneous in-flight __call__ invocations before any pop is ≈ 24.
        64 > 24, so no overflow should occur.
        """
        from dino_loader.shard_reader import _DEFAULT_META_QUEUE_SIZE

        max_in_flight = 16 + 6 + 2  # cpu_queue + gpu_queue + prefetch_factor
        assert _DEFAULT_META_QUEUE_SIZE > max_in_flight, (
            f"Default meta_queue_size ({_DEFAULT_META_QUEUE_SIZE}) must exceed "
            f"max in-flight DALI calls ({max_in_flight}) to prevent overflow."
        )

    def test_custom_queue_size_is_respected(self) -> None:
        """meta_queue_size parameter controls actual queue capacity."""
        custom_size = 5
        adapter = _make_adapter([_batch(i) for i in range(custom_size + 1)], meta_queue_size=custom_size)

        for _ in range(custom_size):
            adapter()

        with pytest.raises(RuntimeError):
            adapter()  # slot custom_size+1 must raise