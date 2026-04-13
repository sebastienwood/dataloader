"""tests/test_pipeline.py
=========================
Unit tests for :mod:`dino_loader.pipeline`.

Coverage
--------
NormSource [M2]
- set_dataset_indices is a full copy-on-write replacement
- __call__ returns independent numpy copies — mutating the returned array
  must NOT affect the internal lookup table or subsequent calls
- Concurrent set + call does not corrupt or raise

pipeline dispatch
- build_pipeline raises RuntimeError when DALI is not installed
- build_pipeline raises TypeError on unknown AugmentationSpec
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.config import DINOAugConfig, NormStats

# ══════════════════════════════════════════════════════════════════════════════
# NormSource — thread-safe copy-on-write [M2]
# ══════════════════════════════════════════════════════════════════════════════


def _make_norm_source(n_datasets: int = 2):
    from dino_loader.pipeline import NormSource
    specs = [MagicMock(mean=None, std=None) for _ in range(n_datasets)]
    return NormSource(aug_cfg=DINOAugConfig(), specs=specs)


class TestNormSourceSetIndices:

    def test_set_indices_replaces_entire_list(self):
        ns = _make_norm_source(3)
        ns.set_dataset_indices([0, 1, 2])
        ns.set_dataset_indices([2, 1])
        means, stds = ns()
        assert len(means) == 2
        assert len(stds) == 2

    def test_call_length_matches_indices(self):
        ns = _make_norm_source(4)
        ns.set_dataset_indices([0, 3, 1])
        means, stds = ns()
        assert len(means) == 3
        assert len(stds) == 3

    def test_single_index_returns_length_one(self):
        ns = _make_norm_source(2)
        ns.set_dataset_indices([0])
        means, stds = ns()
        assert len(means) == 1
        assert len(stds) == 1


class TestNormSourceReturnsCopies:

    def test_mutating_returned_array_does_not_affect_next_call(self):
        """[M2] Copy-on-write guarantee: returned arrays are independent copies.

        Mutating the first call's result must leave subsequent calls unaffected.
        If NormSource returns views into its internal lookup table, writes to
        those views would corrupt the lookup, and the second call would produce
        the mutated values — which this test catches.
        """
        ns = _make_norm_source(1)
        ns.set_dataset_indices([0])

        means_first, _ = ns()
        # Corrupt the returned array in-place.
        original_value = float(means_first[0][0])
        means_first[0][:] = 99.0

        means_second, _ = ns()
        # The second call must produce the original (uncorrupted) values.
        assert not np.allclose(means_second[0], 99.0), (
            "NormSource returned a view into its internal lookup table. "
            "Mutating the result corrupted the next call — copy-on-write is broken."
        )
        # Sanity: the second call should reproduce the original value.
        assert np.isclose(means_second[0][0], original_value), (
            f"Expected original value {original_value}, got {means_second[0][0]}."
        )

    def test_two_successive_calls_return_distinct_arrays(self):
        """Each call must allocate fresh arrays, never the same object."""
        ns = _make_norm_source(1)
        ns.set_dataset_indices([0])
        means_a, _ = ns()
        means_b, _ = ns()
        # Different objects (not just different values).
        assert means_a[0] is not means_b[0], (
            "NormSource returned the same array object in two consecutive calls."
        )


class TestNormSourceConcurrency:

    def test_concurrent_set_and_call_no_errors(self):
        ns = _make_norm_source(4)
        ns.set_dataset_indices([0, 1, 2, 3])
        errors: list[Exception] = []

        def _setter():
            for i in range(200):
                ns.set_dataset_indices([i % 4])

        def _caller():
            for _ in range(200):
                try:
                    means, stds = ns()
                    # Basic sanity: every returned array must be float32 with 3 elements.
                    for arr in means + stds:
                        assert arr.dtype == np.float32
                        assert arr.shape == (3,)
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=_setter),
            threading.Thread(target=_caller),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in concurrent NormSource access: {errors}"

    def test_set_indices_is_atomic(self):
        """set_dataset_indices must never leave _indices in a partially-updated state.

        Two threads alternate between two index lists of different lengths.
        The caller thread must always observe a complete, consistent list —
        never a mix of elements from the two lists.
        """
        ns = _make_norm_source(4)
        ns.set_dataset_indices([0])
        errors: list[str] = []

        def _alternating_setter():
            for i in range(500):
                ns.set_dataset_indices([0] if i % 2 == 0 else [1, 2])

        def _length_checker():
            for _ in range(500):
                try:
                    means, _ = ns()
                    # Length must always be 1 or 2 — never anything else.
                    if len(means) not in (1, 2):
                        errors.append(f"Unexpected means length: {len(means)}")
                except Exception as exc:
                    errors.append(str(exc))

        t1 = threading.Thread(target=_alternating_setter)
        t2 = threading.Thread(target=_length_checker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Atomicity violation: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# build_pipeline dispatch
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildPipelineDispatch:

    def test_raises_runtime_error_when_dali_unavailable(self):
        """build_pipeline must surface a clear error when nvidia-dali is absent."""
        from unittest.mock import patch

        with patch("dino_loader.pipeline.HAS_DALI", False):
            from dino_loader.augmentation import DinoV2AugSpec
            from dino_loader.pipeline import build_pipeline
            with pytest.raises(RuntimeError, match="nvidia-dali"):
                build_pipeline(
                    source         = MagicMock(),
                    aug_spec       = DinoV2AugSpec(aug_cfg=DINOAugConfig()),
                    batch_size     = 4,
                    num_threads    = 1,
                    device_id      = 0,
                    resolution_src = MagicMock(),
                )

    def test_raises_type_error_on_unknown_aug_spec(self):
        """An unrecognised AugmentationSpec subtype must raise TypeError."""
        from unittest.mock import patch

        # We need DALI to appear available so dispatch reaches the match statement.
        with patch("dino_loader.pipeline.HAS_DALI", True):
            from dino_loader.augmentation import AugmentationSpec
            from dino_loader.pipeline import build_pipeline

            # [FIX] Implement all 4 abstract methods so the class can be
            # instantiated.  The TypeError we want is from the match/case
            # dispatch inside build_pipeline, not from ABC instantiation.
            class _UnknownSpec(AugmentationSpec):
                @property
                def output_map(self) -> list[str]:
                    return ["view_0"]

                @property
                def norm_stats(self) -> NormStats:
                    return NormStats.imagenet()

                @property
                def initial_global_size(self) -> int:
                    return 224

                @property
                def initial_local_size(self) -> int:
                    return 96

                def split_views(self, views: list) -> tuple[list, list]:
                    return views, []

            with pytest.raises(TypeError, match="Unknown augmentation spec"):
                build_pipeline(
                    source         = MagicMock(),
                    aug_spec       = _UnknownSpec(),
                    batch_size     = 4,
                    num_threads    = 1,
                    device_id      = 0,
                    resolution_src = MagicMock(),
                )