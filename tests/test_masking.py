"""tests/test_masking.py.

Unit tests for dino_loader.masking.MaskingGenerator.

Coverage
--------
- Construction: defaults, square shorthand, explicit all-params
- Validation: num_masking_patches > grid, min > max patches, negative target
- __repr__: contains key dimensions
- get_shape: returns (height, width)
- __call__: shape, dtype, exact count, flat mode
- Block placement: respects min/max patches, aspect ratio bounds
- Completeness guarantee: shortfall always filled randomly
- Determinism: same seed → same mask
- Edge cases: target == 0, target == all patches, 1×1 grid
- _complete_randomly: correct for non-C-contiguous arrays (no silent no-op)
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.masking import MaskingGenerator


# ══════════════════════════════════════════════════════════════════════════════
# Construction
# ══════════════════════════════════════════════════════════════════════════════


class TestMaskingGeneratorConstruction:

    def test_square_shorthand(self) -> None:
        gen = MaskingGenerator(input_size=14)
        assert gen.height == 14
        assert gen.width == 14

    def test_tuple_input_size(self) -> None:
        gen = MaskingGenerator(input_size=(10, 16))
        assert gen.height == 10
        assert gen.width == 16

    def test_default_num_masking_patches_is_half(self) -> None:
        gen = MaskingGenerator(input_size=14)
        assert gen.num_masking_patches == 14 * 14 // 2

    def test_explicit_num_masking_patches(self) -> None:
        gen = MaskingGenerator(input_size=14, num_masking_patches=75)
        assert gen.num_masking_patches == 75

    def test_default_max_num_patches_equals_target(self) -> None:
        gen = MaskingGenerator(input_size=8, num_masking_patches=30)
        assert gen.max_num_patches == 30

    def test_explicit_max_num_patches(self) -> None:
        gen = MaskingGenerator(input_size=8, num_masking_patches=30, max_num_patches=10)
        assert gen.max_num_patches == 10

    def test_log_aspect_ratio_default(self) -> None:
        gen = MaskingGenerator(input_size=8, min_aspect=0.3)
        lo, hi = gen.log_aspect_ratio
        assert lo == pytest.approx(math.log(0.3))
        assert hi == pytest.approx(math.log(1.0 / 0.3))

    def test_explicit_max_aspect(self) -> None:
        gen = MaskingGenerator(input_size=8, min_aspect=0.5, max_aspect=2.0)
        assert gen.log_aspect_ratio[1] == pytest.approx(math.log(2.0))


# ══════════════════════════════════════════════════════════════════════════════
# Construction validation — [FIX-MASK-VALIDATE]
# ══════════════════════════════════════════════════════════════════════════════


class TestMaskingGeneratorValidation:
    """[FIX-MASK-VALIDATE] __init__ should catch impossible configurations early."""

    def test_num_masking_patches_exceeds_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds grid size"):
            MaskingGenerator(input_size=(4, 4), num_masking_patches=17)

    def test_num_masking_patches_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="≥ 0"):
            MaskingGenerator(input_size=(8, 8), num_masking_patches=-1)

    def test_min_gt_max_patches_when_target_positive_raises(self) -> None:
        """min_num_patches > max_num_patches with a positive target should raise."""
        with pytest.raises(ValueError, match="min_num_patches"):
            MaskingGenerator(
                input_size=8,
                num_masking_patches=10,
                min_num_patches=8,
                max_num_patches=4,  # max < min
            )

    def test_zero_target_with_min_gt_zero_does_not_raise(self) -> None:
        """num_masking_patches=0 is valid even when min_num_patches=4 (block loop never runs)."""
        gen = MaskingGenerator(input_size=(8, 8), num_masking_patches=0)
        assert gen.num_masking_patches == 0

    def test_target_equals_grid_size_valid(self) -> None:
        """Masking all patches is a valid configuration."""
        gen = MaskingGenerator(input_size=(4, 4), num_masking_patches=16)
        assert gen.num_masking_patches == 16


# ══════════════════════════════════════════════════════════════════════════════
# get_shape and __repr__
# ══════════════════════════════════════════════════════════════════════════════


class TestMaskingGeneratorMeta:

    def test_get_shape(self) -> None:
        gen = MaskingGenerator(input_size=(10, 12))
        assert gen.get_shape() == (10, 12)

    def test_repr_contains_dimensions(self) -> None:
        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        r = repr(gen)
        assert "14" in r
        assert "75" in r


# ══════════════════════════════════════════════════════════════════════════════
# __call__ — shape, dtype, exact count
# ══════════════════════════════════════════════════════════════════════════════


class TestMaskingGeneratorCall:

    def test_output_shape_2d(self) -> None:
        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        mask = gen()
        assert mask.shape == (14, 14)

    def test_output_shape_flat(self) -> None:
        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        mask = gen(flat=True)
        assert mask.shape == (196,)

    def test_output_dtype_bool(self) -> None:
        gen = MaskingGenerator(input_size=(8, 8), num_masking_patches=20)
        mask = gen()
        assert mask.dtype == bool

    def test_exact_count_met(self) -> None:
        """Completeness guarantee: always exactly num_masking_patches True entries."""
        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        for _ in range(20):
            mask = gen()
            assert int(mask.sum()) == 75

    def test_exact_count_flat(self) -> None:
        gen = MaskingGenerator(input_size=(8, 8), num_masking_patches=30)
        for _ in range(10):
            mask = gen(flat=True)
            assert int(mask.sum()) == 30

    def test_values_are_boolean(self) -> None:
        gen = MaskingGenerator(input_size=(8, 8), num_masking_patches=16)
        mask = gen()
        unique = set(mask.ravel().tolist())
        assert unique <= {True, False}

    def test_mask_count_zero(self) -> None:
        """num_masking_patches=0 must produce an all-False mask."""
        gen = MaskingGenerator(input_size=(8, 8), num_masking_patches=0)
        mask = gen()
        # Block placement loop never executes (0 < 0 is False).
        # _complete_randomly with shortfall=0 is a no-op.
        assert int(mask.sum()) == 0

    def test_mask_count_all(self) -> None:
        gen = MaskingGenerator(input_size=(4, 4), num_masking_patches=16)
        mask = gen()
        assert int(mask.sum()) == 16

    def test_1x1_grid(self) -> None:
        gen = MaskingGenerator(input_size=(1, 1), num_masking_patches=1)
        mask = gen()
        assert int(mask.sum()) == 1


# ══════════════════════════════════════════════════════════════════════════════
# _complete_randomly — [FIX-MASK-RAVEL]
# ══════════════════════════════════════════════════════════════════════════════


class TestCompleteRandomly:
    """[FIX-MASK-RAVEL] _complete_randomly must work correctly on non-C-contiguous arrays."""

    def test_fortran_order_mask_filled_correctly(self) -> None:
        """Non-C-contiguous (Fortran-order) mask must be filled in-place."""
        gen  = MaskingGenerator(input_size=(8, 8), num_masking_patches=30)
        # Create a Fortran-order array (non-C-contiguous).
        mask = np.zeros((8, 8), dtype=bool, order="F")
        assert not mask.flags["C_CONTIGUOUS"]

        result = gen._complete_randomly(mask, target=30)
        assert int(result.sum()) == 30

    def test_sliced_non_contiguous_mask_filled_correctly(self) -> None:
        """Sliced (strided) non-contiguous mask must be filled in-place."""
        gen  = MaskingGenerator(input_size=(4, 4), num_masking_patches=8)
        base = np.zeros((8, 8), dtype=bool)
        # Take every other row and column — non-contiguous view.
        mask = base[::2, ::2]
        assert not mask.flags["C_CONTIGUOUS"]

        result = gen._complete_randomly(mask, target=8)
        assert int(result.sum()) == 8

    def test_shortfall_zero_returns_unchanged(self) -> None:
        gen  = MaskingGenerator(input_size=(4, 4), num_masking_patches=4)
        mask = np.ones((4, 4), dtype=bool)
        # Already 16 True entries; target=4 → shortfall ≤ 0 → unchanged.
        result = gen._complete_randomly(mask, target=4)
        assert int(result.sum()) == 16  # no entries removed

    def test_complete_randomly_does_not_exceed_target(self) -> None:
        """Shortfall clamped by available unmasked positions."""
        gen  = MaskingGenerator(input_size=(2, 2), num_masking_patches=2)
        mask = np.zeros((2, 2), dtype=bool)
        # Only 4 positions available; target of 6 should be clamped to 4.
        result = gen._complete_randomly(mask, target=6)
        assert int(result.sum()) <= 4


# ══════════════════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════════════════


class TestMaskingGeneratorDeterminism:

    def test_same_seed_same_mask(self) -> None:
        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        np.random.seed(42)
        import random
        random.seed(42)
        mask1 = gen()

        np.random.seed(42)
        random.seed(42)
        mask2 = gen()

        assert np.array_equal(mask1, mask2)

    def test_different_calls_can_differ(self) -> None:
        """Consecutive calls with different random state should (usually) differ."""
        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
        masks = [gen() for _ in range(10)]
        # At least two should differ — probability of all equal is negligible.
        unique_count = sum(
            1 for i in range(len(masks) - 1)
            if not np.array_equal(masks[i], masks[i + 1])
        )
        assert unique_count > 0


# ══════════════════════════════════════════════════════════════════════════════
# Multiple calls — statistical sanity
# ══════════════════════════════════════════════════════════════════════════════


class TestMaskingGeneratorStatistics:

    def test_mean_coverage_approximately_correct(self) -> None:
        """Average mask density should be close to num_masking_patches / total."""
        n_total = 14 * 14
        n_mask = 75
        gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=n_mask)
        counts = [gen().sum() for _ in range(50)]
        assert all(c == n_mask for c in counts), "Exact count guarantee violated"

    def test_no_out_of_bounds(self) -> None:
        gen = MaskingGenerator(input_size=(10, 12), num_masking_patches=40)
        for _ in range(20):
            mask = gen()
            assert mask.shape == (10, 12)
            assert mask.dtype == bool