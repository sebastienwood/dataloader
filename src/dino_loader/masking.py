"""dino_loader.masking.

iBOT-style block-masked patch generator for Vision Transformer training.

This module is intentionally dependency-free: no DALI, no torch.distributed,
no CUDA.  It operates on numpy arrays and can be imported in any environment.

The ``MaskingGenerator`` produces boolean masks over a 2-D grid of ViT patch
tokens.  It fills the grid with randomly-placed rectangular blocks until the
requested number of masked patches is reached, then completes any shortfall by
randomly sampling from the remaining unmasked positions.

The ``MaskMapNode`` in ``dino_loader.nodes`` wraps this generator as a
``torchdata.nodes.BaseNode`` so that masking can be composed into the
torchdata graph alongside shard reading and augmentation.

Typical usage (standalone)
--------------------------
::

    from dino_loader.masking import MaskingGenerator

    gen = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
    mask = gen()          # numpy bool array, shape (14, 14)
    mask_flat = gen(flat=True)  # shape (196,)

Typical usage (torchdata graph)
--------------------------------
::

    from dino_loader.nodes import MaskMapNode
    from dino_loader.masking import MaskingGenerator

    gen    = MaskingGenerator(input_size=(14, 14), num_masking_patches=75)
    reader = ShardReaderNode(...)
    masked = MaskMapNode(reader, mask_generator=gen, patch_size=14, img_size=224)

Corrections
-----------
[FIX-MASK-VALIDATE] __init__ now validates num_masking_patches ≤ num_patches
    and min_num_patches ≤ max_num_patches, raising ValueError early rather than
    allowing random.uniform(min, max) to crash with a silent bad state when
    max < min (e.g. num_masking_patches=0 with min_num_patches=4).
[FIX-MASK-RAVEL] _complete_randomly uses np.ravel_multi_index / flat indexing
    in a way that guarantees a view (not a copy) when the mask is C-contiguous,
    with an explicit writeback when it is not, to avoid silent no-ops.
"""

import math
import random

import numpy as np

__all__ = ["MaskingGenerator"]

# Maximum number of rectangle-placement attempts per block before giving up.
_MAX_ATTEMPTS_PER_BLOCK: int = 10


class MaskingGenerator:
    """Generates iBOT-style random block masks over a ViT patch grid.

    Masks are produced by placing randomly-sized axis-aligned rectangles on a
    2-D boolean grid until ``num_masking_patches`` tokens are masked.  Any
    remaining shortfall is filled by uniform random sampling from the unmasked
    positions, guaranteeing the exact count is always met.

    Args:
        input_size: ``(height, width)`` of the patch grid.  A single integer
            is broadcast to a square grid.
        num_masking_patches: Target number of masked tokens.  Defaults to
            ``height * width // 2`` (50 %).
        min_num_patches: Minimum area (in patches) for each placed rectangle.
        max_num_patches: Maximum area for each rectangle.  Defaults to
            ``num_masking_patches``.
        min_aspect: Minimum rectangle aspect ratio.
        max_aspect: Maximum rectangle aspect ratio.  Defaults to
            ``1 / min_aspect``.

    Raises:
        ValueError: If ``num_masking_patches`` exceeds the grid size, or if
            ``min_num_patches > max_num_patches`` (would make block placement
            impossible).

    """

    def __init__(
        self,
        input_size: int | tuple[int, int],
        num_masking_patches: int | None = None,
        min_num_patches: int = 4,
        max_num_patches: int | None = None,
        min_aspect: float = 0.3,
        max_aspect: float | None = None,
    ) -> None:
        """Initialise the generator and validate all constraints."""
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.height, self.width = input_size
        self.num_patches = self.height * self.width

        if num_masking_patches is None:
            num_masking_patches = self.num_patches // 2
        self.num_masking_patches = num_masking_patches

        # [FIX-MASK-VALIDATE] Validate target is achievable.
        if self.num_masking_patches > self.num_patches:
            msg = (
                f"MaskingGenerator: num_masking_patches={self.num_masking_patches} "
                f"exceeds grid size ({self.height}×{self.width}={self.num_patches})."
            )
            raise ValueError(msg)
        if self.num_masking_patches < 0:
            msg = (
                f"MaskingGenerator: num_masking_patches must be ≥ 0, "
                f"got {self.num_masking_patches}."
            )
            raise ValueError(msg)

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        # [FIX-MASK-VALIDATE] When num_masking_patches=0, max_num_patches is 0
        # which is < min_num_patches=4.  Block placement never runs (the while
        # loop condition is false from the start), so this is safe — but we
        # validate only when block placement would actually be attempted.
        if self.num_masking_patches > 0 and self.min_num_patches > self.max_num_patches:
            msg = (
                f"MaskingGenerator: min_num_patches={self.min_num_patches} > "
                f"max_num_patches={self.max_num_patches}. "
                "Block placement would always fail. "
                "Reduce min_num_patches or increase max_num_patches."
            )
            raise ValueError(msg)

        effective_max_aspect = max_aspect if max_aspect is not None else 1.0 / min_aspect
        self.log_aspect_ratio = (
            math.log(min_aspect),
            math.log(effective_max_aspect),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, flat: bool = False) -> np.ndarray:
        """Generate one mask.

        Args:
            flat: When ``True``, return a 1-D array of shape
                ``(height * width,)``.  When ``False`` (default), return a 2-D
                array of shape ``(height, width)``.

        Returns:
            Boolean numpy array.  ``True`` indicates a masked (hidden) token.

        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask_count = 0

        while mask_count < self.num_masking_patches:
            remaining = self.num_masking_patches - mask_count
            cap = min(remaining, self.max_num_patches)
            delta = self._place_block(mask, cap)
            if delta == 0:
                break
            mask_count += delta

        mask = self._complete_randomly(mask, self.num_masking_patches)
        return mask.ravel() if flat else mask

    def get_shape(self) -> tuple[int, int]:
        """Return ``(height, width)`` of the patch grid."""
        return self.height, self.width

    def __repr__(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"MaskingGenerator("
            f"{self.height}x{self.width} → "
            f"[{self.min_num_patches}~{self.max_num_patches}] "
            f"target={self.num_masking_patches}, "
            f"log_aspect=[{self.log_aspect_ratio[0]:.3f}, {self.log_aspect_ratio[1]:.3f}]"
            f")"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _place_block(self, mask: np.ndarray, max_patches: int) -> int:
        """Attempt to place one rectangle on *mask*.

        Tries up to ``_MAX_ATTEMPTS_PER_BLOCK`` random placements.  Returns
        the number of newly masked patches (0 if no valid placement was found).

        Args:
            mask: Current boolean mask (mutated in place on success).
            max_patches: Upper bound on the area of the placed rectangle.

        Returns:
            Number of newly masked patches (0 on failure).

        """
        for _ in range(_MAX_ATTEMPTS_PER_BLOCK):
            target_area = random.uniform(self.min_num_patches, max_patches)
            log_ratio = random.uniform(*self.log_aspect_ratio)
            aspect = math.exp(log_ratio)

            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))

            if w >= self.width or h >= self.height:
                continue

            top  = random.randint(0, self.height - h)
            left = random.randint(0, self.width - w)

            block = mask[top: top + h, left: left + w]
            already_masked = int(block.sum())
            new_patches = h * w - already_masked

            # Accept only if the block contributes new patches within the cap.
            if 0 < new_patches <= max_patches:
                mask[top: top + h, left: left + w] = True
                return new_patches

        return 0

    @staticmethod
    def _complete_randomly(mask: np.ndarray, target: int) -> np.ndarray:
        """Fill any shortfall by randomly sampling from unmasked positions.

        This guarantees the output always has exactly *target* True entries
        even when block placement terminates early.

        [FIX-MASK-RAVEL] Uses np.ndarray.flat (a flatiter) for in-place writes
        that work correctly regardless of memory layout (C-contiguous or not),
        avoiding the subtle bug where ravel() returns a copy for non-contiguous
        arrays and the write is silently discarded.

        Args:
            mask: Partially filled boolean mask.
            target: Desired total number of True entries.

        Returns:
            Completed boolean mask (same object, modified in place).

        """
        current   = int(mask.sum())
        shortfall = target - current

        if shortfall <= 0:
            return mask

        # np.where on the flat view to get 1-D indices directly.
        flat_view   = mask.ravel()
        unmasked    = np.where(~flat_view)[0]
        shortfall   = min(shortfall, len(unmasked))
        chosen      = np.random.choice(unmasked, size=shortfall, replace=False)

        # [FIX-MASK-RAVEL] Write through mask.flat to guarantee in-place
        # mutation even when the array is not C-contiguous.  For C-contiguous
        # arrays (the common case) this is equivalent to flat_view[chosen]=True
        # but avoids the copy-on-ravel footgun for non-contiguous inputs.
        mask.flat[chosen] = True
        return mask
