"""
dino_loader.checkpoint
======================
Dataloader state checkpointing.

Design choices (unchanged)
--------------------------
- JSON (not pickle): stable across Python versions and environments.
- Atomic write: write to .tmp → rename() — POSIX rename is atomic.
- Rank 0 writes; all ranks can read.
- Retains only the 3 most recent checkpoints to bound Lustre usage.
- Fast-forward is a lightweight counter skip.

Changes vs previous version
----------------------------
[CK-1]  Supports new CheckpointState fields: global_crop_size, local_crop_size.
        Fully backward-compatible: missing fields default to 224 / 96 via
        dataclass defaults in CheckpointState.

[CK-2]  load() returns None gracefully when no checkpoint exists, and logs
        a WARNING (not ERROR) when a checkpoint file is corrupt so that the
        training job can start fresh rather than crash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from dino_loader.config import CheckpointState

log = logging.getLogger(__name__)

_KEEP_LAST = 3


class DataLoaderCheckpointer:
    """
    Manages JSON checkpoint files for DINODataLoader state.

    Writes are rank-0-only and throttled to every N steps.
    Reads are available to all ranks for resume.
    """

    def __init__(self, ckpt_dir: str, every_n_steps: int = 500, rank: int = 0) -> None:
        self._dir    = Path(ckpt_dir)
        self._every  = every_n_steps
        self._rank   = rank
        if rank == 0:
            self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: CheckpointState) -> None:
        """Save state to disk (rank 0 only, every N steps)."""
        if self._rank != 0 or state.step % self._every != 0:
            return
        path = self._dir / f"dl_state_{state.step:012d}.json"
        state.save(path)
        self._prune()
        log.info("DataLoader checkpoint saved: %s", path.name)

    def load(self) -> Optional[CheckpointState]:
        """Load the most recent checkpoint, or return None if none exists."""
        candidates = sorted(self._dir.glob("dl_state_*.json"))
        if not candidates:
            return None
        try:
            state = CheckpointState.load(candidates[-1])
            log.info(
                "Resuming from %s (step=%d epoch=%d global=%d local=%d)",
                candidates[-1].name,
                state.step, state.epoch,
                state.global_crop_size, state.local_crop_size,
            )
            return state
        except Exception as exc:
            # [CK-2] Warning, not error — training continues from scratch.
            log.warning(
                "Could not load checkpoint %s: %s — starting from scratch.",
                candidates[-1], exc,
            )
            return None

    def _prune(self) -> None:
        candidates = sorted(self._dir.glob("dl_state_*.json"))
        for old in candidates[:-_KEEP_LAST]:
            try:
                old.unlink()
            except FileNotFoundError:
                pass
