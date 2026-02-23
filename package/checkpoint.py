"""
dino_loader.checkpoint
======================
Dataloader state checkpointing.

Design choices
--------------
- JSON (not pickle): stable across Python versions and environments.
  Critical on HPC clusters where nodes may run different conda envs.
- Atomic write: write to .tmp, then rename() — POSIX rename is atomic.
- Rank 0 writes; all ranks can read.
- Retains only the 3 most recent checkpoints to bound Lustre usage.
- Fast-forward is a lightweight counter skip (no re-processing of data).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from dino_loader.config import CheckpointState

log = logging.getLogger(__name__)

_KEEP_LAST = 3


class DataLoaderCheckpointer:
    def __init__(self, ckpt_dir: str, every_n_steps: int = 500, rank: int = 0):
        self._dir        = Path(ckpt_dir)
        self._every      = every_n_steps
        self._rank       = rank
        if rank == 0:
            self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: CheckpointState) -> None:
        if self._rank != 0 or state.step % self._every != 0:
            return
        path = self._dir / f"dl_state_{state.step:012d}.json"
        state.save(path)
        self._prune()
        log.info("DataLoader checkpoint saved: %s", path.name)

    def load(self) -> Optional[CheckpointState]:
        candidates = sorted(self._dir.glob("dl_state_*.json"))
        if not candidates:
            return None
        try:
            state = CheckpointState.load(candidates[-1])
            log.info("Resuming from %s (step %d, epoch %d)",
                     candidates[-1].name, state.step, state.epoch)
            return state
        except Exception as exc:
            log.warning("Could not load checkpoint %s: %s", candidates[-1], exc)
            return None

    def _prune(self) -> None:
        candidates = sorted(self._dir.glob("dl_state_*.json"))
        for old in candidates[:-_KEEP_LAST]:
            try:
                old.unlink()
            except FileNotFoundError:
                pass
