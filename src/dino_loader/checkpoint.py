"""dino_loader.checkpoint
======================
Dataloader state checkpointing.

Design
------
- JSON (not pickle): stable across Python versions and environments.
- Atomic write: write to .tmp → rename() — POSIX rename is atomic.
- Rank 0 writes; all ranks can read.
- Retains only the 3 most recent checkpoints to bound Lustre usage.

[M3-FIX] SHA-256 integrity envelope: CheckpointState.save() wraps the payload
          in {"payload": {...}, "sha256": "<hex>"}. load() verifies the checksum
          before deserialising, raising ValueError on mismatch. Legacy flat-format
          checkpoints (no "payload" key) are still supported with a WARNING.

[CK-3]   LATEST pointer file for robust checkpoint discovery. Falls back to
          glob-sort for backward compat with older checkpoint directories.
"""

import logging
from pathlib import Path

from dino_loader.config import CheckpointState

log = logging.getLogger(__name__)

_KEEP_LAST   = 3
_LATEST_FILE = "LATEST"


class DataLoaderCheckpointer:
    """Manages JSON checkpoint files for DINODataLoader state.

    Writes are rank-0-only and throttled to every N steps.
    Reads are available to all ranks for resume.

    File format (envelope with SHA-256)::

        {
          "payload": {
            "step": 1000,
            "epoch": 5,
            "dataset_names": ["laion2b", "imagenet22k"],
            "mixing_weights": [0.7, 0.3],
            "global_crop_size": 224,
            "local_crop_size": 96
          },
          "sha256": "a3f2..."
        }

    Legacy flat format (no checksum) is still supported for backward compat.
    """

    def __init__(
        self,
        ckpt_dir:      str,
        every_n_steps: int = 500,
        rank:          int = 0,
    ) -> None:
        self._dir   = Path(ckpt_dir)
        self._every = every_n_steps
        self._rank  = rank
        if rank == 0:
            self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: CheckpointState) -> None:
        """Save state to disk (rank 0 only, every N steps).

        Write order:
        1. Atomic JSON write with SHA-256 envelope (tmp → rename).
        2. Atomic LATEST pointer update (tmp → rename).
        3. Prune old checkpoints (best-effort; does not affect LATEST).
        """
        if self._rank != 0 or state.step % self._every != 0:
            return

        filename = f"dl_state_{state.step:012d}.json"
        path     = self._dir / filename
        state.save(path)

        self._write_latest(filename)
        self._prune()

        log.info("DataLoader checkpoint saved: %s", filename)

    def load(self) -> CheckpointState | None:
        """Load the most recent checkpoint, or return None if none exists.

        Returns None (with a WARNING) if the checkpoint file is corrupt or
        fails the SHA-256 integrity check.
        """
        path = self._resolve_latest()
        if path is None:
            return None
        try:
            state = CheckpointState.load(path)
            log.info(
                "Resuming from %s (step=%d epoch=%d global=%d local=%d)",
                path.name, state.step, state.epoch,
                state.global_crop_size, state.local_crop_size,
            )
            return state
        except ValueError as exc:
            log.warning(
                "Checkpoint %s failed integrity check: %s — starting from scratch.",
                path, exc,
            )
            return None
        except Exception as exc:
            log.warning(
                "Could not load checkpoint %s: %s — starting from scratch.",
                path, exc,
            )
            return None

    def state_dict(self) -> dict:
        """Return checkpoint state as a plain dict."""
        state = self.load()
        if state is None:
            return {}
        return {
            "step":             state.step,
            "epoch":            state.epoch,
            "dataset_names":    state.dataset_names,
            "mixing_weights":   state.mixing_weights,
            "global_crop_size": state.global_crop_size,
            "local_crop_size":  state.local_crop_size,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from a dict produced by state_dict(). Caller applies fields."""

    def _write_latest(self, filename: str) -> None:
        """Atomically update the LATEST pointer file."""
        latest_tmp = self._dir / f"{_LATEST_FILE}.tmp"
        latest     = self._dir / _LATEST_FILE
        try:
            latest_tmp.write_text(filename, encoding="utf-8")
            latest_tmp.rename(latest)
        except Exception as exc:
            log.warning("Failed to write LATEST pointer: %s", exc)
            try:
                latest_tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _resolve_latest(self) -> Path | None:
        """Return the Path of the most recent checkpoint, or None."""
        latest_ptr = self._dir / _LATEST_FILE
        if latest_ptr.exists():
            try:
                filename  = latest_ptr.read_text(encoding="utf-8").strip()
                candidate = self._dir / filename
                if candidate.exists():
                    return candidate
                log.warning(
                    "LATEST pointer references non-existent file %s; "
                    "falling back to glob-sort.",
                    filename,
                )
            except Exception as exc:
                log.warning(
                    "Could not read LATEST pointer: %s; falling back to glob-sort.",
                    exc,
                )

        candidates = sorted(self._dir.glob("dl_state_*.json"))
        return candidates[-1] if candidates else None

    def _prune(self) -> None:
        """Keep only the _KEEP_LAST most recent checkpoint JSON files."""
        candidates = sorted(self._dir.glob("dl_state_*.json"))
        for old in candidates[:-_KEEP_LAST]:
            try:
                old.unlink()
                log.debug("Pruned old checkpoint: %s", old.name)
            except FileNotFoundError:
                pass
