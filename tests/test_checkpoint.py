"""tests/test_checkpoint.py
===========================
Tests unitaires pour :class:`dino_loader.checkpoint.DataLoaderCheckpointer`
et les fonctions ``save_checkpoint`` / ``load_checkpoint``.

Convention de signature
-----------------------
``save_checkpoint(path, state)`` — la path est toujours le premier argument,
cohérent avec les conventions Python standard (pathlib, open, json.dump, etc.).

Coverage
--------
save_checkpoint / load_checkpoint
- Round-trip SHA-256 (envelope format)
- Payload modifié → ValueError (integrity check)
- Format plat legacy (pas d'envelope) → chargement + WARNING
- Champs manquants (crop sizes) → valeurs par défaut 224/96
- Clés inconnues ignorées silencieusement
- Écriture atomique : pas de fichier .tmp après succès

DataLoaderCheckpointer.save
- Crée le fichier, encode le step dans le nom
- No-op sur step non-multiple
- No-op si rank != 0
- Crée le répertoire automatiquement

DataLoaderCheckpointer.load
- Retourne le plus récent, None si vide
- Warning + None sur fichier corrompu
- Warning + None sur SHA-256 invalide

Pointeur LATEST [CK-3]
- Créé au premier save, mis à jour, survit au pruning
- Fallback glob-sort quand LATEST est absent ou stale
- Nettoyage du .tmp si l'écriture de LATEST échoue

Pruning
- Retient exactement les 3 derniers checkpoints
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dino_loader.checkpoint import (
    DataLoaderCheckpointer,
    _KEEP_LAST,
    _LATEST_FILE,
    load_checkpoint,
    save_checkpoint,
)
from dino_loader.config import CheckpointState


def _state(step: int = 100, epoch: int = 1) -> CheckpointState:
    return CheckpointState(
        step             = step,
        epoch            = epoch,
        dataset_names    = ["laion", "imagenet"],
        mixing_weights   = [0.7, 0.3],
        global_crop_size = 224,
        local_crop_size  = 96,
    )


def _write_flat_json(path: Path, data: dict) -> None:
    """Écrit un checkpoint legacy sans envelope SHA-256 (simule l'ancienne version)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)


# ══════════════════════════════════════════════════════════════════════════════
# save_checkpoint / load_checkpoint — fonctions de bas niveau
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveLoadCheckpoint:

    def test_round_trip_step(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_checkpoint(path, _state(step=42))
        assert load_checkpoint(path).step == 42

    def test_round_trip_epoch(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_checkpoint(path, _state(epoch=7))
        assert load_checkpoint(path).epoch == 7

    def test_round_trip_crop_sizes(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_checkpoint(path, _state())
        loaded = load_checkpoint(path)
        assert loaded.global_crop_size == 224
        assert loaded.local_crop_size  == 96

    def test_envelope_has_sha256(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_checkpoint(path, _state())
        raw = json.loads(path.read_text())
        assert "payload" in raw
        assert "sha256" in raw
        assert len(raw["sha256"]) == 64

    def test_tampered_payload_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_checkpoint(path, _state())
        raw = json.loads(path.read_text())
        raw["payload"]["step"] = 9999
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="integrity check"):
            load_checkpoint(path)

    def test_atomic_write_no_tmp_on_success(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        save_checkpoint(path, _state())
        assert not (tmp_path / "state.tmp").exists()
        assert path.exists()

    def test_legacy_flat_format_loads_with_warning(self, tmp_path: Path, caplog) -> None:
        import logging
        path = tmp_path / "old.json"
        _write_flat_json(path, {
            "step": 10, "epoch": 0,
            "dataset_names": ["laion"], "mixing_weights": [1.0],
        })
        with caplog.at_level(logging.WARNING):
            state = load_checkpoint(path)
        assert state.step == 10

    def test_legacy_missing_crop_sizes_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "old_no_sizes.json"
        _write_flat_json(path, {
            "step": 50, "epoch": 1,
            "dataset_names": ["ds"], "mixing_weights": [1.0],
        })
        state = load_checkpoint(path)
        assert state.global_crop_size == 224
        assert state.local_crop_size  == 96

    def test_legacy_unknown_keys_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "old_extra.json"
        _write_flat_json(path, {
            "step": 1, "epoch": 0,
            "dataset_names": ["ds"], "mixing_weights": [1.0],
            "unknown_future_field": "ignored",
        })
        state = load_checkpoint(path)
        assert state.step == 1

    def test_path_is_first_argument(self, tmp_path: Path) -> None:
        """Canonical signature: save_checkpoint(path, state) — path first."""
        path  = tmp_path / "sig.json"
        state = _state(step=55)
        save_checkpoint(path, state)
        assert load_checkpoint(path).step == 55


# ══════════════════════════════════════════════════════════════════════════════
# DataLoaderCheckpointer.save
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointerSave:

    def test_creates_file(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=10))
        assert len(list(tmp_path.glob("dl_state_*.json"))) == 1

    def test_filename_encodes_step(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=42))
        files = list(tmp_path.glob("dl_state_*.json"))
        assert "000000000042" in files[0].name

    def test_noop_on_non_multiple(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=10, rank=0)
        ckpt.save(_state(step=7))
        assert len(list(tmp_path.glob("dl_state_*.json"))) == 0

    def test_noop_for_non_rank0(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=1)
        ckpt.save(_state(step=10))
        assert len(list(tmp_path.glob("dl_state_*.json"))) == 0

    def test_directory_created_automatically(self, tmp_path: Path) -> None:
        nested = str(tmp_path / "a" / "b" / "c")
        ckpt   = DataLoaderCheckpointer(nested, every_n_steps=1, rank=0)
        ckpt.save(_state(step=1))
        assert Path(nested).is_dir()


# ══════════════════════════════════════════════════════════════════════════════
# DataLoaderCheckpointer.load
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointerLoad:

    def test_returns_latest_by_step(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in (10, 20, 30):
            ckpt.save(_state(step=step))
        loaded = ckpt.load()
        assert loaded is not None and loaded.step == 30

    def test_returns_none_when_no_files(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        assert ckpt.load() is None

    def test_warns_and_returns_none_on_corrupt_json(self, tmp_path: Path, caplog) -> None:
        import logging
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        (tmp_path / "dl_state_000000000001.json").write_text("NOT JSON {{{")
        with caplog.at_level(logging.WARNING):
            result = ckpt.load()
        assert result is None

    def test_warns_and_returns_none_on_tampered_sha256(self, tmp_path: Path, caplog) -> None:
        import logging
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=10))
        ck_file = list(tmp_path.glob("dl_state_*.json"))[0]
        raw = json.loads(ck_file.read_text())
        raw["payload"]["step"] = 9999
        ck_file.write_text(json.dumps(raw))
        with caplog.at_level(logging.WARNING):
            result = ckpt.load()
        assert result is None
        assert any("integrity" in r.message.lower() for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════════
# Pointeur LATEST [CK-3]
# ══════════════════════════════════════════════════════════════════════════════


class TestLatestPointer:

    def test_latest_file_created_on_first_save(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        ckpt.save(_state(step=10))
        assert (tmp_path / _LATEST_FILE).exists()

    def test_latest_points_to_most_recent_save(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in (10, 20, 30):
            ckpt.save(_state(step=step))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert "000000000030" in latest_name

    def test_load_uses_latest_pointer(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in (10, 20, 30):
            ckpt.save(_state(step=step))
        loaded = ckpt.load()
        assert loaded is not None and loaded.step == 30

    def test_latest_survives_pruning(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        n = _KEEP_LAST + 2
        for step in range(1, n + 1):
            ckpt.save(_state(step=step))
        latest_name = (tmp_path / _LATEST_FILE).read_text().strip()
        assert f"{n:012d}" in latest_name

    def test_fallback_to_glob_when_no_latest(self, tmp_path: Path) -> None:
        path = tmp_path / "dl_state_000000000050.json"
        save_checkpoint(path, _state(step=50))
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        loaded = ckpt.load()
        assert loaded is not None and loaded.step == 50

    def test_latest_tmp_cleaned_up_on_write_failure(self, tmp_path: Path) -> None:
        """LATEST .tmp must be removed even when the rename to LATEST fails.

        [FIX] The previous test patched Path.rename globally, which also broke
        the JSON checkpoint write (save_checkpoint also calls rename).
        We now patch only _write_latest so save_checkpoint succeeds first
        and only the LATEST pointer write fails.
        """
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)

        original_write_latest = ckpt._write_latest

        def _bad_write_latest(filename: str) -> None:
            # Write the .tmp but then raise to simulate disk-full on rename.
            latest_tmp = ckpt._dir / f"{_LATEST_FILE}.tmp"
            latest_tmp.write_text(filename)
            raise OSError("simulated disk full on LATEST rename")

        with patch.object(ckpt, "_write_latest", _bad_write_latest):
            # save_checkpoint itself succeeds; only _write_latest fails.
            ckpt.save(_state(step=5))

        # The .tmp for LATEST must have been cleaned up despite the error.
        assert not (tmp_path / f"{_LATEST_FILE}.tmp").exists()
        # The actual checkpoint JSON must still exist.
        assert len(list(tmp_path.glob("dl_state_*.json"))) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Pruning
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointerPruning:

    def test_prune_keeps_only_three(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in [10, 20, 30, 40, 50]:
            ckpt.save(_state(step=step))
        files = sorted(tmp_path.glob("dl_state_*.json"))
        assert len(files) == 3

    def test_prune_retains_most_recent_three(self, tmp_path: Path) -> None:
        ckpt = DataLoaderCheckpointer(str(tmp_path), every_n_steps=1, rank=0)
        for step in [10, 20, 30, 40, 50]:
            ckpt.save(_state(step=step))
        files = sorted(tmp_path.glob("dl_state_*.json"))
        steps = [int(f.stem.split("_")[-1]) for f in files]
        assert max(steps) == 50
        assert min(steps) == 30


# ══════════════════════════════════════════════════════════════════════════════
# CheckpointState — pur dataclass (sans méthodes I/O)
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointStateIsDataclass:

    def test_has_no_save_method(self) -> None:
        assert not hasattr(CheckpointState, "save"), (
            "CheckpointState.save() a été trouvé — la logique I/O doit vivre "
            "dans DataLoaderCheckpointer / save_checkpoint()."
        )

    def test_has_no_load_method(self) -> None:
        assert not hasattr(CheckpointState, "load"), (
            "CheckpointState.load() a été trouvé — utiliser load_checkpoint() "
            "depuis dino_loader.checkpoint."
        )

    def test_to_dict_round_trip(self) -> None:
        state  = _state(step=99, epoch=3)
        d      = state.to_dict()
        state2 = CheckpointState.from_dict(d)
        assert state2.step  == 99
        assert state2.epoch == 3

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {
            "step": 1, "epoch": 0,
            "dataset_names": ["ds"], "mixing_weights": [1.0],
            "unknown_future_field": "ignored",
        }
        state = CheckpointState.from_dict(d)
        assert state.step == 1