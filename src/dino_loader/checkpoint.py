"""dino_loader.checkpoint
======================
Checkpointing de l'état du dataloader.

Design
------
- JSON (pas pickle) : stable entre versions Python et environnements.
- Écriture atomique : write vers ``.tmp`` → ``rename()`` — POSIX atomic.
- Rank 0 écrit ; tous les ranks peuvent lire.
- Retient uniquement les 3 checkpoints les plus récents.

Responsabilité centralisée
---------------------------
Toute la logique de sérialisation avec SHA-256, écriture atomique et
vérification d'intégrité vit ici.  ``CheckpointState`` dans ``config.py``
est un pur dataclass sans I/O.

Format d'enveloppe (SHA-256)::

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

Le format plat legacy (sans checksum) est encore supporté en lecture
pour la rétrocompatibilité.

Pointeur LATEST [CK-3]
------------------------
Un fichier ``LATEST`` pointe vers le checkpoint le plus récent pour une
découverte rapide.  Fallback sur glob-sort pour la compat ascendante.

Signature des fonctions I/O
-----------------------------
``save_checkpoint(path, state)`` et ``load_checkpoint(path)`` — la path
est toujours le premier argument pour cohérence avec les conventions
Python (pathlib, open, etc.).
"""

import contextlib
import hashlib
import json
import logging
from pathlib import Path

from dino_loader.config import CheckpointState

log = logging.getLogger(__name__)

_KEEP_LAST   = 3
_LATEST_FILE = "LATEST"


def _serialize_payload(payload: dict) -> str:
    """Sérialise un payload en JSON déterministe (sort_keys=True).

    ``sort_keys=True`` garantit un checksum reproductible indépendamment de
    l'ordre d'insertion des clés.
    """
    return json.dumps(payload, indent=2, sort_keys=True)


def save_checkpoint(path: Path, state: CheckpointState) -> None:
    """Écrit un checkpoint atomiquement avec une enveloppe SHA-256.

    Args:
        path:  Chemin de destination.
        state: État à persister.

    Raises:
        OSError: En cas d'échec d'écriture ou de rename.

    """
    payload      = state.to_dict()
    payload_json = _serialize_payload(payload)
    checksum     = hashlib.sha256(payload_json.encode()).hexdigest()
    envelope     = {"payload": payload, "sha256": checksum}
    tmp          = path.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(envelope, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.rename(path)
    except Exception:
        with contextlib.suppress(Exception):
            tmp.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path) -> CheckpointState:
    """Charge et vérifie un checkpoint depuis ``path``.

    Supporte le format enveloppe (avec SHA-256) et le format plat legacy.

    Args:
        path: Chemin vers le fichier JSON.

    Returns:
        Instance de ``CheckpointState``.

    Raises:
        ValueError: Si la vérification SHA-256 échoue.
        json.JSONDecodeError: Si le fichier n'est pas du JSON valide.

    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    if "payload" in raw and "sha256" in raw:
        payload_json = _serialize_payload(raw["payload"])
        expected     = hashlib.sha256(payload_json.encode()).hexdigest()
        if raw["sha256"] != expected:
            msg = (
                f"Checkpoint {path} failed integrity check: "
                f"stored sha256={raw['sha256']!r}, computed={expected!r}. "
                "File may be corrupt or truncated."
            )
            raise ValueError(msg)
        data = raw["payload"]
    else:
        log.warning(
            "Checkpoint %s uses legacy flat format (no SHA-256 envelope). "
            "Resave with the current code to upgrade the format.",
            path,
        )
        data = raw

    data.setdefault("global_crop_size", 224)
    data.setdefault("local_crop_size",  96)
    return CheckpointState.from_dict(data)


class DataLoaderCheckpointer:
    """Gère les fichiers JSON de checkpoint pour DINODataLoader.

    Les écritures sont réservées au rank 0 et limitées à tous les N steps.
    Les lectures sont disponibles pour tous les ranks lors de la reprise.

    Args:
        ckpt_dir:      Répertoire de stockage des checkpoints.
        every_n_steps: Fréquence d'écriture en nombre de steps.
        rank:          Rang global du processus appelant.

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
        """Sauvegarde l'état sur disque (rank 0 uniquement, tous les N steps).

        Ordre d'écriture :
        1. Écriture JSON atomique avec enveloppe SHA-256 (tmp → rename).
        2. Mise à jour atomique du pointeur LATEST (tmp → rename).
        3. Élagage des anciens checkpoints (best-effort).

        """
        if self._rank != 0 or state.step % self._every != 0:
            return

        filename = f"dl_state_{state.step:012d}.json"
        path     = self._dir / filename
        save_checkpoint(path, state)

        self._write_latest(filename)
        self._prune()

        log.info("DataLoader checkpoint saved: %s", filename)

    def load(self) -> CheckpointState | None:
        """Charge le checkpoint le plus récent, ou retourne None.

        Retourne None (avec un WARNING) si le fichier est corrompu ou si la
        vérification SHA-256 échoue.

        """
        path = self._resolve_latest()
        if path is None:
            return None
        try:
            state = load_checkpoint(path)
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
        """Retourne l'état du checkpoint en dict plat."""
        state = self.load()
        if state is None:
            return {}
        return state.to_dict()

    def load_state_dict(self, d: dict) -> None:
        """Restaure depuis un dict produit par ``state_dict()``."""

    def _write_latest(self, filename: str) -> None:
        """Met à jour le pointeur LATEST de façon atomique."""
        latest_tmp = self._dir / f"{_LATEST_FILE}.tmp"
        latest     = self._dir / _LATEST_FILE
        try:
            latest_tmp.write_text(filename, encoding="utf-8")
            latest_tmp.rename(latest)
        except Exception as exc:
            log.warning("Failed to write LATEST pointer: %s", exc)
            with contextlib.suppress(Exception):
                latest_tmp.unlink(missing_ok=True)

    def _resolve_latest(self) -> Path | None:
        """Retourne le Path du checkpoint le plus récent, ou None."""
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
        """Conserve uniquement les _KEEP_LAST checkpoints JSON les plus récents."""
        candidates = sorted(self._dir.glob("dl_state_*.json"))
        for old in candidates[:-_KEEP_LAST]:
            with contextlib.suppress(FileNotFoundError):
                old.unlink()
                log.debug("Pruned old checkpoint: %s", old.name)
