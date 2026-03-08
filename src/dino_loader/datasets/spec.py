"""
dino_loader.datasets.spec
=========================
:class:`DatasetSpec` — the canonical descriptor of one WebDataset source.

Changes in this version
-----------------------
[SPEC-V] **schema_version + migration helpers**

    ``DatasetSpec`` now carries a ``schema_version`` integer (current: 1).
    This decouples the on-disk checkpoint format from the in-memory API:
    when a future breaking change is made to the spec fields, a migration
    function can be registered via ``register_spec_migration()`` and will
    be called automatically by ``DatasetSpec.from_dict()``.

    Backward compatibility is maintained: old checkpoints without
    ``schema_version`` are treated as version 0 and upgraded automatically.

[SPEC-V2] **Rich eager validation with actionable messages**

    All validation now runs in ``__post_init__`` and raises ``ValueError``
    with a precise human-readable message that includes:
    - the field name and the offending value
    - the expected range or set
    - a suggestion for how to fix the issue

    Additional validators added:
    - ``name`` must be a non-empty identifier string (no path separators)
    - ``mean`` / ``std`` must be 3-tuples in ``[0, 1]``
    - ``shard_sampling`` must be one of the declared literals
    - All shard paths are checked to be non-empty strings

[SPEC-V3] **``to_dict()`` / ``from_dict()`` round-trip**

    Enables JSON serialisation without depending on ``dataclasses.asdict``
    (which does not handle ``Optional[Tuple]`` cleanly across Python versions).
    Used by :class:`~dino_loader.checkpoint.DataLoaderCheckpointer`.

Why this module exists
----------------------
``DatasetSpec`` is the primary data contract of the ``datasets`` sub-system:
it is produced by :meth:`~dino_loader.datasets.dataset.Dataset.to_spec`,
consumed by :class:`~dino_loader.datasets.stub_gen` when generating IDE stubs,
and referenced throughout the ``hub/`` generated package.

Keeping it in ``dino_loader.config`` — alongside loader-level dataclasses such
as ``LoaderConfig`` and ``DINOAugConfig`` — created an upward dependency that
prevented ``dino_loader.datasets`` from operating as a self-contained,
independently-importable sub-package (e.g. for dataset cataloguing tools that
have no interest in DALI or CUDA).

``DatasetSpec`` is now the **sole** export of this module.
``dino_loader.config`` re-exports it transparently so all existing imports
remain valid without modification.

Backward compatibility
----------------------
::

    # All continue to work:
    from dino_loader.datasets import DatasetSpec      # canonical
    from dino_loader.datasets.spec import DatasetSpec  # canonical
    from dino_loader.config import DatasetSpec          # shim (unchanged API)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple


# ── Schema versioning ─────────────────────────────────────────────────────────

#: Current schema version.  Bump when adding required fields or renaming keys.
CURRENT_SCHEMA_VERSION: int = 1

#: Registry of migration functions: ``_MIGRATIONS[from_version]`` upgrades a
#: raw dict from version ``from_version`` to ``from_version + 1``.
_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def register_spec_migration(
    from_version: int,
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    """
    Register a migration function that upgrades a ``DatasetSpec`` dict from
    ``from_version`` to ``from_version + 1``.

    Call this once at module import time in the module that introduces the
    breaking change.  Migrations are applied in order by ``DatasetSpec.from_dict``.

    Parameters
    ----------
    from_version
        The schema version this migration upgrades *from*.
    fn
        A function ``(raw_dict) → upgraded_dict``.  Must not mutate the input
        in place — return a new dict.

    Example::

        # In a hypothetical future change that renames "weight" → "mixing_weight":
        def _v1_to_v2(d):
            d = dict(d)
            d["mixing_weight"] = d.pop("weight", 1.0)
            return d

        register_spec_migration(from_version=1, fn=_v1_to_v2)
    """
    if from_version in _MIGRATIONS:
        raise ValueError(
            f"A migration for schema version {from_version} → {from_version + 1} "
            "is already registered.  Each version may only have one migration."
        )
    _MIGRATIONS[from_version] = fn


# ── DatasetSpec ───────────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    """
    One WebDataset source with mixing weight, optional quality metadata,
    and discovery metadata populated by :meth:`Dataset.to_spec
    <dino_loader.datasets.dataset.Dataset.to_spec>`.

    Parameters
    ----------
    name
        Human-readable identifier, used in logs and checkpoint state.
        Must be a non-empty string without ``/`` or ``\\`` characters.
    shards
        List of absolute shard paths (``.tar`` files on Lustre).
        Must be non-empty.  Each entry must be a non-empty string.
    weight
        Initial mixing weight (re-normalised automatically; need not sum to 1).
        Must be ≥ 0.
    prob
        Alias for ``weight=`` to align with the ``wds.RandomMix`` API.  If
        both are provided, ``weight=`` takes precedence and a
        :class:`DeprecationWarning` is emitted.
    shard_sampling
        How shards are sampled within this dataset:

        ``"epoch"`` (default)
            One full, deterministic-shuffled pass per epoch.

        ``"resampled"``
            Infinite with-replacement sampling via ``wds.ResampledShards``.
            Use for small curated sets you want to over-sample, or for
            streaming datasets without epoch boundaries.

    shard_quality_scores
        Optional per-shard quality score ≥ 0.  When provided,
        :class:`~dino_loader.mixing_source.ShardIterator` samples shards
        proportionally to these scores rather than uniformly.  Scores are
        re-normalised internally.  Length must match ``len(shards)`` if
        provided.
    min_sample_quality
        Hard filter: samples whose ``.json`` sidecar ``quality_score`` field
        is below this threshold are discarded before entering the augmentation
        pipeline.  Must be in ``[0, 1]``.  Set to ``None`` to disable (default).
    metadata_key
        WebDataset sidecar extension to extract alongside ``.jpg`` files.
        Set to ``None`` to skip sidecar extraction (legacy behaviour, faster).
    mean
        Per-channel normalisation mean for this dataset (3-tuple, values in
        ``[0, 1]``).  When ``None``, the global
        :attr:`~dino_loader.config.DINOAugConfig.mean` is used (ImageNet stats).
    std
        Per-channel normalisation std for this dataset (3-tuple, values in
        ``(0, 1]``).  When ``None``, the global std is used.
    schema_version
        Internal version counter used for checkpoint migration.  Do not set
        manually — it is managed by :meth:`to_dict` / :meth:`from_dict`.
    confidentialities / modalities / splits / strategies
        Discovery metadata populated by
        :meth:`~dino_loader.datasets.dataset.Dataset.to_spec`.
        Informational only — not used by the loader itself.
    """

    name:   str
    shards: List[str]
    weight: float = 1.0
    prob:   Optional[float] = None  # [CFG-S2] wds.RandomMix alias

    shard_sampling:       Literal["epoch", "resampled"] = "epoch"
    shard_quality_scores: Optional[List[float]]         = None
    min_sample_quality:   Optional[float]               = None
    metadata_key:         Optional[str]                 = "json"
    mean:                 Optional[Tuple[float, float, float]] = None
    std:                  Optional[Tuple[float, float, float]] = None

    # [SPEC-V] Schema version — managed by to_dict/from_dict
    schema_version: int = field(default=CURRENT_SCHEMA_VERSION, repr=False)

    # Discovery metadata (populated by Dataset.to_spec — informational only)
    confidentialities: List[str] = field(default_factory=list)
    modalities:        List[str] = field(default_factory=list)
    splits:            List[str] = field(default_factory=list)
    strategies:        List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # ── [CFG-S2] prob= alias ──────────────────────────────────────────────
        if self.prob is not None and self.weight == 1.0:
            self.weight = self.prob
        elif self.prob is not None:
            warnings.warn(
                "DatasetSpec: both 'weight' and 'prob' provided; "
                "'weight' takes precedence.  'prob' is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )

        # ── [SPEC-V2] Rich validation ─────────────────────────────────────────
        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """
        Run all field validators eagerly.

        Raises ``ValueError`` with a precise, actionable message for every
        detected error.  Stops at the first error so the message is focused.
        """
        # name
        if not self.name:
            raise ValueError(
                "DatasetSpec: 'name' must be a non-empty string.  "
                "Got an empty string."
            )
        if "/" in self.name or "\\" in self.name:
            raise ValueError(
                f"DatasetSpec '{self.name}': 'name' must not contain path "
                f"separators ('/' or '\\\\').  Got: {self.name!r}.  "
                "Use a simple identifier like 'laion2b' or 'imagenet_22k'."
            )

        # shards
        if not self.shards:
            raise ValueError(
                f"DatasetSpec '{self.name}': 'shards' must not be empty.  "
                "Provide at least one .tar shard path."
            )
        bad_shards = [s for s in self.shards if not isinstance(s, str) or not s]
        if bad_shards:
            raise ValueError(
                f"DatasetSpec '{self.name}': all shards must be non-empty strings.  "
                f"Found {len(bad_shards)} invalid entry/entries: {bad_shards[:3]}"
                f"{'...' if len(bad_shards) > 3 else ''}."
            )

        # weight
        if self.weight < 0.0:
            raise ValueError(
                f"DatasetSpec '{self.name}': 'weight' must be ≥ 0, "
                f"got {self.weight}.  Use weight=0 to disable a dataset "
                "without removing it from the spec list."
            )

        # shard_sampling
        valid_sampling = {"epoch", "resampled"}
        if self.shard_sampling not in valid_sampling:
            raise ValueError(
                f"DatasetSpec '{self.name}': 'shard_sampling' must be one of "
                f"{sorted(valid_sampling)}, got {self.shard_sampling!r}.  "
                "Use 'resampled' for small curated datasets you want to "
                "over-sample without duplicating shards on disk."
            )

        # shard_quality_scores
        if self.shard_quality_scores is not None:
            if len(self.shard_quality_scores) != len(self.shards):
                raise ValueError(
                    f"DatasetSpec '{self.name}': 'shard_quality_scores' length "
                    f"({len(self.shard_quality_scores)}) must match 'shards' "
                    f"length ({len(self.shards)}).  "
                    "Provide one score per shard, or set shard_quality_scores=None."
                )
            bad = [
                (i, s) for i, s in enumerate(self.shard_quality_scores)
                if not isinstance(s, (int, float)) or s < 0.0
            ]
            if bad:
                examples = ", ".join(f"index {i}: {v}" for i, v in bad[:3])
                raise ValueError(
                    f"DatasetSpec '{self.name}': all 'shard_quality_scores' must "
                    f"be numeric and ≥ 0.  Offending entries: {examples}"
                    f"{'...' if len(bad) > 3 else ''}."
                )

        # min_sample_quality
        if self.min_sample_quality is not None:
            if not (0.0 <= self.min_sample_quality <= 1.0):
                raise ValueError(
                    f"DatasetSpec '{self.name}': 'min_sample_quality' must be "
                    f"in [0, 1], got {self.min_sample_quality}.  "
                    "Values outside [0, 1] cannot match any quality_score field."
                )

        # mean
        if self.mean is not None:
            self._validate_channel_stats("mean", self.mean, allow_zero=True)

        # std
        if self.std is not None:
            self._validate_channel_stats("std", self.std, allow_zero=False)

        # schema_version
        if not isinstance(self.schema_version, int) or self.schema_version < 0:
            raise ValueError(
                f"DatasetSpec '{self.name}': 'schema_version' must be a "
                f"non-negative integer, got {self.schema_version!r}."
            )

    @staticmethod
    def _validate_channel_stats(
        field_name: str,
        value: Tuple,
        allow_zero: bool,
    ) -> None:
        """Validate a (R, G, B) channel stats tuple."""
        if not (isinstance(value, (tuple, list)) and len(value) == 3):
            raise ValueError(
                f"DatasetSpec: '{field_name}' must be a 3-tuple of floats "
                f"(R, G, B), got {value!r}."
            )
        for i, v in enumerate(value):
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"DatasetSpec: '{field_name}[{i}]' must be a float, "
                    f"got {type(v).__name__} = {v!r}."
                )
            lo = 0.0 if allow_zero else 1e-9
            if not (lo <= float(v) <= 1.0):
                desc = "in [0, 1]" if allow_zero else "in (0, 1]"
                raise ValueError(
                    f"DatasetSpec: '{field_name}[{i}]' must be {desc}, "
                    f"got {v}.  "
                    f"{'(std=0 would cause division by zero in normalisation)' if not allow_zero else ''}"
                )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise to a JSON-compatible dict, always stamped with
        ``schema_version = CURRENT_SCHEMA_VERSION``.

        Used by :class:`~dino_loader.checkpoint.DataLoaderCheckpointer`
        to write per-epoch state.  The output is suitable for ``json.dumps``.

        Example::

            import json
            spec = DatasetSpec(name="laion", shards=["shard-000000.tar"])
            json.dumps(spec.to_dict())
        """
        return {
            "schema_version":       CURRENT_SCHEMA_VERSION,
            "name":                 self.name,
            "shards":               list(self.shards),
            "weight":               self.weight,
            "shard_sampling":       self.shard_sampling,
            "shard_quality_scores": (
                list(self.shard_quality_scores)
                if self.shard_quality_scores is not None else None
            ),
            "min_sample_quality":   self.min_sample_quality,
            "metadata_key":         self.metadata_key,
            "mean":                 list(self.mean)  if self.mean is not None else None,
            "std":                  list(self.std)   if self.std  is not None else None,
            "confidentialities":    list(self.confidentialities),
            "modalities":           list(self.modalities),
            "splits":               list(self.splits),
            "strategies":           list(self.strategies),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSpec":
        """
        Deserialise from a dict, applying any registered schema migrations.

        Old dicts without ``schema_version`` are treated as version 0 and
        upgraded through all registered migrations before construction.

        Parameters
        ----------
        data
            Raw dict (e.g. from ``json.loads``).  Not mutated in place.

        Raises
        ------
        ValueError
            If the dict is missing required fields after migration, or if
            any field fails validation.

        Example::

            import json
            spec = DatasetSpec.from_dict(json.loads(raw))
        """
        d = dict(data)  # shallow copy — migrations must not mutate the input

        # Apply migrations in order.
        version = int(d.pop("schema_version", 0))
        while version < CURRENT_SCHEMA_VERSION:
            fn = _MIGRATIONS.get(version)
            if fn is None:
                # No migration registered: assume forward-compatible addition.
                break
            d = fn(d)
            version += 1

        # Restore schema_version for the dataclass field.
        d["schema_version"] = CURRENT_SCHEMA_VERSION

        # Coerce list → tuple for mean/std (JSON stores them as lists).
        if d.get("mean") is not None:
            d["mean"] = tuple(d["mean"])
        if d.get("std") is not None:
            d["std"] = tuple(d["std"])

        # Remove unknown keys so that old checkpoints with extra fields do not
        # cause a TypeError on construction.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            warnings.warn(
                f"DatasetSpec.from_dict: ignoring unknown field(s): "
                f"{sorted(unknown)}.  These may be from a newer schema version.",
                UserWarning,
                stacklevel=2,
            )
            for k in unknown:
                d.pop(k)

        return cls(**d)

    # ── Human-readable summary ────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Return a one-line human-readable summary for log output.

        Example::

            DatasetSpec(name='laion2b', shards=50000, weight=0.500,
                        sampling=epoch, quality_filter=0.30)
        """
        quality = (
            f"quality_filter={self.min_sample_quality:.2f}"
            if self.min_sample_quality is not None
            else "quality_filter=off"
        )
        return (
            f"DatasetSpec(name={self.name!r}, shards={len(self.shards)}, "
            f"weight={self.weight:.3f}, sampling={self.shard_sampling}, "
            f"{quality})"
        )
