"""dino_loader.sources._wds_mix
================================
Mixing de pipelines WebDataset avec exposition de l'indice de dataset source.

``wds.RandomMix`` ne remonte pas l'indice du pipeline qui a produit chaque
sample.  Ce module fournit une alternative minimaliste qui expose cette
information sans sacrifier la reproductibilité ni la compatibilité WDS.

API publique
------------
``indexed_random_mix(sources, probs, seed)``
    Générateur pur : yield ``(sample, ds_index)``.  Testable sans WDS.

``IndexedRandomMixDataset``
    ``IterableDataset`` WDS wrappant ``indexed_random_mix``.  Drop-in
    replacement de ``wds.RandomMix`` quand les indices sont nécessaires.
"""

from __future__ import annotations

import random
from collections.abc import Generator, Iterable, Iterator, Sequence
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Générateur pur (testable sans WDS)
# ---------------------------------------------------------------------------


def indexed_random_mix(
    sources: Sequence[Iterable[T]],
    probs:   Sequence[float],
    seed:    int | None = None,
) -> Generator[tuple[T, int], None, None]:
    """Yield ``(sample, dataset_index)`` depuis plusieurs sources pondérées.

    S'arrête dès qu'une source est épuisée (comportement ``longest=False``
    de ``wds.RandomMix``).  Chaque source est consommée via un ``Iterator``
    indépendant ; les sources elles-mêmes ne sont pas modifiées.

    Args:
        sources: Séquence d'iterables (pipelines WDS ou quelconques).
        probs:   Poids bruts correspondants ; n'ont pas besoin d'être normalisés.
        seed:    Graine RNG pour la reproductibilité.  ``None`` → non déterministe.

    Yields:
        Paire ``(sample, ds_index)`` où ``ds_index`` est l'index dans *sources*.

    Raises:
        ValueError: Si *sources* et *probs* n'ont pas la même longueur, ou si
            la somme des poids est nulle ou négative.

    Examples:
        >>> list(indexed_random_mix([[1, 2], [10, 20]], [1.0, 1.0], seed=0))
        [(10, 1), (1, 0), (20, 1), (2, 0)]

    """
    if len(sources) != len(probs):
        msg = (
            f"indexed_random_mix: sources and probs must have the same length, "
            f"got {len(sources)} sources and {len(probs)} probs."
        )
        raise ValueError(msg)

    total = sum(probs)
    if total <= 0:
        msg = f"indexed_random_mix: sum of probs must be positive, got {list(probs)}."
        raise ValueError(msg)

    rng = random.Random(seed)  # noqa: S311 — reproductibilité, pas cryptographie
    norm_probs = [p / total for p in probs]

    iterators: list[Iterator[T]] = [iter(s) for s in sources]
    cum_probs  = list(np.cumsum(norm_probs))

    while True:
        r = rng.random()
        i = int(np.searchsorted(cum_probs, r, side="right"))
        # Clamp pour éviter un dépassement dû aux erreurs d'arrondi float.
        i = min(i, len(iterators) - 1)

        try:
            sample = next(iterators[i])
        except StopIteration:
            return

        yield sample, i


# ---------------------------------------------------------------------------
# IterableDataset WDS (wrappant le générateur pur)
# ---------------------------------------------------------------------------

try:
    from torch.utils.data import IterableDataset as _IterableDataset

    class IndexedRandomMixDataset(_IterableDataset):
        """``IterableDataset`` mixant plusieurs pipelines WDS avec indices exposés.

        Drop-in replacement de ``wds.RandomMix`` quand les indices de dataset
        source sont nécessaires.  Yield ``(sample, ds_index)``.

        Args:
            datasets: Séquence d'``IterableDataset`` (pipelines WDS).
            probs:    Poids bruts de mixage.
            seed:     Graine RNG.

        Examples:
            >>> ds = IndexedRandomMixDataset([pipe_a, pipe_b], probs=[0.7, 0.3], seed=42)
            >>> for sample, idx in ds:
            ...     process(sample, dataset_index=idx)

        """

        def __init__(
            self,
            datasets: Sequence[Any],
            probs:    Sequence[float],
            seed:     int | None = None,
        ) -> None:
            """Initialise le dataset mixé."""
            self._datasets = list(datasets)
            self._probs    = list(probs)
            self._seed     = seed

        def __iter__(self) -> Generator[tuple[Any, int], None, None]:
            """Yield ``(sample, ds_index)`` depuis les datasets pondérés."""
            yield from indexed_random_mix(self._datasets, self._probs, self._seed)

except ImportError:
    # Environnement sans torch : la classe n'est pas disponible mais le
    # générateur pur ``indexed_random_mix`` reste utilisable.
    pass
