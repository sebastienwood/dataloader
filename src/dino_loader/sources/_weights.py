"""dino_loader.sources._weights
==============================
Vecteur de poids normalisé thread-safe pour le mixage de datasets.

Ce module est interne au package ``sources`` et partagé par ``hpc_source``
et ``wds_source``.  Il n'expose qu'une seule classe publique : ``MixingWeights``.
"""

import threading
from collections.abc import Sequence

from dino_datasets import DatasetSpec


class MixingWeights:
    """Vecteur de poids normalisé, thread-safe, pour le mixage de datasets.

    Le constructeur accepte des primitives ``(names, weights)``.  Utiliser la
    classmethod :meth:`from_specs` pour construire depuis des ``DatasetSpec``.

    Args:
        names: Liste ordonnée des noms de datasets.
        weights: Poids bruts (non normalisés) correspondants.  Doivent être
            positifs et de même longueur que *names*.

    Raises:
        ValueError: Si les longueurs diffèrent ou si la somme des poids est nulle.

    """

    def __init__(self, names: list[str], weights: Sequence[float]) -> None:
        """Initialise avec des noms et poids explicites."""
        if len(names) != len(weights):
            msg = (
                f"MixingWeights: names and weights must have the same length, "
                f"got {len(names)} names and {len(weights)} weights."
            )
            raise ValueError(msg)
        self.names    = list(names)
        self._lock    = threading.Lock()
        self._weights = self._normalise(list(weights))

    @classmethod
    def from_specs(cls, specs: list[DatasetSpec]) -> "MixingWeights":
        """Construit depuis une liste de ``DatasetSpec``.

        Args:
            specs: Spécifications de datasets (champs ``name`` et ``weight``).

        Returns:
            Instance de ``MixingWeights`` avec les poids normalisés.

        """
        names   = [s.name   for s in specs]
        weights = [s.weight for s in specs]
        return cls(names, weights)

    def get(self) -> list[float]:
        """Retourne une copie du vecteur de poids normalisés.

        Returns:
            Liste de flottants sommant à 1.0.

        """
        with self._lock:
            return list(self._weights)

    def set(self, weights: Sequence[float]) -> None:
        """Remplace le vecteur (re-normalisé automatiquement).

        Args:
            weights: Nouveaux poids bruts.  Doit avoir la même longueur que
                le vecteur courant.

        Raises:
            ValueError: Si la longueur ne correspond pas ou si la somme est nulle.

        """
        if len(weights) != len(self.names):
            msg = (
                f"MixingWeights.set: expected {len(self.names)} weights, "
                f"got {len(weights)}."
            )
            raise ValueError(msg)
        with self._lock:
            self._weights = self._normalise(list(weights))

    def set_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids brut d'un dataset par son nom.

        Les autres poids bruts restent inchangés ; seule la normalisation
        est recalculée.

        Args:
            name: Nom du dataset à modifier.
            weight: Nouveau poids brut.

        Raises:
            KeyError: Si *name* n'est pas dans le vecteur.

        """
        try:
            idx = self.names.index(name)
        except ValueError:
            msg = f"Dataset '{name}' not found. Available: {self.names}"
            raise KeyError(msg) from None
        with self._lock:
            raw      = list(self._weights)
            raw[idx] = weight
            self._weights = self._normalise(raw)

    @staticmethod
    def _normalise(weights: list[float]) -> list[float]:
        """Normalise un vecteur de poids pour que leur somme soit 1.0.

        Args:
            weights: Poids bruts.

        Returns:
            Poids normalisés.

        Raises:
            ValueError: Si la somme est nulle ou négative.

        """
        total = sum(weights)
        if total <= 0:
            msg = f"Weights must sum to a positive number, got {weights}."
            raise ValueError(msg)
        return [w / total for w in weights]
