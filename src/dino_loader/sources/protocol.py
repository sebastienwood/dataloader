"""dino_loader.sources.protocol
================================
Interface commune (Protocol) pour toutes les sources de données dino_loader.

Motivation
----------
Le projet dispose de deux implémentations de source :

- ``MixingSource`` (``hpc_source``) : optimisée HPC, cache /dev/shm,
  double-buffering strict I/O + extraction. Source de production sur
  B200 / GB200 NVL72 avec Lustre.
- ``WDSSource`` (``wds_source``) : basée webdataset, plus simple, adaptée
  aux configurations NVMe local ou Lustre MDS rapide.

``SourceProtocol`` garantit que les deux sources sont interchangeables du
point de vue du reste du codebase (``ShardReaderNode``, ``_ReaderAdapter``,
``NormSource``). Tout consommateur de source doit typer ses arguments avec
ce protocol plutôt qu'avec une implémentation concrète.

Contrat
-------
Une source doit :

1. Être appelable (``__call__``) et retourner un batch de JPEGs numpy.
2. Exposer les métadonnées du dernier batch via ``pop_last_metadata()``.
3. Accepter un callback d'indices dataset via ``register_dataset_index_callback()``.
4. Supporter la mise à jour de l'époque (``set_epoch``), des poids
   (``set_weights``, ``set_by_name``), et la libération des ressources
   (``close``).
5. Exposer les propriétés de lecture ``current_weights`` et ``dataset_names``.

Note sur ``batch_size``
------------------------
``_batch_size`` est une convention (pas une méthode du protocol) utilisée par
les backends (``CPUBackend``, ``DALIBackend``) pour inférer la taille de batch
depuis la source via ``getattr(source, '_batch_size', 1)``. Les implémentations
concrètes doivent donc exposer cet attribut.
"""

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SourceProtocol(Protocol):
    """Interface commune pour les sources de données dino_loader.

    Toute source retourne des JPEGs bruts (numpy uint8) et expose les
    métadonnées et indices dataset associés à chaque batch.

    Ce protocol est ``runtime_checkable`` : ``isinstance(src, SourceProtocol)``
    fonctionne à l'exécution pour les vérifications de type dans les factories.
    """

    # Attribut de convention lu par les backends via getattr.
    _batch_size: int

    def __call__(self) -> list[np.ndarray]:
        """Retourne un batch de ``_batch_size`` tableaux numpy uint8 (JPEGs bruts).

        Returns:
            Liste de tableaux numpy de dtype ``uint8``, un par sample.

        """
        ...

    def pop_last_metadata(self) -> list[dict | None]:
        """Retourne les métadonnées du dernier ``__call__`` et vide le buffer.

        Returns:
            Liste de longueur ``_batch_size`` de dicts JSON ou ``None``.

        """
        ...

    def register_dataset_index_callback(
        self,
        cb: Callable[[list[int]], None],
    ) -> None:
        """Enregistre un callback recevant les indices dataset par sample.

        Le callback est appelé avec une liste d'entiers de longueur
        ``_batch_size`` après chaque ``__call__``.

        Args:
            cb: Callable ``(list[int]) -> None``.

        """
        ...

    def set_epoch(self, epoch: int) -> None:
        """Prépare la source pour une nouvelle époque (re-shuffle des shards).

        Args:
            epoch: Numéro de la nouvelle époque.

        """
        ...

    def set_weights(self, weights: Sequence[float]) -> None:
        """Met à jour les poids de mixage (re-normalisés automatiquement).

        Args:
            weights: Nouveaux poids bruts, même longueur que ``dataset_names``.

        """
        ...

    def set_by_name(self, name: str, weight: float) -> None:
        """Met à jour le poids d'un dataset par son nom.

        Args:
            name:   Nom du dataset.
            weight: Nouveau poids brut.

        """
        ...

    @property
    def current_weights(self) -> list[float]:
        """Poids de mixage normalisés courants (somme = 1.0)."""
        ...

    @property
    def dataset_names(self) -> list[str]:
        """Liste ordonnée des noms de datasets."""
        ...

    def close(self) -> None:
        """Libère toutes les ressources (threads, fichiers, connexions)."""
        ...
