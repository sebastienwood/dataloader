"""dino_loader.sources.resolution
=================================
Holder thread-safe de la résolution courante de crop.

``ResolutionSource`` joue le rôle de callback DALI ``ExternalSource``
(``batch=False``) : chaque appel retourne les dimensions courantes sous forme
de scalaires numpy, consommés par le pipeline DALI pour redimensionner les
crops à chaque batch.

L'appel à :meth:`set` est immédiatement visible par le prochain prefetch DALI,
ce qui permet de changer la résolution en cours d'entraînement (curriculum de
résolution progressive) sans reconstruire le pipeline.

Cette classe est intentionnellement simple et sans dépendance : elle peut être
importée dans n'importe quel environnement (tests, benchmarks, CPU-only).
"""

import threading

import numpy as np


class ResolutionSource:
    """Holder thread-safe de la résolution courante de crop.

    Utilisée comme callback DALI ``ExternalSource`` et comme source de vérité
    pour les backends CPU qui lisent directement la valeur à chaque batch.

    Args:
        global_size: Taille initiale du crop global en pixels.
        local_size: Taille initiale du crop local en pixels.

    """

    def __init__(self, global_size: int, local_size: int) -> None:
        """Initialise avec les tailles de crop initiales."""
        self._global = global_size
        self._local  = local_size
        self._lock   = threading.Lock()

    def set(self, global_size: int, local_size: int) -> None:
        """Met à jour les deux dimensions de manière atomique.

        Appelé par :meth:`DINODataLoader.set_resolution` depuis le thread
        principal.  La valeur est visible par le prochain appel à
        :meth:`__call__` quel que soit le thread appelant.

        Args:
            global_size: Nouvelle taille du crop global en pixels.
            local_size: Nouvelle taille du crop local en pixels.

        """
        with self._lock:
            self._global = global_size
            self._local  = local_size

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """Retourne ``(global_size, local_size)`` comme scalaires numpy int32.

        Signature compatible avec DALI ``ExternalSource`` (``batch=False,
        num_outputs=2``).

        Returns:
            Paire de scalaires numpy int32.

        """
        with self._lock:
            return (
                np.array(self._global, dtype=np.int32),
                np.array(self._local,  dtype=np.int32),
            )
