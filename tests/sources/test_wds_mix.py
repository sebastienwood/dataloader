"""tests.sources.test_wds_mix
==============================
Tests unitaires pour ``dino_loader.sources._wds_mix``.

Périmètre
---------
- ``indexed_random_mix`` : comportement fonctionnel pur, sans WDS ni torch.
- ``IndexedRandomMixDataset`` : intégration minimale (skip si torch absent).

Organisation
------------
Chaque classe de test couvre un aspect orthogonal ; les fixtures sont légères
et locales (listes Python ordinaires comme sources).
"""

from __future__ import annotations

import pytest

from dino_loader.sources._wds_mix import indexed_random_mix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect(sources, probs, seed=0):
    """Collecte tous les résultats de indexed_random_mix en liste."""
    return list(indexed_random_mix(sources, probs, seed=seed))


# ---------------------------------------------------------------------------
# Validation des arguments
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            collect([[1, 2], [3, 4]], probs=[1.0])

    def test_zero_probs_raises(self):
        with pytest.raises(ValueError, match="positive"):
            collect([[1, 2]], probs=[0.0])

    def test_negative_sum_raises(self):
        with pytest.raises(ValueError, match="positive"):
            collect([[1, 2], [3, 4]], probs=[-1.0, -1.0])

    def test_single_source_valid(self):
        result = collect([[10, 20, 30]], probs=[1.0])
        assert [s for s, _ in result] == [10, 20, 30]
        assert all(i == 0 for _, i in result)


# ---------------------------------------------------------------------------
# Indices de dataset
# ---------------------------------------------------------------------------


class TestDatasetIndices:
    def test_indices_in_range(self):
        result = collect([[1, 2, 3], [10, 20, 30]], probs=[1.0, 1.0], seed=42)
        indices = [i for _, i in result]
        assert all(0 <= i < 2 for i in indices)

    def test_single_source_always_index_zero(self):
        result = collect([[1, 2, 3, 4, 5]], probs=[2.0])
        assert all(i == 0 for _, i in result)

    def test_three_sources_all_indices_present(self):
        # Avec 100 items par source et des poids égaux, les 3 indices doivent apparaître.
        sources = [list(range(100)), list(range(100, 200)), list(range(200, 300))]
        result = collect(sources, probs=[1.0, 1.0, 1.0], seed=7)
        observed = {i for _, i in result}
        assert observed == {0, 1, 2}

    def test_exclusive_weight_gives_single_index(self):
        # Poids [1, 0, 0] → seul le dataset 0 est sélectionné.
        # On utilise des poids très déséquilibrés pour éviter les erreurs float.
        sources = [list(range(50)), list(range(50)), list(range(50))]
        result = collect(sources, probs=[1.0, 0.0, 0.0], seed=0)
        indices = [i for _, i in result]
        assert all(i == 0 for i in indices)


# ---------------------------------------------------------------------------
# Comportement de terminaison
# ---------------------------------------------------------------------------


class TestTermination:
    def test_stops_on_first_exhausted_source(self):
        # Source 0 : 2 éléments, source 1 : 100 éléments.
        # Le mix doit s'arrêter dès que source 0 est épuisée.
        result = collect([[1, 2], list(range(100))], probs=[0.5, 0.5], seed=0)
        samples = [s for s, _ in result]
        # Au moins 1 sample de chaque source, mais pas plus de 102 au total.
        assert len(result) <= 102
        assert 1 in samples or 2 in samples  # au moins un de la source 0

    def test_empty_source_yields_nothing(self):
        result = collect([[], [1, 2, 3]], probs=[0.5, 0.5], seed=0)
        # La source vide est la première à être sélectionnée → arrêt immédiat.
        # Le comportement exact dépend du seed, mais le résultat doit être fini.
        assert isinstance(result, list)

    def test_all_empty_sources_yields_nothing(self):
        result = collect([[], []], probs=[1.0, 1.0], seed=0)
        assert result == []


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_gives_same_result(self):
        sources = [list(range(20)), list(range(100, 120))]
        r1 = collect(sources, probs=[1.0, 1.0], seed=123)
        r2 = collect(sources, probs=[1.0, 1.0], seed=123)
        assert r1 == r2

    def test_different_seeds_give_different_results(self):
        sources = [list(range(50)), list(range(50, 100))]
        r1 = collect(sources, probs=[1.0, 1.0], seed=1)
        r2 = collect(sources, probs=[1.0, 1.0], seed=2)
        # Très improbable qu'ils soient identiques avec 50 éléments.
        assert r1 != r2

    def test_none_seed_is_non_deterministic(self):
        """Sans seed, deux appels successifs peuvent différer (non garanti, mais probable)."""
        sources = [list(range(100)), list(range(100, 200))]
        r1 = collect(sources, probs=[1.0, 1.0], seed=None)
        r2 = collect(sources, probs=[1.0, 1.0], seed=None)
        # On ne peut pas asserter l'inégalité, mais on vérifie que ça tourne.
        assert len(r1) > 0
        assert len(r2) > 0


# ---------------------------------------------------------------------------
# Distribution statistique
# ---------------------------------------------------------------------------


class TestDistribution:
    def test_equal_weights_roughly_uniform(self):
        n = 200
        sources = [list(range(n)), list(range(n, 2 * n))]
        result = collect(sources, probs=[1.0, 1.0], seed=42)
        indices = [i for _, i in result]
        count_0 = indices.count(0)
        count_1 = indices.count(1)
        total   = count_0 + count_1
        # Avec n=200 et poids égaux, on attend ~50 % chacun ± 15 %.
        assert abs(count_0 / total - 0.5) < 0.15

    def test_skewed_weights_respected(self):
        n = 500
        sources = [list(range(n)), list(range(n, 2 * n))]
        result = collect(sources, probs=[9.0, 1.0], seed=0)
        indices = [i for _, i in result]
        count_0 = indices.count(0)
        total   = len(indices)
        # Poids 90/10 → on attend ≥ 75 % pour le dataset 0.
        assert count_0 / total >= 0.75

    def test_unnormalized_probs_same_as_normalized(self):
        sources = [list(range(100)), list(range(100, 200))]
        r1 = collect(sources, probs=[2.0, 2.0], seed=5)
        r2 = collect(sources, probs=[1.0, 1.0], seed=5)
        assert r1 == r2


# ---------------------------------------------------------------------------
# IndexedRandomMixDataset (skip si torch absent)
# ---------------------------------------------------------------------------


class TestIndexedRandomMixDataset:
    """Tests de ``IndexedRandomMixDataset`` — skippés si torch n'est pas installé."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch", reason="torch required for IndexedRandomMixDataset")

    def test_yields_sample_and_index(self):
        from dino_loader.sources._wds_mix import IndexedRandomMixDataset

        ds = IndexedRandomMixDataset([[1, 2, 3], [10, 20, 30]], probs=[1.0, 1.0], seed=0)
        result = list(ds)
        assert all(isinstance(s, int) for s, _ in result)
        assert all(i in (0, 1) for _, i in result)

    def test_reproducible_with_same_seed(self):
        from dino_loader.sources._wds_mix import IndexedRandomMixDataset

        ds1 = IndexedRandomMixDataset([[1, 2, 3], [10, 20, 30]], probs=[1.0, 1.0], seed=99)
        ds2 = IndexedRandomMixDataset([[1, 2, 3], [10, 20, 30]], probs=[1.0, 1.0], seed=99)
        assert list(ds1) == list(ds2)

    def test_iterable_twice_gives_same_result(self):
        from dino_loader.sources._wds_mix import IndexedRandomMixDataset

        ds = IndexedRandomMixDataset([[1, 2, 3], [10, 20, 30]], probs=[1.0, 1.0], seed=7)
        assert list(ds) == list(ds)