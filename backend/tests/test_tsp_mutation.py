"""Tests for the TSP mutation operators, the 2-opt/3-opt dispatcher, and the
bit-string projection helper."""

import random

import pytest

from optimization_framework.operators import gaoperators
from optimization_framework.algorithms.one_plus_one_EA import OnePlusOneEATSP
from optimization_framework.problems.tsp import tour_cost


def _ring_distance_matrix(n):
    """Symmetric |i-j| distance matrix on n cities."""
    return [[abs(i - j) for j in range(n)] for i in range(n)]


def _coords(n):
    return {i: (float(i), 0.0) for i in range(n)}


class TestThreeOptMutation:
    def test_returns_a_permutation(self):
        random.seed(0)
        dm = _ring_distance_matrix(6)
        tour = [0, 1, 2, 3, 4, 5]
        out = gaoperators.three_opt_mutation(tour, dm)
        assert sorted(out) == list(range(6))
        assert len(out) == 6

    def test_never_increases_cyclic_cost(self):
        # 3-opt picks the best reconnection, which always includes a rotation of
        # the original tour (same cyclic cost), so it cannot make the tour worse.
        random.seed(1)
        dm = _ring_distance_matrix(7)
        tour = [3, 0, 5, 1, 6, 2, 4]
        out = gaoperators.three_opt_mutation(tour, dm)
        assert tour_cost(out, dm) <= tour_cost(tour, dm)


class TestTspMutationDispatch:
    def test_two_opt_preserves_cities(self):
        random.seed(2)
        dm = _ring_distance_matrix(6)
        out = gaoperators.tsp_mutation([0, 1, 2, 3, 4, 5], dm, "2opt")
        assert sorted(out) == list(range(6))

    def test_three_opt_selected(self):
        random.seed(3)
        dm = _ring_distance_matrix(6)
        out = gaoperators.tsp_mutation([0, 1, 2, 3, 4, 5], dm, "3opt")
        assert sorted(out) == list(range(6))

    def test_unknown_kind_raises(self):
        dm = _ring_distance_matrix(4)
        with pytest.raises(ValueError):
            gaoperators.tsp_mutation([0, 1, 2, 3], dm, "swap")


class TestSolverMutationParameter:
    def test_one_plus_one_tsp_accepts_3opt(self):
        random.seed(4)
        dm = _ring_distance_matrix(6)
        best, *_ = OnePlusOneEATSP(dm, _coords(6), max_iterations=30, mutation="3opt")
        assert sorted(best) == list(range(6))

    def test_one_plus_one_tsp_rejects_unknown_mutation(self):
        dm = _ring_distance_matrix(6)
        with pytest.raises(ValueError):
            OnePlusOneEATSP(dm, _coords(6), max_iterations=5, mutation="nonsense")


class TestMapBitstring:
    def test_all_zeros_maps_to_bottom(self):
        assert gaoperators.map_bitstring("0000") == (0.0, -1.0)

    def test_all_ones_maps_to_top(self):
        assert gaoperators.map_bitstring("1111") == (0.0, 1.0)

    def test_returns_two_floats(self):
        x, y = gaoperators.map_bitstring("1010")
        assert isinstance(x, float) and isinstance(y, float)
        assert -1.0 <= y <= 1.0
