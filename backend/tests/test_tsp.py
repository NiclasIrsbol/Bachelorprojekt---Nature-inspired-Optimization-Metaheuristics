"""Unit tests for the TSP problem helpers (tour cost, optima, coord mapping)."""

import pytest

from optimization_framework.problems import tsp


class TestTourCost:
    def test_cyclic_cost_includes_return_edge(self):
        # Distances = |i - j| on a line of 4 cities.
        dm = [[abs(i - j) for j in range(4)] for i in range(4)]
        # Tour 0-1-2-3-(back to 0): 1 + 1 + 1 + 3 = 6.
        assert tsp.tour_cost([0, 1, 2, 3], dm) == 6

    def test_cost_is_rotation_invariant(self):
        dm = [[abs(i - j) for j in range(4)] for i in range(4)]
        assert tsp.tour_cost([0, 1, 2, 3], dm) == tsp.tour_cost([2, 3, 0, 1], dm)


class TestOptimaAndGap:
    def test_known_optimum(self):
        assert tsp.get_optimum("berlin52") == 7542

    def test_unknown_optimum_is_none(self):
        assert tsp.get_optimum("does_not_exist") is None

    def test_gap_zero_at_optimum(self):
        assert tsp.gap_percent(7542, "berlin52") == pytest.approx(0.0)

    def test_gap_positive_above_optimum(self):
        assert tsp.gap_percent(7542 * 1.1, "berlin52") == pytest.approx(10.0, abs=1e-6)

    def test_gap_none_when_optimum_unknown(self):
        assert tsp.gap_percent(1000, "does_not_exist") is None


class TestTourToCoords:
    def test_orders_points_by_tour(self):
        coords = {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 2.0)}
        assert tsp.tour_to_coords([2, 0, 1], coords) == [(2.0, 2.0), (0.0, 0.0), (1.0, 1.0)]

    def test_empty_coords_returns_empty(self):
        assert tsp.tour_to_coords([0, 1], {}) == []

    def test_out_of_range_index_returns_empty(self):
        coords = {0: (0.0, 0.0), 1: (1.0, 1.0)}
        assert tsp.tour_to_coords([0, 1, 2], coords) == []
