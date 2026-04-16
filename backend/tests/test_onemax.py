"""Comprehensive tests for OneMax problem and (1+1) EA algorithm."""
import pytest
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.algorithms.one_plus_one_EA import OnePlusOneEA, OnePlusOneEATSP
from optimization_framework.problems.tsp import tour_cost


class TestOneMaxProblem:
    """Test OneMax objective function."""

    def test_all_ones(self):
        """Fitness should equal bitstring length for all ones."""
        assert fitnessOnemax("11111") == 5
        assert fitnessOnemax("1" * 10) == 10
        assert fitnessOnemax("1" * 100) == 100

    def test_all_zeros(self):
        """Fitness should be 0 for all zeros."""
        assert fitnessOnemax("00000") == 0
        assert fitnessOnemax("0" * 10) == 0

    def test_mixed_bits(self):
        """Fitness should count number of ones."""
        assert fitnessOnemax("10101") == 3
        assert fitnessOnemax("11001") == 3
        assert fitnessOnemax("10001") == 2

    def test_empty_string(self):
        """Empty string should have fitness 0."""
        assert fitnessOnemax("") == 0

    def test_single_bit(self):
        """Single bit tests."""
        assert fitnessOnemax("0") == 0
        assert fitnessOnemax("1") == 1


class TestOnePlusOneEABitstring:
    """Test (1+1) EA for bitstring optimization."""

    def test_convergence_onemax(self):
        """(1+1) EA should converge to optimum on OneMax."""
        bit_length = 10
        best, iterations, temp, population, evals, coords, fitness_history = OnePlusOneEA(
            fitnessOnemax, bit_length=bit_length, prob=None
        )

        # Should find optimum (all ones)
        assert fitnessOnemax(best) == bit_length
        assert best.count("1") == bit_length

        # Check return types and structure
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0
        assert isinstance(coords, list) and len(coords) > 0
        assert isinstance(fitness_history, list)
        assert len(fitness_history) == iterations + 1

    def test_fitness_history_monotonic(self):
        """Fitness history should be monotonically non-decreasing."""
        _, _, _, _, _, _, fitness_history = OnePlusOneEA(
            fitnessOnemax, bit_length=8
        )
        for i in range(len(fitness_history) - 1):
            assert fitness_history[i + 1] >= fitness_history[i]

    def test_small_bitstring(self):
        """Test with very small bitstring."""
        best, _, _, _, _, _, _ = OnePlusOneEA(fitnessOnemax, bit_length=5)
        assert len(best) == 5
        assert fitnessOnemax(best) == 5

    def test_mutation_parameter_effect(self):
        """Different mutation probabilities should be handled."""
        # Test with low mutation probability
        best_low, iters_low, _, _, _, _, _ = OnePlusOneEA(
            fitnessOnemax, bit_length=10, prob=0.01
        )
        # Test with high mutation probability
        best_high, iters_high, _, _, _, _, _ = OnePlusOneEA(
            fitnessOnemax, bit_length=10, prob=0.3
        )

        # Both should find optimum
        assert fitnessOnemax(best_low) == 10
        assert fitnessOnemax(best_high) == 10

    def test_population_dict_structure(self):
        """Population dict should be valid."""
        best, _, _, population, _, _, _ = OnePlusOneEA(fitnessOnemax, bit_length=5)
        assert isinstance(population, dict)

    def test_coords_are_bitstring_maps(self):
        """Coords should be mapped representations (tuples of x, y coordinates)."""
        _, _, _, _, _, coords, _ = OnePlusOneEA(fitnessOnemax, bit_length=8)
        assert isinstance(coords, list)
        assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in coords)

    def test_large_bitstring(self):
        """Test with larger bitstring."""
        best, _, _, _, _, _, _ = OnePlusOneEA(fitnessOnemax, bit_length=50)
        assert len(best) == 50
        assert best.count("1") == 50


class TestOnePlusOneEATSP:
    """Test (1+1) EA for TSP."""

    def test_simple_tsp(self):
        """Test (1+1) EA on a small TSP instance."""
        distance_matrix = [
            [0, 1, 4, 5],
            [1, 0, 2, 3],
            [4, 2, 0, 1],
            [5, 3, 1, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

        best_tour, iterations, temp, population, evals, tour_coords, cost_over_time = OnePlusOneEATSP(
            distance_matrix, city_coords, max_iterations=1000
        )

        # Verify return types
        assert isinstance(best_tour, list)
        assert len(best_tour) == len(distance_matrix)
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0
        assert isinstance(cost_over_time, list)

        # Tour should be valid (contains all cities once)
        assert len(set(best_tour)) == len(distance_matrix)

    def test_tsp_cost_calculation(self):
        """Cost should decrease over time."""
        distance_matrix = [
            [0, 1, 4],
            [1, 0, 2],
            [4, 2, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        _, _, _, _, _, _, cost_over_time = OnePlusOneEATSP(
            distance_matrix, city_coords, max_iterations=100
        )

        # First cost should be reasonable
        assert cost_over_time[0] > 0
        # Cost should not increase over time
        for i in range(len(cost_over_time) - 1):
            assert cost_over_time[i + 1] <= cost_over_time[i]

    def test_tour_validity(self):
        """Tour should visit each city exactly once."""
        distance_matrix = [
            [0, 2, 9, 10],
            [2, 0, 10, 5],
            [9, 10, 0, 3],
            [10, 5, 3, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)}

        best_tour, _, _, _, _, _, _ = OnePlusOneEATSP(
            distance_matrix, city_coords, max_iterations=500
        )

        # Check all cities visited
        assert set(best_tour) == set(range(len(distance_matrix)))
        assert len(best_tour) == len(distance_matrix)
