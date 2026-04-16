"""Comprehensive tests for Simulated Annealing algorithm."""
import pytest
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.problems.leadingones import fitnessLeadingOnes
from optimization_framework.algorithms.simulated_annealing import (
    simulated_annealing,
    simulated_annealingTSP,
)
from optimization_framework.problems.tsp import tour_cost


class TestSimulatedAnnealingBitstring:
    """Test Simulated Annealing for bitstring optimization."""

    def test_convergence_onemax(self):
        """SA should solve OneMax."""
        best, iterations, temp_final, population, evals, coords, fitness_history = simulated_annealing(
            fitnessOnemax, bit_length=10, cooling=0.99, T0=100.0
        )

        # Should find optimum
        assert fitnessOnemax(best) == 10
        assert best.count("1") == 10

        # Check return types
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(temp_final, (int, float))
        assert isinstance(evals, int) and evals > 0
        assert isinstance(coords, list)
        assert isinstance(fitness_history, list)
        assert len(fitness_history) == iterations + 1

    def test_convergence_leadingones(self):
        """SA should solve LeadingOnes."""
        best, _, _, _, _, _, _ = simulated_annealing(
            fitnessLeadingOnes, bit_length=12, cooling=0.95, T0=50.0
        )

        assert fitnessLeadingOnes(best) == 12
        assert best == "1" * 12

    def test_temperature_decreases(self):
        """Temperature should decrease over iterations."""
        _, _, temp_final, _, _, _, _ = simulated_annealing(
            fitnessOnemax, bit_length=8, cooling=0.9, T0=100.0
        )

        # Temperature should decrease
        assert temp_final < 100.0

    def test_cooling_rate_effect(self):
        """Different cooling rates should be accepted."""
        for cooling in [0.90, 0.95, 0.99, 0.999]:
            best, _, _, _, _, _, _ = simulated_annealing(
                fitnessOnemax, bit_length=8, cooling=cooling, T0=50.0
            )
            assert len(best) == 8

    def test_initial_temperature_effect(self):
        """Different initial temperatures should work."""
        # Low initial temperature
        best1, iters1, _, _, _, _, _ = simulated_annealing(
            fitnessOnemax, bit_length=8, T0=10.0
        )
        # High initial temperature
        best2, iters2, _, _, _, _, _ = simulated_annealing(
            fitnessOnemax, bit_length=8, T0=1000.0
        )

        assert len(best1) == 8
        assert len(best2) == 8

    def test_mutation_probability(self):
        """Different mutation probabilities should work."""
        for prob in [0.01, 0.05, 0.1]:
            best, _, _, _, _, _, _ = simulated_annealing(
                fitnessOnemax, bit_length=8, prob=prob
            )
            assert len(best) == 8

    def test_fitness_history_structure(self):
        """Fitness history should match iteration count."""
        _, iterations, _, _, _, _, fitness_history = simulated_annealing(
            fitnessOnemax, bit_length=8
        )
        assert len(fitness_history) == iterations + 1

    def test_population_dict_structure(self):
        """Population should be valid dict."""
        best, _, _, population, _, _, _ = simulated_annealing(
            fitnessOnemax, bit_length=5
        )
        assert isinstance(population, dict)

    def test_small_bitstring(self):
        """Test with small bitstring."""
        best, _, _, _, _, _, _ = simulated_annealing(
            fitnessOnemax, bit_length=5, T0=50.0
        )
        assert len(best) == 5

    def test_large_bitstring(self):
        """Test with larger bitstring."""
        best, _, _, _, _, _, _ = simulated_annealing(
            fitnessOnemax, bit_length=30, T0=100.0, cooling=0.99
        )
        assert len(best) == 30


class TestSimulatedAnnealingTSP:
    """Test Simulated Annealing for TSP."""

    def test_tsp_convergence(self):
        """SA should find reasonable tours for TSP."""
        distance_matrix = [
            [0, 1, 4, 5],
            [1, 0, 2, 3],
            [4, 2, 0, 1],
            [5, 3, 1, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

        best_tour, iterations, temp_final, population, evals, tour_coords, cost_over_time = simulated_annealingTSP(
            distance_matrix,
            city_coords,
            cooling=0.9995,
            T0=1000.0,
            max_iterations=2000,
        )

        # Verify return types
        assert isinstance(best_tour, list)
        assert len(best_tour) == len(distance_matrix)
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0
        assert isinstance(cost_over_time, list)

        # Tour should be valid
        assert len(set(best_tour)) == len(distance_matrix)

    def test_cost_improves_or_stable(self):
        """Best cost should not increase significantly over time."""
        distance_matrix = [
            [0, 2, 9],
            [2, 0, 10],
            [9, 10, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        _, _, _, _, _, _, cost_over_time = simulated_annealingTSP(
            distance_matrix,
            city_coords,
            cooling=0.99,
            T0=100.0,
            max_iterations=500,
        )

        # Cost should not increase
        for i in range(len(cost_over_time) - 1):
            assert cost_over_time[i + 1] <= cost_over_time[i] + 0.01

    def test_temperature_decreases_tsp(self):
        """Temperature should decrease over time in TSP."""
        distance_matrix = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        _, _, temp_final, _, _, _, _ = simulated_annealingTSP(
            distance_matrix,
            city_coords,
            cooling=0.9,
            T0=500.0,
            max_iterations=100,
        )

        assert temp_final < 500.0

    def test_different_cooling_rates(self):
        """Different cooling rates should be accepted."""
        distance_matrix = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        for cooling in [0.9, 0.95, 0.999]:
            best_tour, _, _, _, _, _, _ = simulated_annealingTSP(
                distance_matrix, city_coords, cooling=cooling, T0=100.0, max_iterations=100
            )
            assert len(best_tour) == 3

    def test_different_initial_temperatures(self):
        """Different initial temperatures should work."""
        distance_matrix = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        for T0 in [10.0, 100.0, 1000.0]:
            best_tour, _, _, _, _, _, _ = simulated_annealingTSP(
                distance_matrix, city_coords, T0=T0, max_iterations=100
            )
            assert len(best_tour) == 3
            assert set(best_tour) == {0, 1, 2}

    def test_tour_validity(self):
        """Tour should contain all cities exactly once."""
        distance_matrix = [
            [0, 2, 9, 10],
            [2, 0, 10, 5],
            [9, 10, 0, 3],
            [10, 5, 3, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)}

        best_tour, _, _, _, _, _, _ = simulated_annealingTSP(
            distance_matrix, city_coords, max_iterations=500
        )

        # Check all cities are present
        assert set(best_tour) == set(range(len(distance_matrix)))
        assert len(best_tour) == len(distance_matrix)

    def test_cost_over_time_structure(self):
        """Cost over time should match expected iterations."""
        distance_matrix = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        _, iterations, _, _, _, _, cost_over_time = simulated_annealingTSP(
            distance_matrix, city_coords, max_iterations=100
        )

        assert len(cost_over_time) == iterations + 1
