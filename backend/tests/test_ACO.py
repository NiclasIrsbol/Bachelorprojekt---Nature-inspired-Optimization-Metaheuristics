"""Comprehensive tests for ACO algorithms (MMAS and PACO)."""
import pytest
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.problems.leadingones import fitnessLeadingOnes
from optimization_framework.algorithms.ant_optimization_problem import (
    ant_colony_optimization,
    ant_colony_optimizationTSP,
    population_based_aco,
    population_based_acoTSP,
)
from optimization_framework.problems.tsp import tour_cost


class TestMMASACOBitstring:
    """Test MMAS ACO for bitstring optimization."""

    def test_convergence_onemax(self):
        """MMAS ACO should solve OneMax."""
        best, iterations, _, population, evals, coords, fitness_history = ant_colony_optimization(
            fitnessOnemax, bit_length=20, rho=0.1, max_iterations=5000
        )

        # Should find optimum
        assert fitnessOnemax(best) == 20

        # Check return types
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0
        assert isinstance(coords, list)
        assert isinstance(fitness_history, list)
        assert len(fitness_history) == iterations + 1

    def test_convergence_leadingones(self):
        """MMAS ACO should solve LeadingOnes."""
        best, _, _, _, _, _, _ = ant_colony_optimization(
            fitnessLeadingOnes, bit_length=15, rho=0.1, max_iterations=5000
        )

        assert fitnessLeadingOnes(best) == 15
        assert best == "1" * 15

    def test_evaporation_parameter_effect(self):
        """Different evaporation rates should be accepted."""
        for rho in [0.05, 0.1, 0.5]:
            best, _, _, _, _, _, _ = ant_colony_optimization(
                fitnessOnemax, bit_length=10, rho=rho, max_iterations=1000
            )
            assert len(best) == 10
            assert all(b in "01" for b in best)

    def test_max_iterations_limit(self):
        """Algorithm should respect iteration limit."""
        max_iters = 100
        _, iterations, _, _, _, _, _ = ant_colony_optimization(
            fitnessOnemax, bit_length=100, max_iterations=max_iters
        )
        assert iterations <= max_iters

    def test_fitness_history_monotonic(self):
        """Fitness should not decrease."""
        _, _, _, _, _, _, fitness_history = ant_colony_optimization(
            fitnessOnemax, bit_length=10, max_iterations=500
        )
        for i in range(len(fitness_history) - 1):
            assert fitness_history[i + 1] >= fitness_history[i]

    def test_solution_validity(self):
        """Solution should be valid bitstring."""
        best, _, _, _, _, _, _ = ant_colony_optimization(
            fitnessOnemax, bit_length=8, max_iterations=1000
        )
        assert len(best) == 8
        assert all(b in "01" for b in best)

    def test_population_structure(self):
        """Population dict should be valid."""
        best, _, _, population, _, _, _ = ant_colony_optimization(
            fitnessOnemax, bit_length=5, max_iterations=500
        )
        assert isinstance(population, dict)

    def test_coords_validity(self):
        """Coords should have valid structure."""
        _, _, _, _, _, coords, _ = ant_colony_optimization(
            fitnessOnemax, bit_length=8, max_iterations=500
        )
        assert isinstance(coords, list)
        assert len(coords) > 0


class TestMMASACOTSP:
    """Test MMAS ACO for TSP."""

    def test_tsp_convergence(self):
        """MMAS ACO should find valid tours for TSP."""
        distance_matrix = [
            [0, 1, 4, 5],
            [1, 0, 2, 3],
            [4, 2, 0, 1],
            [5, 3, 1, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

        best_tour, iterations, _, population, evals, tour_coords, cost_over_time = ant_colony_optimizationTSP(
            distance_matrix,
            city_coords,
            rho=0.1,
            max_iterations=500,
            alpha=1,
            beta=2,
        )

        # Verify return types
        assert isinstance(best_tour, list)
        assert len(best_tour) == len(distance_matrix)
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0

        # Tour should be valid
        assert set(best_tour) == set(range(len(distance_matrix)))

    def test_tsp_cost_improvement(self):
        """Cost should improve or stay stable."""
        distance_matrix = [
            [0, 2, 9],
            [2, 0, 10],
            [9, 10, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        _, _, _, _, _, _, cost_over_time = ant_colony_optimizationTSP(
            distance_matrix, city_coords, max_iterations=500
        )

        # Cost should not increase
        for i in range(len(cost_over_time) - 1):
            assert cost_over_time[i + 1] <= cost_over_time[i] + 0.01

    def test_different_alpha_beta(self):
        """Different alpha/beta values should work."""
        distance_matrix = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        for alpha in [0.5, 1, 2]:
            for beta in [1, 2, 3]:
                best_tour, _, _, _, _, _, _ = ant_colony_optimizationTSP(
                    distance_matrix,
                    city_coords,
                    alpha=alpha,
                    beta=beta,
                    max_iterations=100,
                )
                assert len(best_tour) == 3
                assert set(best_tour) == {0, 1, 2}

    def test_tour_validity(self):
        """Tour should visit each city exactly once."""
        distance_matrix = [
            [0, 2, 9, 10],
            [2, 0, 10, 5],
            [9, 10, 0, 3],
            [10, 5, 3, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)}

        best_tour, _, _, _, _, _, _ = ant_colony_optimizationTSP(
            distance_matrix, city_coords, max_iterations=500
        )

        assert set(best_tour) == set(range(4))
        assert len(best_tour) == 4


class TestPopulationBasedACOBitstring:
    """Test Population-Based ACO (PACO) for bitstrings."""

    def test_convergence_onemax(self):
        """PACO should solve OneMax."""
        best, iterations, _, population, evals, coords, fitness_history = population_based_aco(
            fitnessOnemax,
            bit_length=20,
            archive_size=10,
            num_ants=30,
            max_iterations=2000,
        )

        # Should find optimum
        assert fitnessOnemax(best) == 20

        # Check return types
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0
        assert isinstance(fitness_history, list)
        assert len(fitness_history) == iterations + 1

    def test_convergence_leadingones(self):
        """PACO should solve LeadingOnes."""
        best, _, _, _, _, _, _ = population_based_aco(
            fitnessLeadingOnes,
            bit_length=15,
            archive_size=10,
            num_ants=30,
            max_iterations=2000,
        )

        assert fitnessLeadingOnes(best) == 15

    def test_archive_size_effect(self):
        """Different archive sizes should work."""
        for archive_size in [5, 10, 20]:
            best, _, _, _, _, _, _ = population_based_aco(
                fitnessOnemax,
                bit_length=10,
                archive_size=archive_size,
                max_iterations=1000,
            )
            assert len(best) == 10

    def test_num_ants_effect(self):
        """Different number of ants should work."""
        for num_ants in [10, 30, 50]:
            best, _, _, _, _, _, _ = population_based_aco(
                fitnessOnemax,
                bit_length=10,
                num_ants=num_ants,
                max_iterations=500,
            )
            assert len(best) == 10

    def test_greedy_probability(self):
        """Different q0 values should work."""
        for q0 in [0.0, 0.5, 0.9, 1.0]:
            best, _, _, _, _, _, _ = population_based_aco(
                fitnessOnemax,
                bit_length=8,
                q0=q0,
                max_iterations=500,
            )
            assert len(best) == 8

    def test_fitness_history_monotonic(self):
        """Fitness should not decrease."""
        _, _, _, _, _, _, fitness_history = population_based_aco(
            fitnessOnemax, bit_length=10, max_iterations=500
        )
        for i in range(len(fitness_history) - 1):
            assert fitness_history[i + 1] >= fitness_history[i]

    def test_solution_validity(self):
        """Solution should be valid bitstring."""
        best, _, _, _, _, _, _ = population_based_aco(
            fitnessOnemax, bit_length=10, max_iterations=1000
        )
        assert len(best) == 10
        assert all(b in "01" for b in best)


class TestPopulationBasedACOTSP:
    """Test Population-Based ACO (PACO) for TSP."""

    def test_tsp_convergence(self):
        """PACO should find valid tours for TSP."""
        distance_matrix = [
            [0, 1, 4, 5],
            [1, 0, 2, 3],
            [4, 2, 0, 1],
            [5, 3, 1, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

        best_tour, iterations, _, population, evals, tour_coords, cost_over_time = population_based_acoTSP(
            distance_matrix,
            city_coords,
            archive_size=10,
            num_ants=30,
            max_iterations=500,
        )

        # Verify return types
        assert isinstance(best_tour, list)
        assert len(best_tour) == len(distance_matrix)
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0

        # Tour should be valid
        assert set(best_tour) == set(range(len(distance_matrix)))

    def test_tsp_cost_improvement(self):
        """Cost should improve or stay stable."""
        distance_matrix = [
            [0, 2, 9],
            [2, 0, 10],
            [9, 10, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        _, _, _, _, _, _, cost_over_time = population_based_acoTSP(
            distance_matrix, city_coords, max_iterations=500
        )

        # Cost should not increase
        for i in range(len(cost_over_time) - 1):
            assert cost_over_time[i + 1] <= cost_over_time[i] + 0.01

    def test_archive_size_effect_tsp(self):
        """Different archive sizes should work for TSP."""
        distance_matrix = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        for archive_size in [5, 10, 20]:
            best_tour, _, _, _, _, _, _ = population_based_acoTSP(
                distance_matrix,
                city_coords,
                archive_size=archive_size,
                max_iterations=100,
            )
            assert len(best_tour) == 3

    def test_different_alpha_beta_paco(self):
        """Different alpha/beta values should work for PACO."""
        distance_matrix = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        for alpha in [0.5, 1, 2]:
            for beta in [1, 2]:
                best_tour, _, _, _, _, _, _ = population_based_acoTSP(
                    distance_matrix,
                    city_coords,
                    alpha=alpha,
                    beta=beta,
                    max_iterations=100,
                )
                assert set(best_tour) == {0, 1, 2}

    def test_tour_validity_paco(self):
        """Tour should visit each city exactly once."""
        distance_matrix = [
            [0, 2, 9, 10],
            [2, 0, 10, 5],
            [9, 10, 0, 3],
            [10, 5, 3, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)}

        best_tour, _, _, _, _, _, _ = population_based_acoTSP(
            distance_matrix, city_coords, max_iterations=500
        )

        assert set(best_tour) == set(range(4))
        assert len(best_tour) == 4
