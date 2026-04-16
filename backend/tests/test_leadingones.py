"""Comprehensive tests for LeadingOnes problem and (μ+λ) EA algorithm."""
import pytest
from optimization_framework.problems.leadingones import fitnessLeadingOnes
from optimization_framework.algorithms.mu_plus_lambda_EA import MuPlusLambdaEA, MuPlusLambdaEATSP
from optimization_framework.problems.tsp import tour_cost


class TestLeadingOnesProblem:
    """Test LeadingOnes objective function."""

    def test_all_ones(self):
        """Fitness should equal bitstring length for all ones."""
        assert fitnessLeadingOnes("11111") == 5
        assert fitnessLeadingOnes("1" * 10) == 10
        assert fitnessLeadingOnes("1" * 50) == 50

    def test_all_zeros(self):
        """Fitness should be 0 for all zeros."""
        assert fitnessLeadingOnes("00000") == 0
        assert fitnessLeadingOnes("0" * 10) == 0

    def test_leading_ones_stop_at_first_zero(self):
        """LeadingOnes should stop at first zero."""
        assert fitnessLeadingOnes("11011") == 2
        assert fitnessLeadingOnes("11101") == 3
        assert fitnessLeadingOnes("10001") == 1
        assert fitnessLeadingOnes("100") == 1

    def test_empty_string(self):
        """Empty string should have fitness 0."""
        assert fitnessLeadingOnes("") == 0

    def test_single_bit(self):
        """Single bit tests."""
        assert fitnessLeadingOnes("0") == 0
        assert fitnessLeadingOnes("1") == 1


class TestMuPlusLambdaEABitstring:
    """Test (μ+λ) EA for bitstring optimization."""

    def test_convergence_leadingones(self):
        """(μ+λ) EA should converge to optimum on LeadingOnes."""
        bit_length = 12
        best, iterations, temp, population, evals, coords, fitness_history = MuPlusLambdaEA(
            fitnessLeadingOnes,
            bit_length=bit_length,
            mu_size=10,
            lambda_size=20,
            tournament_k=3,
        )

        # best is a dict with "bit" and "fitness" keys
        assert isinstance(best, dict)
        best_bit = best["bit"]
        assert fitnessLeadingOnes(best_bit) == bit_length
        assert best_bit == "1" * bit_length

        # Check return types
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0
        assert isinstance(coords, list)
        assert isinstance(fitness_history, list)
        assert len(fitness_history) == iterations + 1

    def test_convergence_onemax(self):
        """(μ+λ) EA should also solve OneMax."""
        from optimization_framework.problems.onemax import fitnessOnemax
        
        bit_length = 10
        best, iterations, _, _, evals, _, _ = MuPlusLambdaEA(
            fitnessOnemax,
            bit_length=bit_length,
            mu_size=15,
            lambda_size=30,
        )

        # best is a dict with "bit" and "fitness" keys
        assert isinstance(best, dict)
        assert fitnessOnemax(best["bit"]) == bit_length

    def test_fitness_history_monotonic(self):
        """Fitness history should be monotonically non-decreasing."""
        _, _, _, _, _, _, fitness_history = MuPlusLambdaEA(
            fitnessLeadingOnes, bit_length=8
        )
        for i in range(len(fitness_history) - 1):
            assert fitness_history[i + 1] >= fitness_history[i]

    def test_population_size_effect(self):
        """Different population sizes should work."""
        # Small population
        best1, _, _, _, _, _, _ = MuPlusLambdaEA(
            fitnessLeadingOnes, bit_length=10, mu_size=5, lambda_size=10
        )
        # Large population
        best2, _, _, _, _, _, _ = MuPlusLambdaEA(
            fitnessLeadingOnes, bit_length=10, mu_size=30, lambda_size=60
        )

        # Both should return valid bitstrings (in dict format)
        assert isinstance(best1, dict) and isinstance(best1["bit"], str)
        assert isinstance(best2, dict) and isinstance(best2["bit"], str)

    def test_tournament_parameter(self):
        """Different tournament k values should be accepted."""
        for k in [2, 3, 5, 10]:
            best, _, _, _, _, _, _ = MuPlusLambdaEA(
                fitnessLeadingOnes, bit_length=8, tournament_k=k
            )
            # best is a dict with "bit" key
            assert isinstance(best, dict)
            assert isinstance(best["bit"], str)
            assert all(b in "01" for b in best["bit"])

    def test_population_dict_structure(self):
        """Population dict should have correct structure."""
        best, _, _, population, _, _, _ = MuPlusLambdaEA(
            fitnessLeadingOnes, bit_length=6
        )
        assert isinstance(population, dict)
        assert len(population) > 0

    def test_coords_length_matches_iterations(self):
        """Coords should have entry for each iteration."""
        _, iterations, _, _, _, coords, _ = MuPlusLambdaEA(
            fitnessLeadingOnes, bit_length=8
        )
        assert len(coords) == iterations + 1

    def test_mutation_probability(self):
        """Different mutation probabilities should work."""
        for mut_prob in [0.01, 0.05, 0.1, 0.5]:
            best, iterations, _, _, _, _, _ = MuPlusLambdaEA(
                fitnessLeadingOnes,
                bit_length=8,
                mutation_prob=mut_prob,
            )
            # best is a dict with "bit" and "fitness" keys
            assert isinstance(best, dict)
            assert isinstance(best["bit"], str)
            assert all(b in "01" for b in best["bit"])


class TestMuPlusLambdaEATSP:
    """Test (μ+λ) EA for TSP."""

    def test_tsp_convergence(self):
        """(μ+λ) EA should find reasonable tours."""
        distance_matrix = [
            [0, 1, 4, 5],
            [1, 0, 2, 3],
            [4, 2, 0, 1],
            [5, 3, 1, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

        best_tour, iterations, temp, population, evals, tour_coords, cost_over_time = MuPlusLambdaEATSP(
            distance_matrix,
            city_coords,
            mu_size=10,
            lambda_size=20,
            max_iterations=500,
        )

        # Verify return types
        assert isinstance(best_tour, list)
        assert len(best_tour) == len(distance_matrix)
        assert isinstance(iterations, int) and iterations > 0
        assert isinstance(evals, int) and evals > 0

        # Tour should be valid
        assert len(set(best_tour)) == len(distance_matrix)

    def test_cost_improves_or_stable(self):
        """Cost should not increase over time."""
        distance_matrix = [
            [0, 2, 9, 10],
            [2, 0, 10, 5],
            [9, 10, 0, 3],
            [10, 5, 3, 0],
        ]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0)}

        _, _, _, _, _, _, cost_over_time = MuPlusLambdaEATSP(
            distance_matrix, city_coords, max_iterations=100
        )

        # Cost should not increase
        for i in range(len(cost_over_time) - 1):
            assert cost_over_time[i + 1] <= cost_over_time[i] + 0.01  # Small tolerance for float

    def test_different_population_sizes(self):
        """Different population sizes should work."""
        distance_matrix = [[0, 1, 4], [1, 0, 2], [4, 2, 0]]
        city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1)}

        for mu_size in [5, 10, 20]:
            best_tour, _, _, _, _, _, _ = MuPlusLambdaEATSP(
                distance_matrix,
                city_coords,
                mu_size=mu_size,
                lambda_size=mu_size * 2,
                max_iterations=100,
            )
            assert len(best_tour) == 3
            assert set(best_tour) == {0, 1, 2}
