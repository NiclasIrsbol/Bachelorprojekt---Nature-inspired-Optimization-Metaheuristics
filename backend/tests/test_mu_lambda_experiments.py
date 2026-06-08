"""Smoke tests for GA-style (mu+lambda) report experiments."""

from optimization_framework.algorithms.mu_plus_lambda_EA import MuPlusLambdaEA, MuPlusLambdaGA
from optimization_framework.experiments import (
    crossover_comparison,
    onemax_mu_lambda_scaling,
    leadingones_mu_lambda_scaling,
    tsp_mu_lambda_berlin52,
)
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.problems.leadingones import fitnessLeadingOnes


def test_mu_plus_lambda_ea_accepts_crossover_type():
    best, iterations, _, population, evals, coords, history = MuPlusLambdaEA(
        fitnessOnemax,
        bit_length=12,
        mu_size=3,
        lambda_size=6,
        mutation_prob=1 / 12,
        max_iterations=2000,
        crossover_type="uniform",
    )

    assert isinstance(best, dict)
    assert best["fitness"] == 12
    assert iterations >= 0
    assert len(population) == 3
    assert evals >= 3
    assert len(coords) == iterations + 1
    assert len(history) == iterations + 1


def test_mu_plus_lambda_ga_alias_matches_framework_variant():
    best, *_ = MuPlusLambdaGA(
        fitnessLeadingOnes,
        bit_length=8,
        mu_size=3,
        lambda_size=6,
        mutation_prob=1 / 8,
        max_iterations=3000,
        crossover_type="single_point",
    )

    assert isinstance(best, dict)
    assert best["fitness"] == 8


def test_onemax_mu_lambda_scaling_single_trial_smoke():
    row = onemax_mu_lambda_scaling.run_single_trial(
        n=12,
        mu_size=2,
        lambda_size=4,
        seed=0,
        max_iterations=3000,
    )

    assert row["n"] == 12
    assert row["mu"] == 2
    assert row["lambda"] == 4
    assert row["reached_optimum"]
    assert row["fitness_evaluations"] >= 2


def test_leadingones_mu_lambda_scaling_single_trial_smoke():
    row = leadingones_mu_lambda_scaling.run_single_trial(
        n=8,
        mu_size=2,
        lambda_size=4,
        seed=1,
        max_iterations=5000,
    )

    assert row["n"] == 8
    assert row["mu"] == 2
    assert row["lambda"] == 4
    assert row["reached_optimum"]
    assert row["fitness_evaluations"] >= 2


def test_crossover_comparison_import_and_trial_smoke():
    row = crossover_comparison.run_single_trial(
        problem="onemax",
        crossover_type="two_point",
        bit_length=10,
        seed=2,
        mu_size=2,
        lambda_size=4,
        max_iterations=3000,
    )

    assert row["problem"] == "onemax"
    assert row["crossover"] == "two_point"
    assert row["reached_optimum"]
    assert isinstance(row["fitness_history"], list)
    assert row["fitness_history"][-1] == row["final_fitness"]


def test_crossover_comparison_curve_rows_smoke():
    rows = [
        crossover_comparison.run_single_trial(
            problem="onemax",
            crossover_type="single_point",
            bit_length=10,
            seed=0,
            mu_size=2,
            lambda_size=4,
            max_iterations=3000,
        ),
        crossover_comparison.run_single_trial(
            problem="onemax",
            crossover_type="single_point",
            bit_length=10,
            seed=1,
            mu_size=2,
            lambda_size=4,
            max_iterations=3000,
        ),
    ]

    curve_rows = crossover_comparison.build_curve_rows(rows, max_points=10)

    assert curve_rows
    assert curve_rows[0]["problem"] == "onemax"
    assert curve_rows[0]["crossover"] == "single_point"
    assert curve_rows[0]["fitness_evaluations"] == 2
    assert curve_rows[-1]["mean_fitness"] == 10


def test_tsp_mu_lambda_budget_helpers_and_trial_smoke():
    distance_matrix = [
        [0, 1, 4, 5],
        [1, 0, 2, 3],
        [4, 2, 0, 1],
        [5, 3, 1, 0],
    ]
    city_coords = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}

    generations = tsp_mu_lambda_berlin52.max_generations_for_budget(
        mu_size=2,
        lambda_size=4,
        eval_budget=50,
    )
    assert generations == 12

    row = tsp_mu_lambda_berlin52.run_single_trial(
        distance_matrix=distance_matrix,
        city_coords=city_coords,
        instance_name="unknown-test-instance",
        config_name="test",
        mu_size=2,
        lambda_size=4,
        seed=3,
        eval_budget=50,
    )

    assert row["config"] == "test"
    assert row["fitness_evaluations"] == 2 + 12 * 4
    assert row["best_cost"] > 0
    assert isinstance(row["cost_history"], list)
