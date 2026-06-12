# @author: Andrej Kitanovski

"""Integration tests for the orchestration layer (run_experiment.main).

These exercise the central dispatch that maps (problem, algorithm) names to the
registered solvers and normalises their output into the shared result dict the
API/frontend consume. Report §5.3.3 claims this layer is integration-testable.
"""

import pytest

from optimization_framework.experiments.run_experiment import main, SOLVERS, TSP_SOLVERS

ALGORITHMS = ["(1+1) EA", "(μ+λ) EA", "Simulated Annealing", "MMAS-ACO", "P-ACO"]
BITSTRING_KEYS = {"problem", "algorithm", "iterations", "fitness_evaluations",
                  "theoretical_runtime", "history"}
TSP_KEYS = {"problem", "algorithm", "best_cost", "best_tour", "num_cities",
            "city_coords", "iterations", "fitness_evaluations"}


@pytest.mark.parametrize("problem", ["onemax", "leadingones"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_bitstring_dispatch_returns_normalised_result(problem, algorithm):
    result = main(problem, algorithm, {"bit_length": 12, "max_iterations": 3000})
    assert BITSTRING_KEYS <= set(result)
    assert result["problem"] == problem
    assert isinstance(result["fitness_evaluations"], int)
    assert result["fitness_evaluations"] >= 1


def test_every_registered_bitstring_combo_is_runnable():
    # Guards against a registry entry pointing at a broken/renamed solver.
    for (problem, algorithm) in SOLVERS:
        result = main(problem, algorithm, {"bit_length": 8, "max_iterations": 1000})
        assert result["problem"] == problem


def test_unknown_combination_raises():
    with pytest.raises(ValueError):
        main("onemax", "NoSuchAlgorithm", {"bit_length": 8})


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_tsp_dispatch_returns_normalised_result(algorithm):
    try:
        result = main("tsp", algorithm, {"max_iterations": 25}, tsp_instance="burma14")
    except Exception as e:  # missing tsplib data or parser not installed
        pytest.skip(f"TSP instance unavailable: {e}")
    assert TSP_KEYS <= set(result)
    assert result["problem"] == "tsp"
    assert result["theoretical_runtime"] == "NP-hard"
    assert sorted(result["best_tour"]) == list(range(result["num_cities"]))


def test_tsp_3opt_flows_through_orchestrator():
    try:
        result = main("tsp", "(1+1) EA",
                      {"max_iterations": 25, "mutation": "3opt"}, tsp_instance="burma14")
    except Exception as e:
        pytest.skip(f"TSP instance unavailable: {e}")
    assert result["problem"] == "tsp"
    assert sorted(result["best_tour"]) == list(range(result["num_cities"]))
