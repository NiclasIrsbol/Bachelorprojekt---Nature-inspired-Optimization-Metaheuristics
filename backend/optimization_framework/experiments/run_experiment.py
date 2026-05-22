from optimization_framework.problems import onemax, leadingones, tsp
from optimization_framework.algorithms import (
    simulated_annealing, mu_plus_lambda_EA, one_plus_one_EA, ant_optimization_problem,
)
from optimization_framework.experiments.tsp_visualization import generate_tour_map
import json
from pathlib import Path
from functools import partial

FITNESS_FNS = {
    "onemax": onemax.fitnessOnemax,
    "leadingones": leadingones.fitnessLeadingOnes,
}

SOLVERS = {
    ("onemax", "(μ+λ) EA"): partial(mu_plus_lambda_EA.MuPlusLambdaEA, onemax.fitnessOnemax),
    ("onemax", "(1+1) EA"): partial(one_plus_one_EA.OnePlusOneEA, onemax.fitnessOnemax),
    ("onemax", "Simulated Annealing"): partial(simulated_annealing.simulated_annealing, onemax.fitnessOnemax),
    ("onemax", "ACO"): partial(ant_optimization_problem.ant_colony_optimization, onemax.fitnessOnemax),
    ("onemax", "PACO"): partial(ant_optimization_problem.population_based_aco, onemax.fitnessOnemax),
    ("leadingones", "(μ+λ) EA"): partial(mu_plus_lambda_EA.MuPlusLambdaEA, leadingones.fitnessLeadingOnes),
    ("leadingones", "(1+1) EA"): partial(one_plus_one_EA.OnePlusOneEA, leadingones.fitnessLeadingOnes),
    ("leadingones", "Simulated Annealing"): partial(simulated_annealing.simulated_annealing, leadingones.fitnessLeadingOnes),
    ("leadingones", "ACO"): partial(ant_optimization_problem.ant_colony_optimization, leadingones.fitnessLeadingOnes),
    ("leadingones", "PACO"): partial(ant_optimization_problem.population_based_aco, leadingones.fitnessLeadingOnes),
}

TSP_SOLVERS = {
    "(1+1) EA": one_plus_one_EA.OnePlusOneEATSP,
    "Simulated Annealing": simulated_annealing.simulated_annealingTSP,
    "(μ+λ) EA": mu_plus_lambda_EA.MuPlusLambdaEATSP,
    "ACO": ant_optimization_problem.ant_colony_optimizationTSP,
    "PACO": ant_optimization_problem.population_based_acoTSP,
}

DISPLAY_NAMES = {
    "(μ+λ) EA": "(μ+λ) EA",
    "(1+1) EA": "(1+1) EA",
    "Simulated Annealing": "Simulated Annealing",
    "ACO": "ACO",
    "PACO": "PACO",
}

THEORETICAL_RUNTIME = {
    ("onemax", "(μ+λ) EA"): "O(n log n)",
    ("onemax", "(1+1) EA"): "O(n log n)",
    ("onemax", "Simulated Annealing"): "O(n log n)",
    ("onemax", "ACO"): "O(n log n)",
    ("onemax", "PACO"): "O(n log n)",
    ("leadingones", "(μ+λ) EA"): "O(n²)",
    ("leadingones", "(1+1) EA"): "O(n²)",
    ("leadingones", "Simulated Annealing"): "O(n²)",
    ("leadingones", "ACO"): "O(n² log n)",
    ("leadingones", "PACO"): "O(n² log n)",
}


def _run_tsp(algorithm_name, params, tsp_instance=None):
    solver = TSP_SOLVERS.get(algorithm_name)
    if not solver:
        raise ValueError(f"Unknown TSP algorithm: {algorithm_name}")

    instance_name, _problem, city_coords, _nodes, distance_matrix = tsp.fetch_tsp_instance(tsp_instance)
    raw = solver(distance_matrix, city_coords, **params)

    if not isinstance(raw, tuple):
        raise TypeError(f"TSP solver must return a tuple, got {type(raw).__name__}")

    best_tour, iterations, temp, population, fitness_evaluations, tour_coords, cost_over_time = raw

    best_cost = tsp.tour_cost(best_tour, distance_matrix)

    if not population:
        population = {"Best": {"tour": best_tour, "cost": best_cost}}

    display_name = DISPLAY_NAMES.get(algorithm_name, algorithm_name)

    result = {
        "problem": "tsp",
        "algorithm": display_name,
        "tsp_instance": instance_name,
        "num_cities": len(distance_matrix),
        "iterations": iterations,
        "fitness_evaluations": fitness_evaluations,
        "theoretical_runtime": "NP-hard",
        "best_cost": best_cost,
        "best_tour": best_tour,
        "history": [{"Population": population}],
    }
    
    if algorithm_name == "Simulated Annealing":
        result["temp"] = temp

    if tour_coords is not None:
        result["coords"] = [{"x": x, "y": y} for x, y in tour_coords]

    city_coords_list = {}
    node_ids = sorted(city_coords.keys())
    for idx, nid in enumerate(node_ids):
        city_coords_list[str(idx)] = city_coords[nid]
    result["city_coords"] = city_coords_list

    if cost_over_time is not None:
        result["fitness_over_time"] = [
            {"generation": i, "fitness": c} for i, c in enumerate(cost_over_time)
        ]

    try:
        tour_image = generate_tour_map(
            city_coords_list, best_tour, distance_matrix, instance_name,
        )
        if tour_image:
            result["tour_map_image"] = tour_image
    except Exception:
        pass

    _save_result(result)
    return result


def _run_bitstring(problem_name, algorithm_name, params):
    solver = SOLVERS.get((problem_name, algorithm_name))
    if not solver:
        raise ValueError(f"Unknown combination: {problem_name} + {algorithm_name}")

    raw = solver(**params) if callable(solver) else solver
    if not isinstance(raw, tuple):
        raise TypeError(f"Solver must return a tuple, got {type(raw).__name__}")

    coords = None
    fitness_over_time = None
    if len(raw) == 7:
        best, iterations, temp, population, fitness_evaluations, coords, fitness_over_time = raw
    elif len(raw) == 6:
        best, iterations, temp, population, fitness_evaluations, coords = raw
    elif len(raw) == 5:
        best, iterations, population, fitness_evaluations, fitness_over_time = raw
        temp = 0.0
    elif len(raw) == 4:
        best, iterations, population, fitness_evaluations = raw
        temp = 0.0
    else:
        raise ValueError(f"Unexpected solver return arity: expected 4-7 values, got {len(raw)}")

    if not population:
        fitness_fn = FITNESS_FNS.get(problem_name)
        if fitness_fn is None:
            raise ValueError(f"Unknown problem for fitness mapping: {problem_name}")

        if isinstance(best, dict) and "bit" in best and "fitness" in best:
            population = {"Best": best}
        elif isinstance(best, str):
            population = {"Best": {"bit": best, "fitness": fitness_fn(best)}}
        else:
            population = {"Best": {"bit": str(best), "fitness": 0}}

    display_name = DISPLAY_NAMES.get(algorithm_name, algorithm_name)
    theoretical = THEORETICAL_RUNTIME.get(
        (problem_name, display_name), "unknown"
    )

    result = {
        "problem": problem_name,
        "algorithm": display_name,
        "iterations": iterations,
        "fitness_evaluations": fitness_evaluations,
        "theoretical_runtime": theoretical,
        "history": [{"Population": population}],
    }
    
    if algorithm_name == "Simulated Annealing":
        result["temp"] = temp

    if coords is not None:
        result["coords"] = [{"x": x, "y": y} for x, y in coords]

    if fitness_over_time is not None:
        result["fitness_over_time"] = [
            {"generation": i, "fitness": f} for i, f in enumerate(fitness_over_time)
        ]

    _save_result(result)
    return result


def _save_result(result):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "latest_run.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def main(problem_name="onemax", algorithm_name="(μ+λ) EA", params=None, tsp_instance=None):
    if params is None:
        params = {}

    if problem_name == "tsp":
        return _run_tsp(algorithm_name, params, tsp_instance)

    return _run_bitstring(problem_name, algorithm_name, params)


if __name__ == "__main__":
    import sys
    problem = sys.argv[1] if len(sys.argv) > 1 else "onemax"
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "(μ+λ) EA"
    result = main(problem, algorithm)
    print(f"Solved {result['problem']} with {result['algorithm']} in {result['iterations']} iterations, {result['fitness_evaluations']} fitness evaluations")
