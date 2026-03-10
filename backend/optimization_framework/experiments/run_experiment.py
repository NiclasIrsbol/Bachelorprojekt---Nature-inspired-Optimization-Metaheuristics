from optimization_framework.problems import onemax, leadingones
from optimization_framework.algorithms import simulated_annealing, mu_plus_lambda_EA, one_plus_one_EA, ant_optimization_problem
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
    ("leadingones", "(μ+λ) EA"): partial(mu_plus_lambda_EA.MuPlusLambdaEA, leadingones.fitnessLeadingOnes),
    ("leadingones", "(1+1) EA"): partial(one_plus_one_EA.OnePlusOneEA, leadingones.fitnessLeadingOnes),
    ("leadingones", "Simulated Annealing"): partial(simulated_annealing.simulated_annealing, leadingones.fitnessLeadingOnes),
    ("onemax", "ACO"): partial(ant_optimization_problem.ant_colony_optimization, onemax.fitnessOnemax),
    ("leadingones", "ACO"): partial(ant_optimization_problem.ant_colony_optimization, leadingones.fitnessLeadingOnes),
}

DISPLAY_NAMES = {
    "(μ+λ) EA": "(μ+λ) EA",
    "(1+1) EA": "(1+1) EA",
    "Simulated Annealing": "Simulated Annealing",
    "ACO": "ACO"
}

THEORETICAL_RUNTIME = {
    ("onemax", "(μ+λ) EA"): "O(n log n)",
    ("onemax", "(1+1) EA"): "O(n log n)",
    ("onemax", "Simulated Annealing"): "O(n log n)",
    ("leadingones", "(μ+λ) EA"): "O(n\u00b2)",
    ("leadingones", "(1+1) EA"): "O(n\u00b2)",
    ("leadingones", "Simulated Annealing"): "O(n\u00b2)",
    ("onemax", "ACO"): "O(n log n)",
    ("leadingones", "ACO"): "O(n\u00b2 log n)",
}

def main(problem_name="onemax", algorithm_name="(μ+λ) EA", params={}):
    if params is None:
        params = {}
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
        "temp": temp,
        "fitness_evaluations": fitness_evaluations,
        "theoretical_runtime": theoretical,
        "history": [
            {
                "Population": population
            },
        ],
    }

    if coords is not None:
        result["coords"] = [{"x": x, "y": y} for x, y in coords]

    if fitness_over_time is not None:
        result["fitness_over_time"] = [
            {"generation": i, "fitness": f} for i, f in enumerate(fitness_over_time)
        ]

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "latest_run.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    import sys
    problem = sys.argv[1] if len(sys.argv) > 1 else "onemax"
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "(μ+λ) EA"
    result = main(problem, algorithm)
    print(f"Solved {result['problem']} with {result['algorithm']} in {result['iterations']} iterations, {result['fitness_evaluations']} fitness evaluations")
