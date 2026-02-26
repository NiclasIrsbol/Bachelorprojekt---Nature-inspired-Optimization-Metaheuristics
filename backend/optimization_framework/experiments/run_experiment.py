from optimization_framework.problems import onemax, leadingones
import json
from pathlib import Path

SOLVERS = {
    ("onemax", "(μ+λ) EA"):            onemax.onemaxMuPlusLambdaEA,
    ("onemax", "(1+1) EA"):      onemax.onemaxOnePlusOneEA,
    ("leadingones", "(μ+λ) EA"):       leadingones.leadingonesMuPlusLambdaEA,
    ("leadingones", "(1+1) EA"): leadingones.leadingonesOnePlusOneEA,
}

DISPLAY_NAMES = {
    "(μ+λ) EA": "(μ+λ) EA",
    "(1+1) EA": "(1+1) EA",
}

THEORETICAL_RUNTIME = {
    ("onemax", "(μ+λ) EA"):  "O(n log n)",
    ("onemax", "(1+1) EA"):           "O(n log n)",
    ("leadingones", "(μ+λ) EA"): "O(n\u00b2)",
    ("leadingones", "(1+1) EA"):         "O(n\u00b2)",
}

def main(problem_name="onemax", algorithm_name="(μ+λ) EA"):
    solver = SOLVERS.get((problem_name, algorithm_name))
    if not solver:
        raise ValueError(f"Unknown combination: {problem_name} + {algorithm_name}")

    best, iterations, population, fitness_evaluations = solver()

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
        "history": [
            {
                "Population": population
            },
        ]
    }

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
