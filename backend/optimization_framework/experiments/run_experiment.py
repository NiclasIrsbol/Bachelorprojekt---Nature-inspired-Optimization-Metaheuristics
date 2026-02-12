from optimization_framework.problems import onemax, leadingones
import json
from pathlib import Path

PROBLEMS = {
    "onemax": onemax.onemax,
    "leadingones": leadingones.leadingones,
}

def main(problem_name="onemax"):
    solver = PROBLEMS.get(problem_name)
    if not solver:
        raise ValueError(f"Unknown problem: {problem_name}")

    best, iterations, population = solver()

    result = {
        "problem": problem_name,
        "algorithm": "genetic algorithm",
        "iterations": iterations,
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
    result = main(problem)
    print(f"Solved {result['problem']} in {result['iterations']} iterations")
