from optimization_framework.problems import onemax
import json
from pathlib import Path

def main():
    population = onemax.generateBitstrings(5)
    result = {
        "problem": "onemax",
        "algorithm": "genetic algorithm",
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

    print(f"Saved result to {output_file}")

if __name__ == "__main__":
    main()