import json
from pathlib import Path

def main():
    result = {
        "problem": "tsp",
        "algorithm": "simulated_annealing",
        "history": [
            { "iteration": 0, "cost": 100 },
            { "iteration": 1, "cost": 90 },
            { "iteration": 2, "cost": 80 }
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