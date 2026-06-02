"""Batch experiment runner.

Runs the configured algorithms across multiple seeds on each problem and
records per-run metrics plus convergence curves. Outputs are written to
``output/batch/`` and consumed by ``batch_analysis.py``.

Reproducibility: the algorithms use Python's global ``random`` module and take
no seed argument, so this runner calls ``random.seed(seed)`` (and seeds numpy)
before every individual run.

Usage:
    cd backend
    python -m optimization_framework.experiments.batch_runner --seeds 30 --bit-length 50
    # opt-in TSP (requires the tsplib clone to be present):
    python -m optimization_framework.experiments.batch_runner \
        --problems tsp --tsp-instances burma14 ulysses16 --tsp-max-iterations 1000
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from optimization_framework.experiments.run_experiment import SOLVERS, TSP_SOLVERS
from optimization_framework.problems import tsp

# --- Defaults -------------------------------------------------------------

BITSTRING_PROBLEMS = ["onemax", "leadingones"]
ALGORITHMS = ["(1+1) EA", "(μ+λ) EA", "Simulated Annealing", "ACO", "P-ACO"]

DEFAULT_SEEDS = 30
DEFAULT_BIT_LENGTH = 50
DEFAULT_OUTPUT_DIR = "output/batch"

RAW_RESULTS_FILE = "raw_results.csv"
CURVES_FILE = "curves.json"

RAW_FIELDNAMES = [
    "problem",
    "algorithm",
    "instance",
    "seed",
    "best_value",
    "reached_optimum",
    "iterations",
    "fitness_evaluations",
    "runtime_seconds",
]


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def _extract(raw):
    """Extract (best_value, iterations, fitness_evaluations, curve) from a
    solver return tuple.

    All current solvers return a 7-tuple of the shape
    ``(best, iterations, temp, population, fitness_evaluations, coords, curve)``
    where ``curve`` is the scalar fitness/cost-over-time list. We read the
    convergence curve as the last element and treat its final value as the
    best value reached.
    """
    if not isinstance(raw, tuple):
        raise TypeError(f"Solver must return a tuple, got {type(raw).__name__}")

    iterations = raw[1]
    fitness_evaluations = raw[4]
    curve = raw[-1]

    # Guard: only accept a scalar (numeric) curve; coordinate lists are tuples.
    if not curve or isinstance(curve[0], (tuple, list)):
        curve = []

    best_value = curve[-1] if curve else None
    return best_value, iterations, fitness_evaluations, curve


def _run_bitstring(problem, algorithm, bit_length, seed, extra_params=None):
    solver = SOLVERS.get((problem, algorithm))
    if solver is None:
        raise ValueError(f"Unknown combination: {problem} + {algorithm}")

    params = {"bit_length": bit_length}
    if extra_params:
        params.update(extra_params)

    _seed_everything(seed)
    start = time.perf_counter()
    raw = solver(**params)
    runtime = time.perf_counter() - start

    best_value, iterations, fitness_evaluations, curve = _extract(raw)
    reached_optimum = best_value == bit_length

    row = {
        "problem": problem,
        "algorithm": algorithm,
        "instance": "",
        "seed": seed,
        "best_value": best_value,
        "reached_optimum": reached_optimum,
        "iterations": iterations,
        "fitness_evaluations": fitness_evaluations,
        "runtime_seconds": runtime,
    }
    return row, curve


def _run_tsp(algorithm, instance_name, distance_matrix, city_coords, tsp_params, seed):
    solver = TSP_SOLVERS.get(algorithm)
    if solver is None:
        raise ValueError(f"Unknown TSP algorithm: {algorithm}")

    _seed_everything(seed)
    start = time.perf_counter()
    raw = solver(distance_matrix, city_coords, **tsp_params)
    runtime = time.perf_counter() - start

    best_value, iterations, fitness_evaluations, curve = _extract(raw)

    row = {
        "problem": "tsp",
        "algorithm": algorithm,
        "instance": instance_name,
        "seed": seed,
        "best_value": best_value,
        "reached_optimum": "",  # not applicable for TSP
        "iterations": iterations,
        "fitness_evaluations": fitness_evaluations,
        "runtime_seconds": runtime,
    }
    return row, curve


def run_batch(
    problems,
    algorithms,
    seeds,
    bit_length=DEFAULT_BIT_LENGTH,
    tsp_instances=None,
    tsp_max_iterations=None,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    tsp_instances = tsp_instances or []
    tsp_params = {}
    if tsp_max_iterations is not None:
        tsp_params["max_iterations"] = tsp_max_iterations

    rows = []
    curves = []

    bitstring_problems = [p for p in problems if p != "tsp"]
    run_tsp = "tsp" in problems and tsp_instances

    # --- Bitstring problems ---
    for problem in bitstring_problems:
        for algorithm in algorithms:
            for seed in seeds:
                print(f"[bitstring] {problem} | {algorithm} | seed={seed}")
                row, curve = _run_bitstring(problem, algorithm, bit_length, seed)
                rows.append(row)
                curves.append(
                    {
                        "problem": problem,
                        "algorithm": algorithm,
                        "instance": "",
                        "seed": seed,
                        "curve": curve,
                    }
                )

    # --- TSP ---
    if "tsp" in problems and not tsp_instances:
        print("Note: 'tsp' requested but no --tsp-instances given; skipping TSP.")

    if run_tsp:
        for instance_name in tsp_instances:
            print(f"[tsp] loading instance {instance_name}")
            name, _problem, coords, _nodes, distance_matrix = tsp.fetch_tsp_instance(
                instance_name
            )
            for algorithm in algorithms:
                for seed in seeds:
                    print(f"[tsp] {name} | {algorithm} | seed={seed}")
                    row, curve = _run_tsp(
                        algorithm, name, distance_matrix, coords, tsp_params, seed
                    )
                    rows.append(row)
                    curves.append(
                        {
                            "problem": "tsp",
                            "algorithm": algorithm,
                            "instance": name,
                            "seed": seed,
                            "curve": curve,
                        }
                    )

    _write_outputs(rows, curves, output_dir)
    return rows, curves


def _write_outputs(rows, curves, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_path = out / RAW_RESULTS_FILE
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    curves_path = out / CURVES_FILE
    with open(curves_path, "w") as f:
        json.dump(curves, f)

    print(f"\nWrote {len(rows)} runs to {raw_path}")
    print(f"Wrote {len(curves)} convergence curves to {curves_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Batch experiment runner.")
    parser.add_argument(
        "--problems",
        nargs="+",
        default=BITSTRING_PROBLEMS,
        help="Problems to run (onemax, leadingones, tsp).",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=ALGORITHMS,
        help="Algorithms to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help="Number of seeds; runs use seeds 0..N-1.",
    )
    parser.add_argument(
        "--bit-length",
        type=int,
        default=DEFAULT_BIT_LENGTH,
        help="Bitstring length for OneMax/LeadingOnes.",
    )
    parser.add_argument(
        "--tsp-instances",
        nargs="*",
        default=[],
        help="TSPLIB instance names (opt-in; requires the tsplib clone).",
    )
    parser.add_argument(
        "--tsp-max-iterations",
        type=int,
        default=None,
        help="Override max_iterations for all TSP solvers (fairness).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    seeds = list(range(args.seeds))
    run_batch(
        problems=args.problems,
        algorithms=args.algorithms,
        seeds=seeds,
        bit_length=args.bit_length,
        tsp_instances=args.tsp_instances,
        tsp_max_iterations=args.tsp_max_iterations,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
