"""
Benchmark experiment: MMAS (ACO) vs P-ACO on TSP across TSPLIB instances.

Organized like the (μ+λ) / ACO bitstring scaling scripts, but the x-axis is the
number of cities (instance size) and the primary metric is the GAP PERCENT above
the known optimal tour length, since TSP runs a fixed iteration budget rather
than terminating at a known optimum. There is no closed-form runtime curve for
TSP here, so the reference line is the optimum itself (gap % = 0).

Series:
  - MMAS  : ant_colony_optimizationTSP
  - P-ACO : population_based_acoTSP

Requires the TSPLIB clone (problems/tsplib). Instances that cannot be loaded are
skipped with a warning so the rest of the run still completes.

Output (under output/aco_scaling):
  - CSV : tsp_aco_scaling_results.csv
  - PNG : tsp_aco_scaling_comparison.png
  - JSON: tsp_aco_scaling_data.json

Usage:
    cd backend
    python3 -m optimization_framework.experiments.legacy_aco.tsp_aco_scaling
    # smoke run (tiny):
    python3 -m optimization_framework.experiments.legacy_aco.tsp_aco_scaling \
        --instances burma14 ulysses16 --seeds 2 --max-iterations 500
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.algorithms.ant_optimization_problem import (
    ant_colony_optimizationTSP,
    population_based_acoTSP,
)
from optimization_framework.problems import tsp


# Configuration
DEFAULT_INSTANCES = ["burma14", "ulysses16", "eil51", "berlin52"]
MAX_ITERATIONS = 1000
DEFAULT_SEEDS = 20
OUTPUT_DIR = Path("output/aco_scaling")

ALGO_CONFIGS = {
    "MMAS": (ant_colony_optimizationTSP, {"rho": 0.1, "alpha": 1, "beta": 2}),
    "P-ACO": (population_based_acoTSP, {"archive_size": 10, "num_ants": 30, "alpha": 1, "beta": 2, "q0": 0.9}),
}

COLORS = {
    "MMAS": "#3b82f6",
    "P-ACO": "#ef4444",
}


def run_single_trial(instance_name, distance_matrix, city_coords, algo_name, seed, max_iterations=MAX_ITERATIONS):
    """Run a single MMAS or P-ACO trial on a TSP instance."""
    solver, extra = ALGO_CONFIGS[algo_name]

    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    best_tour, iterations, _, _, fitness_evals, _coords, _curve = solver(
        distance_matrix, city_coords, max_iterations=max_iterations, **extra
    )
    elapsed = time.time() - start_time

    best_cost = tsp.tour_cost(best_tour, distance_matrix)
    gap = tsp.gap_percent(best_cost, instance_name)

    return {
        "instance": instance_name,
        "n_cities": len(distance_matrix),
        "algorithm": algo_name,
        "seed": seed,
        "best_cost": best_cost,
        "gap_percent": gap,
        "iterations": iterations,
        "fitness_evaluations": fitness_evals,
        "elapsed_seconds": elapsed,
    }


def run_experiment(instances: List[str] = None, seeds: int = DEFAULT_SEEDS, max_iterations: int = MAX_ITERATIONS):
    """Run the TSP benchmark across instances and algorithms.

    Returns (all_results, aggregated). aggregated is keyed by algorithm, then by
    n_cities, with mean/std of gap_percent and best_cost.
    """
    if instances is None:
        instances = DEFAULT_INSTANCES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    aggregated = {name: {} for name in ALGO_CONFIGS}

    print("Starting MMAS vs P-ACO TSP benchmark")
    print(f"  Instances: {instances}")
    print(f"  Algorithms: {list(ALGO_CONFIGS.keys())}")
    print(f"  Seeds per algorithm: {seeds}")
    print(f"  Max iterations: {max_iterations}")
    print()

    # Load instances first (skip any that fail to load).
    loaded = []
    for inst in instances:
        try:
            name, _problem, coords, _nodes, dm = tsp.fetch_tsp_instance(inst)
            loaded.append((name, coords, dm))
        except Exception as e:  # noqa: BLE001 - report and skip unavailable instances
            print(f"  WARNING: could not load instance '{inst}', skipping ({e})")

    if not loaded:
        print("No TSP instances could be loaded (is the tsplib clone present?).")
        return all_results, aggregated

    total_runs = len(loaded) * len(ALGO_CONFIGS) * seeds
    run_count = 0

    for name, coords, dm in loaded:
        n_cities = len(dm)
        for algo_name in ALGO_CONFIGS:
            run_results = []
            for seed in range(seeds):
                run_count += 1
                trial = run_single_trial(name, dm, coords, algo_name, seed, max_iterations)
                all_results.append(trial)
                run_results.append(trial)

                gap_str = "n/a" if trial["gap_percent"] is None else f"{trial['gap_percent']:6.2f}%"
                print(
                    f"[{run_count:3d}/{total_runs}] {name:10s} (n={n_cities:3d}), "
                    f"algo={algo_name:6s}, seed={seed:2d}: cost={trial['best_cost']:10.1f}, gap={gap_str}"
                )

            gaps = [r["gap_percent"] for r in run_results if r["gap_percent"] is not None]
            costs = [r["best_cost"] for r in run_results]
            aggregated[algo_name][n_cities] = {
                "instance": name,
                "mean_gap": float(np.mean(gaps)) if gaps else None,
                "std_gap": float(np.std(gaps)) if gaps else None,
                "mean_cost": float(np.mean(costs)) if costs else None,
                "std_cost": float(np.std(costs)) if costs else None,
                "trials": len(run_results),
            }

    print(f"\nExperiment complete. {run_count} trials run.")
    return all_results, aggregated


def save_results(all_results: List[Dict], aggregated: Dict):
    """Save results to CSV and JSON."""
    csv_file = OUTPUT_DIR / "tsp_aco_scaling_results.csv"
    json_file = OUTPUT_DIR / "tsp_aco_scaling_data.json"

    fieldnames = [
        "instance", "n_cities", "algorithm", "seed",
        "best_cost", "gap_percent", "iterations",
        "fitness_evaluations", "elapsed_seconds",
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Saved raw results to {csv_file}")

    with open(json_file, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    print(f"Saved aggregated results to {json_file}")


def plot_results(aggregated: Dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    plotted_any = False

    for algo_name, color in COLORS.items():
        if algo_name not in aggregated:
            continue

        data = aggregated[algo_name]
        xs, means, stds = [], [], []
        for n_cities in sorted(data.keys()):
            entry = data[n_cities]
            if entry["mean_gap"] is None:
                continue
            xs.append(n_cities)
            means.append(entry["mean_gap"])
            stds.append(entry["std_gap"] or 0.0)

        if xs:
            plotted_any = True
            ax.errorbar(
                xs, means, yerr=stds,
                label=f"{algo_name}",
                marker="o", markersize=8, color=color,
                linewidth=2.5, capsize=5, capthick=1.5, alpha=0.8,
            )

    # Optimal reference (gap % = 0).
    ax.axhline(0.0, linestyle="--", linewidth=2.0, color="#64748b", alpha=0.7,
               label="Optimal (gap = 0%)")

    ax.set_xlabel("Number of Cities (n)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gap above optimum (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "MMAS vs P-ACO on TSP: Solution Quality vs Instance Size",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plot_file = OUTPUT_DIR / "tsp_aco_scaling_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if plot_file.exists() and plotted_any:
        print(f"Saved plot to {plot_file}")
    elif not plotted_any:
        print("No gap data to plot (instances may lack known optima or failed to load).")
    else:
        print("WARNING: plot file was not created")


def print_summary(aggregated: Dict):
    print("\n" + "=" * 80)
    print("SUMMARY: MMAS vs P-ACO on TSP")
    print("=" * 80)

    for algo_name in ALGO_CONFIGS:
        if algo_name not in aggregated:
            continue

        print(f"\n{algo_name}:")
        print("-" * 80)
        print(f"{'instance':>12} | {'n':>4} | {'Mean gap %':>12} | {'Std gap %':>12} | {'Mean cost':>12}")
        print("-" * 80)

        data = aggregated[algo_name]
        for n_cities in sorted(data.keys()):
            e = data[n_cities]
            mean_gap = "n/a" if e["mean_gap"] is None else f"{e['mean_gap']:12.2f}"
            std_gap = "n/a" if e["std_gap"] is None else f"{e['std_gap']:12.2f}"
            mean_cost = "n/a" if e["mean_cost"] is None else f"{e['mean_cost']:12.1f}"
            print(f"{e['instance']:>12} | {n_cities:4d} | {mean_gap} | {std_gap} | {mean_cost}")


def main():
    parser = argparse.ArgumentParser(
        description="MMAS vs P-ACO TSP benchmark across TSPLIB instances"
    )
    parser.add_argument(
        "--instances", nargs="+", default=None,
        help=f"TSPLIB instance names (default: {DEFAULT_INSTANCES})",
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help=f"Number of independent runs per algorithm (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help=f"Max iterations per run (default: {MAX_ITERATIONS})",
    )
    args = parser.parse_args()

    instances = args.instances if args.instances else DEFAULT_INSTANCES
    all_results, aggregated = run_experiment(
        instances=instances, seeds=args.seeds, max_iterations=args.max_iterations
    )
    save_results(all_results, aggregated)
    plot_results(aggregated)
    print_summary(aggregated)


if __name__ == "__main__":
    main()
