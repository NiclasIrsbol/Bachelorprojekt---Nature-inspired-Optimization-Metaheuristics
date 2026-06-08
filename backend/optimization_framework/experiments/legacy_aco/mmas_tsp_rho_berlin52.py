"""
Report section 8.4.2 (MMAS, TSP): evaporation-rate (rho) sweep on berlin52.

Runs MMAS (``ant_colony_optimizationTSP``) on the berlin52 instance for several
evaporation rates rho and plots the convergence of tour length vs evaluations,
one line per rho, with a dashed line at the known optimum (7542).
Each generation constructs ``num_ants`` tours, and all constructions count as
fitness evaluations.

Following Stützle & Hoos, the pheromone bounds are tied to the known optimal
tour length L*: tau_max = 1/(rho·L*), tau_min = tau_max/(2n). These are passed
to the solver via its ``optimum`` argument.

Output (under output/report_aco):
  - CSV : mmas_tsp_berlin52_raw.csv
  - PNG : mmas_tsp_berlin52_convergence.png
  - JSON: mmas_tsp_berlin52_summary.json

Requires the TSPLIB clone (problems/tsplib). If berlin52 cannot be loaded the
script prints a clear message and exits without error.

Usage:
    cd backend
    python3 -m optimization_framework.experiments.legacy_aco.mmas_tsp_rho_berlin52
    # smoke run (tiny):
    python3 -m optimization_framework.experiments.legacy_aco.mmas_tsp_rho_berlin52 \
        --seeds 2 --max-iterations 500
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

from optimization_framework.algorithms.ant_optimization_problem import ant_colony_optimizationTSP
from optimization_framework.problems import tsp


# Configuration
INSTANCE = "berlin52"
RHO_VALUES = [0.01, 0.05, 0.1, 0.5, 0.9]
NUM_ANTS = 10
DEFAULT_SEEDS = 10
MAX_ITERATIONS = 2000  # generations; x num_ants evaluations per generation
OUTPUT_DIR = Path("output/report_aco")

COLORS = {
    0.01: "#3b82f6",
    0.05: "#f59e0b",
    0.1: "#10b981",
    0.5: "#ef4444",
    0.9: "#a855f7",
}


def run_single_trial(distance_matrix, city_coords, rho, seed, optimum,
                     max_iterations=MAX_ITERATIONS, num_ants=NUM_ANTS):
    """Run a single MMAS trial on the TSP instance with evaporation rate rho."""
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    best_tour, iterations, _, _, fitness_evals, _coords, cost_over_time = ant_colony_optimizationTSP(
        distance_matrix, city_coords, rho=rho, max_iterations=max_iterations,
        optimum=optimum, num_ants=num_ants,
    )
    elapsed = time.time() - start_time

    best_cost = tsp.tour_cost(best_tour, distance_matrix)
    gap = tsp.gap_percent(best_cost, INSTANCE)

    return {
        "instance": INSTANCE,
        "rho": rho,
        "num_ants": num_ants,
        "seed": seed,
        "best_cost": best_cost,
        "gap_percent": gap,
        "iterations": iterations,
        "fitness_evaluations": fitness_evals,
        "elapsed_seconds": elapsed,
        "_curve": cost_over_time,
    }


def run_experiment(rhos: List[float] = None, seeds: int = DEFAULT_SEEDS,
                   max_iterations: int = MAX_ITERATIONS, num_ants: int = NUM_ANTS):
    """Run the rho sweep on berlin52.

    Returns (all_results, aggregated, curves). ``curves[rho]`` is a list of
    cost-over-time arrays (one per seed). ``all_results`` rows exclude the raw
    curve. Returns ([], {}, {}) if the instance cannot be loaded.
    """
    if rhos is None:
        rhos = RHO_VALUES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        name, _problem, coords, _nodes, dm = tsp.fetch_tsp_instance(INSTANCE)
    except Exception as e:  # noqa: BLE001
        print(f"Could not load TSP instance '{INSTANCE}' ({e}). Is the tsplib clone present?")
        return [], {}, {}

    optimum = tsp.get_optimum(INSTANCE)

    all_results = []
    aggregated = {rho: {} for rho in rhos}
    curves = {rho: [] for rho in rhos}

    print(f"Starting MMAS rho-sweep on {name} (n={len(dm)}, optimum={optimum})")
    print(f"  rho values: {rhos}")
    print(f"  num_ants={num_ants}")
    print(f"  Seeds: {seeds}")
    print(f"  Max generations: {max_iterations}")
    print()

    total_runs = len(rhos) * seeds
    run_count = 0

    for rho in rhos:
        run_results = []
        for seed in range(seeds):
            run_count += 1
            trial = run_single_trial(
                dm, coords, rho, seed, optimum, max_iterations=max_iterations, num_ants=num_ants)
            curves[rho].append(trial.pop("_curve"))
            all_results.append(trial)
            run_results.append(trial)

            gap_str = "n/a" if trial["gap_percent"] is None else f"{trial['gap_percent']:6.2f}%"
            print(f"[{run_count:3d}/{total_runs}] rho={rho:<4} seed={seed:2d}: "
                  f"cost={trial['best_cost']:9.1f}, gap={gap_str}")

        costs = [r["best_cost"] for r in run_results]
        gaps = [r["gap_percent"] for r in run_results if r["gap_percent"] is not None]
        times = [r["elapsed_seconds"] for r in run_results]
        aggregated[rho] = {
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "min_cost": float(np.min(costs)),
            "max_cost": float(np.max(costs)),
            "mean_gap": float(np.mean(gaps)) if gaps else None,
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "trials": len(run_results),
        }

    print(f"\nExperiment complete. {run_count} trials run.")
    return all_results, aggregated, curves


def _mean_curve(curve_list):
    """Average equal-length cost-over-time curves; pad shorter ones with final value."""
    if not curve_list:
        return np.array([])
    max_len = max(len(c) for c in curve_list)
    padded = []
    for c in curve_list:
        if len(c) < max_len:
            c = list(c) + [c[-1]] * (max_len - len(c))
        padded.append(c)
    return np.array(padded, dtype=float).mean(axis=0)


def save_results(all_results: List[Dict], aggregated: Dict):
    csv_file = OUTPUT_DIR / "mmas_tsp_berlin52_raw.csv"
    json_file = OUTPUT_DIR / "mmas_tsp_berlin52_summary.json"

    fieldnames = [
        "instance", "rho", "num_ants", "seed", "best_cost", "gap_percent",
        "iterations", "fitness_evaluations", "elapsed_seconds",
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved raw results to {csv_file}")

    with open(json_file, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"Saved aggregated results to {json_file}")


def plot_results(curves: Dict, optimum, num_ants: int = NUM_ANTS):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    for rho in sorted(curves.keys()):
        mean_curve = _mean_curve(curves[rho])
        if mean_curve.size == 0:
            continue
        # Each cost-curve step is one generation = num_ants evaluations.
        x = np.arange(len(mean_curve)) * num_ants
        ax.plot(x, mean_curve, label=f"MMAS with ρ = {rho}",
                color=COLORS.get(rho), linewidth=2.0, alpha=0.85)

    if optimum is not None:
        ax.axhline(optimum, linestyle="--", linewidth=2.0, color="#64748b",
                   alpha=0.8, label=f"Optimum = {optimum}")

    ax.set_xlabel("Evaluations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tour Length", fontsize=12, fontweight="bold")
    ax.set_title(f"MMAS on {INSTANCE}", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right")

    plot_file = OUTPUT_DIR / "mmas_tsp_berlin52_convergence.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_file}")


def print_summary(aggregated: Dict, optimum):
    print("\n" + "=" * 80)
    print(f"SUMMARY: MMAS on {INSTANCE} (optimum = {optimum})")
    print("=" * 80)
    print(f"{'rho':>6} | {'Final mean':>11} | {'Std':>9} | {'Min':>9} | {'Mean gap %':>11} | {'Mean time s':>11}")
    print("-" * 80)
    for rho in sorted(aggregated.keys()):
        s = aggregated[rho]
        gap = "n/a" if s["mean_gap"] is None else f"{s['mean_gap']:11.2f}"
        print(f"{rho:>6} | {s['mean_cost']:11.1f} | {s['std_cost']:9.1f} | "
              f"{s['min_cost']:9.1f} | {gap} | {s['mean_time']:11.3f}")


def main():
    parser = argparse.ArgumentParser(description="MMAS berlin52 evaporation-rate sweep")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rhos", nargs="+", type=float, default=None,
                        help=f"Evaporation rates (default: {RHO_VALUES})")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--num-ants", type=int, default=NUM_ANTS)
    args = parser.parse_args()

    rhos = args.rhos if args.rhos else RHO_VALUES
    all_results, aggregated, curves = run_experiment(
        rhos=rhos, seeds=args.seeds, max_iterations=args.max_iterations,
        num_ants=args.num_ants,
    )
    if not all_results:
        return
    optimum = tsp.get_optimum(INSTANCE)
    save_results(all_results, aggregated)
    plot_results(curves, optimum, num_ants=args.num_ants)
    print_summary(aggregated, optimum)


if __name__ == "__main__":
    main()
