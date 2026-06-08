"""
Report section 8.5.2 (P-ACO, TSP): archive-size (K) sweep on berlin52.

Runs P-ACO (``population_based_acoTSP``) on the berlin52 instance for several
archive sizes K and plots the convergence of tour length vs evaluations, one
line per K, with a dashed line at the known optimum (7542). num_ants and q0 are
held fixed. This is the P-ACO analogue of the MMAS evaporation-rate TSP study.

The x-axis is fitness EVALUATIONS (not iterations): P-ACO performs num_ants
constructions per iteration, so each cost-curve step corresponds to num_ants
evaluations.

Output (under output/report_aco):
  - CSV : paco_tsp_berlin52_raw.csv
  - PNG : paco_tsp_berlin52_convergence.png
  - JSON: paco_tsp_berlin52_summary.json

Requires the TSPLIB clone (problems/tsplib). If berlin52 cannot be loaded the
script prints a clear message and exits without error.

Usage:
    cd backend
    python3 -m optimization_framework.experiments.legacy_aco.paco_tsp_archive_berlin52
    # smoke run (tiny):
    python3 -m optimization_framework.experiments.legacy_aco.paco_tsp_archive_berlin52 \
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

from optimization_framework.algorithms.ant_optimization_problem import population_based_acoTSP
from optimization_framework.problems import tsp


# Configuration
INSTANCE = "berlin52"
ARCHIVE_SIZES = [5, 10, 20, 50]
NUM_ANTS = 10
Q0 = 0.9
DEFAULT_SEEDS = 10
MAX_ITERATIONS = 2000  # 2000 iterations x num_ants ≈ 20000 evaluations
OUTPUT_DIR = Path("output/report_aco")

COLORS = {
    5: "#3b82f6",
    10: "#f59e0b",
    20: "#10b981",
    50: "#ef4444",
}


def run_single_trial(distance_matrix, city_coords, archive_size, seed,
                     num_ants=NUM_ANTS, q0=Q0, max_iterations=MAX_ITERATIONS):
    """Run a single P-ACO trial on the TSP instance with the given archive size."""
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    best_tour, iterations, _, _, fitness_evals, _coords, cost_over_time = population_based_acoTSP(
        distance_matrix, city_coords, archive_size=archive_size,
        max_iterations=max_iterations, num_ants=num_ants, q0=q0,
    )
    elapsed = time.time() - start_time

    best_cost = tsp.tour_cost(best_tour, distance_matrix)
    gap = tsp.gap_percent(best_cost, INSTANCE)

    return {
        "instance": INSTANCE,
        "archive_size": archive_size,
        "num_ants": num_ants,
        "q0": q0,
        "seed": seed,
        "best_cost": best_cost,
        "gap_percent": gap,
        "iterations": iterations,
        "fitness_evaluations": fitness_evals,
        "elapsed_seconds": elapsed,
        "_curve": cost_over_time,
    }


def run_experiment(archive_sizes: List[int] = None, seeds: int = DEFAULT_SEEDS,
                   num_ants: int = NUM_ANTS, q0: float = Q0, max_iterations: int = MAX_ITERATIONS):
    """Run the archive-size sweep on berlin52.

    Returns (all_results, aggregated, curves). ``curves[K]`` is a list of
    cost-over-time arrays (one per seed). Returns ([], {}, {}) if the instance
    cannot be loaded.
    """
    if archive_sizes is None:
        archive_sizes = ARCHIVE_SIZES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        name, _problem, coords, _nodes, dm = tsp.fetch_tsp_instance(INSTANCE)
    except Exception as e:  # noqa: BLE001
        print(f"Could not load TSP instance '{INSTANCE}' ({e}). Is the tsplib clone present?")
        return [], {}, {}

    optimum = tsp.get_optimum(INSTANCE)

    all_results = []
    aggregated = {k: {} for k in archive_sizes}
    curves = {k: [] for k in archive_sizes}

    print(f"Starting P-ACO archive-size sweep on {name} (n={len(dm)}, optimum={optimum})")
    print(f"  Archive sizes: {archive_sizes}")
    print(f"  num_ants={num_ants}, q0={q0}")
    print(f"  Seeds: {seeds}")
    print(f"  Max iterations: {max_iterations}")
    print()

    total_runs = len(archive_sizes) * seeds
    run_count = 0

    for k in archive_sizes:
        run_results = []
        for seed in range(seeds):
            run_count += 1
            trial = run_single_trial(dm, coords, k, seed, num_ants, q0, max_iterations)
            curves[k].append(trial.pop("_curve"))
            all_results.append(trial)
            run_results.append(trial)

            gap_str = "n/a" if trial["gap_percent"] is None else f"{trial['gap_percent']:6.2f}%"
            print(f"[{run_count:3d}/{total_runs}] K={k:<3} seed={seed:2d}: "
                  f"cost={trial['best_cost']:9.1f}, gap={gap_str}")

        costs = [r["best_cost"] for r in run_results]
        gaps = [r["gap_percent"] for r in run_results if r["gap_percent"] is not None]
        times = [r["elapsed_seconds"] for r in run_results]
        aggregated[k] = {
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
    csv_file = OUTPUT_DIR / "paco_tsp_berlin52_raw.csv"
    json_file = OUTPUT_DIR / "paco_tsp_berlin52_summary.json"

    fieldnames = [
        "instance", "archive_size", "num_ants", "q0", "seed",
        "best_cost", "gap_percent", "iterations", "fitness_evaluations", "elapsed_seconds",
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

    for k in sorted(curves.keys()):
        mean_curve = _mean_curve(curves[k])
        if mean_curve.size == 0:
            continue
        # Each cost-curve step is one iteration = num_ants evaluations.
        x = np.arange(len(mean_curve)) * num_ants
        ax.plot(x, mean_curve, label=f"P-ACO with K = {k}",
                color=COLORS.get(k), linewidth=2.0, alpha=0.85)

    if optimum is not None:
        ax.axhline(optimum, linestyle="--", linewidth=2.0, color="#64748b",
                   alpha=0.8, label=f"Optimum = {optimum}")

    ax.set_xlabel("Evaluations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tour Length", fontsize=12, fontweight="bold")
    ax.set_title(f"P-ACO on {INSTANCE}", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right")

    plot_file = OUTPUT_DIR / "paco_tsp_berlin52_convergence.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_file}")


def print_summary(aggregated: Dict, optimum):
    print("\n" + "=" * 80)
    print(f"SUMMARY: P-ACO on {INSTANCE} (optimum = {optimum})")
    print("=" * 80)
    print(f"{'K':>6} | {'Final mean':>11} | {'Std':>9} | {'Min':>9} | {'Mean gap %':>11} | {'Mean time s':>11}")
    print("-" * 80)
    for k in sorted(aggregated.keys()):
        s = aggregated[k]
        gap = "n/a" if s["mean_gap"] is None else f"{s['mean_gap']:11.2f}"
        print(f"{k:>6} | {s['mean_cost']:11.1f} | {s['std_cost']:9.1f} | "
              f"{s['min_cost']:9.1f} | {gap} | {s['mean_time']:11.3f}")


def main():
    parser = argparse.ArgumentParser(description="P-ACO berlin52 archive-size sweep")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--archive-sizes", nargs="+", type=int, default=None,
                        help=f"Archive sizes (default: {ARCHIVE_SIZES})")
    parser.add_argument("--num-ants", type=int, default=NUM_ANTS)
    parser.add_argument("--q0", type=float, default=Q0)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()

    archive_sizes = args.archive_sizes if args.archive_sizes else ARCHIVE_SIZES
    all_results, aggregated, curves = run_experiment(
        archive_sizes=archive_sizes, seeds=args.seeds,
        num_ants=args.num_ants, q0=args.q0, max_iterations=args.max_iterations,
    )
    if not all_results:
        return
    optimum = tsp.get_optimum(INSTANCE)
    save_results(all_results, aggregated)
    plot_results(curves, optimum, num_ants=args.num_ants)
    print_summary(aggregated, optimum)


if __name__ == "__main__":
    main()
