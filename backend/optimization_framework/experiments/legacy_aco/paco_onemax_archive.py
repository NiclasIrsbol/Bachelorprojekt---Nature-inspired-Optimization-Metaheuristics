"""
Report section 8.5.1 (P-ACO, Bit strings -- OneMax): archive-size (K) sweep.

Standalone per-problem experiment: P-ACO (``population_based_aco``) on OneMax
for several archive sizes K (num_ants and q0 fixed). Plots average fitness
evaluations to the optimum vs problem size n, one line per K.

This is the P-ACO analogue of the MMAS evaporation-rate study: in MMAS rho
controls pheromone adaptation speed; in P-ACO the archive size K controls how
much historical solution information shapes the pheromone.

Output (under output/report_aco):
  - CSV : paco_onemax_archive_results.csv
  - PNG : paco_onemax_archive_scaling.png
  - JSON: paco_onemax_archive_summary.json

Usage:
    cd backend
    python3 -m optimization_framework.experiments.legacy_aco.paco_onemax_archive
    python3 -m optimization_framework.experiments.legacy_aco.paco_onemax_archive --sizes 20 40 --seeds 2 --max-iterations 20000
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.algorithms.ant_optimization_problem import population_based_aco
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.experiments.legacy_aco.report_aco_utils import (
    seed_everything, aggregate_evals, write_csv, write_json,
)

PROBLEM = "onemax"
ARCHIVE_SIZES = [5, 10, 20, 50]
NUM_ANTS = 10
Q0 = 0.9
PROBLEM_SIZES = [500, 1000, 1500, 2000, 2500]
DEFAULT_SEEDS = 10
MAX_ITERATIONS = 30000  # generations; x num_ants evaluations per generation
OUTPUT_DIR = Path("output/report_aco")

COLORS = {5: "#3b82f6", 10: "#f59e0b", 20: "#10b981", 50: "#ef4444"}


def run_single_trial(n: int, archive_size: int, seed: int,
                     num_ants: int = NUM_ANTS, q0: float = Q0, max_iterations: int = MAX_ITERATIONS) -> Dict:
    seed_everything(seed)
    start = time.time()
    best, iterations, _, _, fitness_evals, _c, _h = population_based_aco(
        fitness_fn=fitnessOnemax, bit_length=n, archive_size=archive_size,
        max_iterations=max_iterations, num_ants=num_ants, q0=q0,
    )
    elapsed = time.time() - start
    final_fitness = fitnessOnemax(best) if isinstance(best, str) else best["fitness"]
    return {
        "problem": PROBLEM, "n": n, "archive_size": archive_size, "num_ants": num_ants, "q0": q0,
        "seed": seed, "fitness_evaluations": fitness_evals, "iterations": iterations,
        "final_fitness": final_fitness, "reached_optimum": final_fitness == n,
        "elapsed_seconds": elapsed,
    }


def run_experiment(seeds: int = DEFAULT_SEEDS, sizes: List[int] = None,
                   archive_sizes: List[int] = None, num_ants: int = NUM_ANTS,
                   q0: float = Q0, max_iterations: int = MAX_ITERATIONS):
    """Returns (all_results, aggregated) with aggregated[K][n] -> stats."""
    sizes = sizes or PROBLEM_SIZES
    archive_sizes = archive_sizes or ARCHIVE_SIZES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    aggregated = {k: {} for k in archive_sizes}

    print(f"P-ACO archive-sweep on OneMax | K={archive_sizes} | sizes={sizes} | seeds={seeds} | num_ants={num_ants}")
    total = len(archive_sizes) * len(sizes) * seeds
    count = 0
    for k in archive_sizes:
        for n in sizes:
            runs = []
            for seed in range(seeds):
                count += 1
                trial = run_single_trial(n, k, seed, num_ants, q0, max_iterations)
                all_results.append(trial)
                runs.append(trial)
                status = "✓" if trial["reached_optimum"] else "✗"
                print(f"[{count:4d}/{total}] K={k:<3} n={n:4d} seed={seed:2d}: "
                      f"evals={trial['fitness_evaluations']:9d} {status}")
            aggregated[k][n] = aggregate_evals(runs)

    print(f"\nExperiment complete. {count} trials run.")
    return all_results, aggregated


def save_results(all_results, aggregated):
    write_csv(all_results, OUTPUT_DIR / "paco_onemax_archive_results.csv",
              ["problem", "n", "archive_size", "num_ants", "q0", "seed",
               "fitness_evaluations", "iterations", "final_fitness", "reached_optimum", "elapsed_seconds"])
    write_json(aggregated, OUTPUT_DIR / "paco_onemax_archive_summary.json")


def plot_results(aggregated):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    for k in sorted(aggregated.keys()):
        xs, means, stds = [], [], []
        for n in sorted(aggregated[k].keys()):
            m = aggregated[k][n]["mean_evals"]
            if np.isfinite(m):
                xs.append(n); means.append(m); stds.append(aggregated[k][n]["std_evals"])
        if xs:
            ax.errorbar(xs, means, yerr=stds, label=f"P-ACO with K = {k}",
                        marker="o", markersize=7, color=COLORS.get(k),
                        linewidth=2.0, capsize=4, alpha=0.85)
    ax.set_xlabel("Bitstring Length (n)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Evaluations to Optimum", fontsize=12, fontweight="bold")
    ax.set_title("P-ACO on OneMax for Different Archive Sizes", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    out = OUTPUT_DIR / "paco_onemax_archive_scaling.png"
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved plot to {out}")


def print_summary(aggregated):
    print("\n" + "=" * 70)
    print("SUMMARY: P-ACO on OneMax - archive-size sweep")
    print("=" * 70)
    for k in sorted(aggregated.keys()):
        print(f"\n  K = {k}")
        print(f"  {'n':>6} | {'Mean Evals':>12} | {'Std Dev':>12} | {'Success %':>10}")
        print("  " + "-" * 48)
        for n in sorted(aggregated[k].keys()):
            s = aggregated[k][n]
            print(f"  {n:6d} | {s['mean_evals']:12.0f} | {s['std_evals']:12.0f} | {s['success_rate']*100:9.1f}%")


def main():
    p = argparse.ArgumentParser(description="P-ACO OneMax archive-size sweep")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--archive-sizes", nargs="+", type=int, default=None)
    p.add_argument("--num-ants", type=int, default=NUM_ANTS)
    p.add_argument("--q0", type=float, default=Q0)
    p.add_argument("--sizes", nargs="+", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = p.parse_args()
    all_results, aggregated = run_experiment(
        args.seeds, args.sizes, args.archive_sizes, args.num_ants, args.q0, args.max_iterations)
    save_results(all_results, aggregated)
    plot_results(aggregated)
    print_summary(aggregated)


if __name__ == "__main__":
    main()
