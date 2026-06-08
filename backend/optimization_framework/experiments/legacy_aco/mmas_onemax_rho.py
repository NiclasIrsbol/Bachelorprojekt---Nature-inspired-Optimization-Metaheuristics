"""
Report section 8.4.1 (MMAS, Bit strings -- OneMax): evaporation-rate (rho) sweep.

Standalone per-problem experiment: MMAS (``ant_colony_optimization``) on OneMax
for several evaporation rates rho. Plots average fitness evaluations to the
optimum vs problem size n, one line per rho. Each generation constructs
``num_ants`` solutions, and all constructions count as fitness evaluations.

Output (under output/report_aco):
  - CSV : mmas_onemax_rho_results.csv
  - PNG : mmas_onemax_rho_scaling.png
  - JSON: mmas_onemax_rho_summary.json

Usage:
    cd backend
    python3 -m optimization_framework.experiments.legacy_aco.mmas_onemax_rho
    python3 -m optimization_framework.experiments.legacy_aco.mmas_onemax_rho --sizes 20 40 --seeds 2 --max-iterations 20000
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.algorithms.ant_optimization_problem import ant_colony_optimization
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.experiments.legacy_aco.report_aco_utils import (
    seed_everything, aggregate_evals, write_csv, write_json,
)

PROBLEM = "onemax"
RHO_VALUES = [0.01, 0.05, 0.1, 0.5, 0.9]
NUM_ANTS = 10
PROBLEM_SIZES = [500, 1000, 1500, 2000, 2500]
DEFAULT_SEEDS = 10
MAX_ITERATIONS = 30000  # generations; x num_ants evaluations per generation
OUTPUT_DIR = Path("output/report_aco")

COLORS = {0.01: "#3b82f6", 0.05: "#f59e0b", 0.1: "#10b981", 0.5: "#ef4444", 0.9: "#a855f7"}


def run_single_trial(n: int, rho: float, seed: int, max_iterations: int = MAX_ITERATIONS,
                     num_ants: int = NUM_ANTS) -> Dict:
    seed_everything(seed)
    start = time.time()
    best, iterations, _, _, fitness_evals, _c, _h = ant_colony_optimization(
        fitness_fn=fitnessOnemax, bit_length=n, rho=rho,
        max_iterations=max_iterations, num_ants=num_ants,
    )
    elapsed = time.time() - start
    final_fitness = fitnessOnemax(best) if isinstance(best, str) else best["fitness"]
    return {
        "problem": PROBLEM, "n": n, "rho": rho, "num_ants": num_ants, "seed": seed,
        "fitness_evaluations": fitness_evals, "iterations": iterations,
        "final_fitness": final_fitness, "reached_optimum": final_fitness == n,
        "elapsed_seconds": elapsed,
    }


def run_experiment(seeds: int = DEFAULT_SEEDS, sizes: List[int] = None,
                   rhos: List[float] = None, max_iterations: int = MAX_ITERATIONS,
                   num_ants: int = NUM_ANTS):
    """Returns (all_results, aggregated) with aggregated[rho][n] -> stats."""
    sizes = sizes or PROBLEM_SIZES
    rhos = rhos or RHO_VALUES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    aggregated = {rho: {} for rho in rhos}

    print(f"MMAS rho-sweep on OneMax | rhos={rhos} | sizes={sizes} | seeds={seeds} | num_ants={num_ants}")
    total = len(rhos) * len(sizes) * seeds
    count = 0
    for rho in rhos:
        for n in sizes:
            runs = []
            for seed in range(seeds):
                count += 1
                trial = run_single_trial(n, rho, seed, max_iterations=max_iterations, num_ants=num_ants)
                all_results.append(trial)
                runs.append(trial)
                status = "✓" if trial["reached_optimum"] else "✗"
                print(f"[{count:4d}/{total}] rho={rho:<4} n={n:4d} seed={seed:2d}: "
                      f"evals={trial['fitness_evaluations']:9d} {status}")
            aggregated[rho][n] = aggregate_evals(runs)

    print(f"\nExperiment complete. {count} trials run.")
    return all_results, aggregated


def save_results(all_results, aggregated):
    write_csv(all_results, OUTPUT_DIR / "mmas_onemax_rho_results.csv",
              ["problem", "n", "rho", "num_ants", "seed", "fitness_evaluations", "iterations",
               "final_fitness", "reached_optimum", "elapsed_seconds"])
    write_json(aggregated, OUTPUT_DIR / "mmas_onemax_rho_summary.json")


def plot_results(aggregated):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    for rho in sorted(aggregated.keys()):
        xs, means, stds = [], [], []
        for n in sorted(aggregated[rho].keys()):
            m = aggregated[rho][n]["mean_evals"]
            if np.isfinite(m):
                xs.append(n); means.append(m); stds.append(aggregated[rho][n]["std_evals"])
        if xs:
            ax.errorbar(xs, means, yerr=stds, label=f"MMAS with ρ = {rho}",
                        marker="o", markersize=7, color=COLORS.get(rho),
                        linewidth=2.0, capsize=4, alpha=0.85)
    ax.set_xlabel("Bitstring Length (n)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Evaluations to Optimum", fontsize=12, fontweight="bold")
    ax.set_title("MMAS on OneMax for Different Evaporation Rates", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    out = OUTPUT_DIR / "mmas_onemax_rho_scaling.png"
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved plot to {out}")


def print_summary(aggregated):
    print("\n" + "=" * 70)
    print("SUMMARY: MMAS on OneMax - evaporation rate sweep")
    print("=" * 70)
    for rho in sorted(aggregated.keys()):
        print(f"\n  ρ = {rho}")
        print(f"  {'n':>6} | {'Mean Evals':>12} | {'Std Dev':>12} | {'Success %':>10}")
        print("  " + "-" * 48)
        for n in sorted(aggregated[rho].keys()):
            s = aggregated[rho][n]
            print(f"  {n:6d} | {s['mean_evals']:12.0f} | {s['std_evals']:12.0f} | {s['success_rate']*100:9.1f}%")


def main():
    p = argparse.ArgumentParser(description="MMAS OneMax evaporation-rate sweep")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--rhos", nargs="+", type=float, default=None)
    p.add_argument("--num-ants", type=int, default=NUM_ANTS)
    p.add_argument("--sizes", nargs="+", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = p.parse_args()
    all_results, aggregated = run_experiment(
        args.seeds, args.sizes, args.rhos, args.max_iterations, args.num_ants)
    save_results(all_results, aggregated)
    plot_results(aggregated)
    print_summary(aggregated)


if __name__ == "__main__":
    main()
