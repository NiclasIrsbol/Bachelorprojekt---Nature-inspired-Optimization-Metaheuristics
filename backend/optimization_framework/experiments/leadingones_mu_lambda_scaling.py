"""
Scaling experiment: framework (μ+λ) EA on LeadingOnes across population configs.

These scaling figures use the same (μ+λ) EA that the framework exposes as
``MuPlusLambdaEA``: tournament parent selection, single-point crossover,
standard bit mutation, and plus-selection.

Mirrors ``leadingones_mutation_scaling.py`` (same CLI/output/averaging
conventions), but sweeps three (μ, λ) configurations of the framework (μ+λ) EA
at the standard mutation rate p = 1/n:
  - (1+1)  : μ=1,  λ=1   (smallest GA-style population baseline)
  - (2+10) : μ=2,  λ=10
  - (5+25) : μ=5,  λ=25

Metric:
  y-axis is *fitness evaluations* to the optimum. We do not overlay a
  theoretical runtime curve: this framework variant uses tournament selection
  and crossover, while the standard LeadingOnes curve is for the (1+1) EA.

Configuration:
  - Problem sizes (n): 100, 200, 300, 400, 500
  - Max generations per run: 6000000  (caps GENERATIONS, not evaluations)
  - Seeds (independent runs): 7 per configuration
  - Metric: Fitness evaluations to reach optimum (LeadingOnes = n)

Output (under output/scaling_experiments):
  - CSV : leadingones_mu_lambda_scaling_results.csv
  - PNG : leadingones_mu_lambda_scaling_comparison.png
  - JSON: leadingones_mu_lambda_scaling_data.json

Usage:
    cd backend
    python3 -m optimization_framework.experiments.leadingones_mu_lambda_scaling
    # smoke run (tiny):
    python3 -m optimization_framework.experiments.leadingones_mu_lambda_scaling \
        --sizes 20 40 --seeds 2 --max-iterations 20000
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

from optimization_framework.algorithms.mu_plus_lambda_EA import MuPlusLambdaEA
from optimization_framework.problems.leadingones import fitnessLeadingOnes


REPO_ROOT = Path(__file__).resolve().parents[3]

# Configuration
PROBLEM_SIZES = [100, 200, 300, 400, 500]
MAX_ITERATIONS = 6000000
DEFAULT_SEEDS = 7  # 5 sizes × 3 configs × 7 seeds = 105 total trial runs (~100)
OUTPUT_DIR = REPO_ROOT / "output/scaling_experiments"
DEFAULT_TOURNAMENT_K = 3
DEFAULT_CROSSOVER_TYPE = "single_point"  # fixed by MuPlusLambdaEA in the framework

# (μ, λ) configurations to test (mutation rate fixed at the standard 1/n).
EA_CONFIGS = {
    "(1+1)": (1, 1),
    "(2+10)": (2, 10),
    "(5+25)": (5, 25),
}


def run_single_trial(
    n: int,
    mu_size: int,
    lambda_size: int,
    seed: int,
    max_iterations: int = MAX_ITERATIONS,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    crossover_type: str = DEFAULT_CROSSOVER_TYPE,
) -> Dict:
    """Run a single framework (μ+λ) EA trial on LeadingOnes(n)."""
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()

    best, iterations, _, _, fitness_evals, coords, fitness_history = MuPlusLambdaEA(
        fitness_fn=fitnessLeadingOnes,
        bit_length=n,
        mu_size=mu_size,
        lambda_size=lambda_size,
        tournament_k=tournament_k,
        mutation_prob=1.0 / n,
        max_iterations=max_iterations,
        crossover_type=crossover_type,
    )

    elapsed = time.time() - start_time

    final_fitness = best["fitness"] if isinstance(best, dict) else fitnessLeadingOnes(best)
    reached_optimum = final_fitness == n

    return {
        "n": n,
        "mu": mu_size,
        "lambda": lambda_size,
        "config": None,  # filled by caller
        "tournament_k": tournament_k,
        "crossover_type": crossover_type,
        "seed": seed,
        "fitness_evaluations": fitness_evals,
        "iterations": iterations,
        "final_fitness": final_fitness,
        "reached_optimum": reached_optimum,
        "elapsed_seconds": elapsed,
    }


def run_experiment(
    seeds: int = DEFAULT_SEEDS,
    sizes: List[int] = None,
    max_iterations: int = MAX_ITERATIONS,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    crossover_type: str = DEFAULT_CROSSOVER_TYPE,
):
    """Run full scaling experiment across all problem sizes and (μ,λ) configs."""
    if sizes is None:
        sizes = PROBLEM_SIZES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    aggregated = {name: {} for name in EA_CONFIGS}

    print("Starting framework (μ+λ) EA LeadingOnes scaling experiment")
    print(f"  Problem sizes: {sizes}")
    print(f"  (μ,λ) configs: {list(EA_CONFIGS.keys())}")
    print(f"  Parent selection: tournament k={tournament_k}")
    print(f"  Crossover: {crossover_type}")
    print(f"  Seeds per config: {seeds}")
    print(f"  Max generations: {max_iterations}")
    print()

    total_runs = len(sizes) * len(EA_CONFIGS) * seeds
    run_count = 0

    for n in sizes:
        for config_name, (mu_size, lambda_size) in EA_CONFIGS.items():
            run_results = []
            for seed in range(seeds):
                run_count += 1
                trial = run_single_trial(
                    n, mu_size, lambda_size, seed, max_iterations,
                    tournament_k=tournament_k,
                    crossover_type=crossover_type,
                )
                trial["config"] = config_name
                all_results.append(trial)
                run_results.append(trial)

                status = "✓" if trial["reached_optimum"] else "✗"
                print(
                    f"[{run_count:3d}/{total_runs}] n={n:4d}, cfg={config_name:7s}, "
                    f"seed={seed:2d}: evals={trial['fitness_evaluations']:8d}, "
                    f"gens={trial['iterations']:6d} {status}"
                )

            evals = [r["fitness_evaluations"] for r in run_results if r["reached_optimum"]]
            iters = [r["iterations"] for r in run_results if r["reached_optimum"]]
            success_count = sum(1 for r in run_results if r["reached_optimum"])

            if evals:
                aggregated[config_name][n] = {
                    "mean_evals": np.mean(evals),
                    "std_evals": np.std(evals),
                    "min_evals": np.min(evals),
                    "max_evals": np.max(evals),
                    "mean_iters": np.mean(iters),
                    "std_iters": np.std(iters),
                    "success_rate": success_count / len(run_results),
                    "trials": len(run_results),
                }
            else:
                aggregated[config_name][n] = {
                    "mean_evals": float("inf"),
                    "std_evals": 0,
                    "min_evals": float("inf"),
                    "max_evals": 0,
                    "mean_iters": float("inf"),
                    "std_iters": 0,
                    "success_rate": 0,
                    "trials": len(run_results),
                }

    print(f"\nExperiment complete. {run_count} trials run.")
    return all_results, aggregated


def save_results(all_results: List[Dict], aggregated: Dict):
    """Save results to CSV and JSON."""
    csv_file = OUTPUT_DIR / "leadingones_mu_lambda_scaling_results.csv"
    json_file = OUTPUT_DIR / "leadingones_mu_lambda_scaling_data.json"

    fieldnames = [
        "n", "config", "mu", "lambda", "tournament_k", "crossover_type", "seed",
        "fitness_evaluations", "iterations", "final_fitness",
        "reached_optimum", "elapsed_seconds",
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Saved raw results to {csv_file}")

    with open(json_file, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    print(f"Saved aggregated results to {json_file}")


def plot_results(aggregated: Dict, sizes: List[int] = None):
    if sizes is None:
        sizes = PROBLEM_SIZES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "(1+1)": "#10b981",
        "(2+10)": "#3b82f6",
        "(5+25)": "#ef4444",
    }

    for config_name, color in colors.items():
        if config_name not in aggregated:
            continue

        data = aggregated[config_name]
        plot_sizes, means, stds = [], [], []
        for n in sorted(data.keys(), key=lambda x: int(x)):
            m = data[n]["mean_evals"]
            if np.isfinite(m):
                plot_sizes.append(int(n))
                means.append(m)
                stds.append(data[n]["std_evals"])

        ax.errorbar(
            plot_sizes, means, yerr=stds,
            label=f"(μ+λ)={config_name}",
            marker="o", markersize=8, color=color,
            linewidth=2.5, capsize=5, capthick=1.5, alpha=0.8,
        )

    ax.set_xlabel("Problem Size (n)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fitness Evaluations to Optimum", fontsize=12, fontweight="bold")
    ax.set_title(
        "Framework (μ+λ) EA on LeadingOnes: Population Size Comparison",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    plt.tight_layout()
    plot_file = OUTPUT_DIR / "leadingones_mu_lambda_scaling_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if plot_file.exists():
        print(f"Saved plot to {plot_file}")
    else:
        print("WARNING: plot file was not created")


def print_summary(aggregated: Dict):
    print("\n" + "=" * 80)
    print("SUMMARY: Framework (μ+λ) EA on LeadingOnes - Population Size Comparison")
    print("=" * 80)

    for config_name in EA_CONFIGS:
        if config_name not in aggregated:
            continue

        print(f"\n(μ+λ) = {config_name}:")
        print("-" * 80)
        print(f"{'n':>6} | {'Mean Evals':>12} | {'Std Dev':>12} | {'Success %':>10}")
        print("-" * 80)

        data = aggregated[config_name]
        for n in sorted(data.keys(), key=lambda x: int(x)):
            stats = data[n]
            n_int = int(n)
            print(
                f"{n_int:6d} | {stats['mean_evals']:12.0f} | {stats['std_evals']:12.0f} | "
                f"{stats['success_rate']*100:9.1f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Framework (μ+λ) EA LeadingOnes scaling across (μ,λ) configurations"
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help=f"Number of independent runs per configuration (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=None,
        help=f"Problem sizes to sweep (default: {PROBLEM_SIZES})",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help=f"Max generations per run (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--tournament-k", type=int, default=DEFAULT_TOURNAMENT_K,
        help=f"Tournament size for parent selection (default: {DEFAULT_TOURNAMENT_K})",
    )
    parser.add_argument(
        "--crossover-type", default=DEFAULT_CROSSOVER_TYPE,
        help=f"Crossover type for bit strings (default: {DEFAULT_CROSSOVER_TYPE})",
    )
    args = parser.parse_args()

    sizes = args.sizes if args.sizes else PROBLEM_SIZES
    all_results, aggregated = run_experiment(
        seeds=args.seeds, sizes=sizes, max_iterations=args.max_iterations,
        tournament_k=args.tournament_k,
        crossover_type=args.crossover_type,
    )
    save_results(all_results, aggregated)
    plot_results(aggregated, sizes=sizes)
    print_summary(aggregated)


if __name__ == "__main__":
    main()
