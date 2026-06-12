# @author: Andrej Kitanovski
"""
Scaling experiment: framework (mu+lambda) EA on OneMax across population configs.

This script uses the same GA-style ``MuPlusLambdaEA`` implementation that the
framework exposes in the application: tournament parent selection, single-point
crossover by default, standard bit mutation with p = 1/n, and plus-selection.

The main metric is fitness evaluations to the optimum, not generations. This is
important because larger lambda values evaluate more offspring per generation.

Output (under output/scaling_experiments):
  - CSV : onemax_mu_lambda_scaling_results.csv
  - PNG : onemax_mu_lambda_scaling_comparison.png
  - JSON: onemax_mu_lambda_scaling_data.json

Usage:
    cd backend
    python3 -m optimization_framework.experiments.onemax_mu_lambda_scaling
    # smoke run:
    python3 -m optimization_framework.experiments.onemax_mu_lambda_scaling \
        --sizes 20 40 --seeds 2 --max-iterations 5000
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
from optimization_framework.problems.onemax import fitnessOnemax


REPO_ROOT = Path(__file__).resolve().parents[3]

PROBLEM_SIZES = [500, 1000, 1500, 2000, 2500]
MAX_ITERATIONS = 1000000
DEFAULT_SEEDS = 7
OUTPUT_DIR = REPO_ROOT / "output/scaling_experiments"
DEFAULT_TOURNAMENT_K = 3
DEFAULT_CROSSOVER_TYPE = "single_point"

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
    """Run one framework (mu+lambda) EA trial on OneMax(n)."""
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    best, iterations, _, _, fitness_evals, _coords, _history = MuPlusLambdaEA(
        fitness_fn=fitnessOnemax,
        bit_length=n,
        mu_size=mu_size,
        lambda_size=lambda_size,
        tournament_k=tournament_k,
        mutation_prob=1.0 / n,
        max_iterations=max_iterations,
        crossover_type=crossover_type,
    )
    elapsed = time.time() - start_time

    final_fitness = best["fitness"] if isinstance(best, dict) else fitnessOnemax(best)
    reached_optimum = final_fitness == n

    return {
        "n": n,
        "config": None,
        "mu": mu_size,
        "lambda": lambda_size,
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
    """Run the full OneMax scaling experiment."""
    if sizes is None:
        sizes = PROBLEM_SIZES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    aggregated = {name: {} for name in EA_CONFIGS}

    print("Starting framework (mu+lambda) EA OneMax scaling experiment")
    print(f"  Problem sizes: {sizes}")
    print(f"  (mu,lambda) configs: {list(EA_CONFIGS.keys())}")
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
                    n=n,
                    mu_size=mu_size,
                    lambda_size=lambda_size,
                    seed=seed,
                    max_iterations=max_iterations,
                    tournament_k=tournament_k,
                    crossover_type=crossover_type,
                )
                trial["config"] = config_name
                all_results.append(trial)
                run_results.append(trial)

                status = "ok" if trial["reached_optimum"] else "dnf"
                print(
                    f"[{run_count:3d}/{total_runs}] n={n:4d}, cfg={config_name:7s}, "
                    f"seed={seed:2d}: evals={trial['fitness_evaluations']:8d}, "
                    f"gens={trial['iterations']:6d} {status}"
                )

            evals = [r["fitness_evaluations"] for r in run_results if r["reached_optimum"]]
            iters = [r["iterations"] for r in run_results if r["reached_optimum"]]
            success_count = sum(1 for r in run_results if r["reached_optimum"])

            aggregated[config_name][n] = {
                "mean_evals": float(np.mean(evals)) if evals else float("inf"),
                "std_evals": float(np.std(evals)) if evals else 0.0,
                "min_evals": float(np.min(evals)) if evals else float("inf"),
                "max_evals": float(np.max(evals)) if evals else 0.0,
                "mean_iters": float(np.mean(iters)) if iters else float("inf"),
                "std_iters": float(np.std(iters)) if iters else 0.0,
                "success_rate": success_count / len(run_results),
                "trials": len(run_results),
            }

    print(f"\nExperiment complete. {run_count} trials run.")
    return all_results, aggregated


def save_results(all_results: List[Dict], aggregated: Dict):
    """Save raw and aggregated results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = OUTPUT_DIR / "onemax_mu_lambda_scaling_results.csv"
    json_file = OUTPUT_DIR / "onemax_mu_lambda_scaling_data.json"

    fieldnames = [
        "n",
        "config",
        "mu",
        "lambda",
        "tournament_k",
        "crossover_type",
        "seed",
        "fitness_evaluations",
        "iterations",
        "final_fitness",
        "reached_optimum",
        "elapsed_seconds",
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
    """Plot mean fitness evaluations to optimum by problem size."""
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
        data = aggregated.get(config_name, {})
        plot_sizes, means, stds = [], [], []
        for n in sorted(data.keys(), key=lambda x: int(x)):
            mean = data[n]["mean_evals"]
            if np.isfinite(mean):
                plot_sizes.append(int(n))
                means.append(mean)
                stds.append(data[n]["std_evals"])

        ax.errorbar(
            plot_sizes,
            means,
            yerr=stds,
            label=f"(mu+lambda)={config_name}",
            marker="o",
            markersize=8,
            color=color,
            linewidth=2.5,
            capsize=5,
            capthick=1.5,
            alpha=0.85,
        )

    ax.set_xlabel("Problem Size (n)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fitness Evaluations to Optimum", fontsize=12, fontweight="bold")
    ax.set_title(
        "Framework (mu+lambda) EA on OneMax: Population Size Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    plt.tight_layout()
    plot_file = OUTPUT_DIR / "onemax_mu_lambda_scaling_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_file}")


def print_summary(aggregated: Dict):
    print("\n" + "=" * 80)
    print("SUMMARY: Framework (mu+lambda) EA on OneMax - Population Size Comparison")
    print("=" * 80)

    for config_name in EA_CONFIGS:
        data = aggregated.get(config_name, {})
        print(f"\n(mu+lambda) = {config_name}:")
        print("-" * 80)
        print(f"{'n':>6} | {'Mean Evals':>12} | {'Std Dev':>12} | {'Success %':>10}")
        print("-" * 80)
        for n in sorted(data.keys(), key=lambda x: int(x)):
            stats = data[n]
            mean = stats["mean_evals"]
            mean_text = f"{mean:12.0f}" if np.isfinite(mean) else f"{'DNF':>12}"
            print(
                f"{int(n):6d} | {mean_text} | {stats['std_evals']:12.0f} | "
                f"{stats['success_rate']*100:9.1f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Framework (mu+lambda) EA OneMax scaling across population configurations"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--tournament-k", type=int, default=DEFAULT_TOURNAMENT_K)
    parser.add_argument("--crossover-type", default=DEFAULT_CROSSOVER_TYPE)
    args = parser.parse_args()

    sizes = args.sizes if args.sizes else PROBLEM_SIZES
    all_results, aggregated = run_experiment(
        seeds=args.seeds,
        sizes=sizes,
        max_iterations=args.max_iterations,
        tournament_k=args.tournament_k,
        crossover_type=args.crossover_type,
    )
    save_results(all_results, aggregated)
    plot_results(aggregated, sizes=sizes)
    print_summary(aggregated)


if __name__ == "__main__":
    main()
