# @author: Niclas Søe Irsbøl
"""
Scaling experiment: (1+1) EA on OneMax with varying mutation probabilities.

Tests how (1+1) EA performance scales with problem size for three mutation rates:
  - p = 2/n (aggressive mutation)
  - p = 1/n (standard mutation)
  - p = 0.89/n (conservative mutation)

Compares empirical convergence against theoretical runtime: e*n*ln(n) - 1.89*n

Configuration:
  - Problem sizes (n): 500, 1000, 1500, 2000, 2500
  - Max iterations per run: 60000
  - Seeds (independent runs): 7 per configuration
  - Total trial runs: 105 (5 problem sizes × 3 mutation rates × 7 seeds)
  - Metric: Fitness evaluations to reach optimum (OneMax = n)

Output:
  - CSV: onemax_mutation_scaling_results.csv
  - PNG: onemax_mutation_scaling_comparison.png (convergence curves + theory)
  - JSON: onemax_mutation_scaling_data.json (detailed results)
"""

import argparse
import csv
import json
import random
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.algorithms.one_plus_one_EA import OnePlusOneEA
from optimization_framework.problems.onemax import fitnessOnemax


# Configuration
PROBLEM_SIZES = [500, 1000, 1500, 2000, 2500]
MAX_ITERATIONS = 60000
DEFAULT_SEEDS = 7  # 5 problem sizes × 3 mutation rates × 7 seeds = 105 total trial runs (~100)
OUTPUT_DIR = Path("output/scaling_experiments")

# Mutation rates to test (as functions of n)
MUTATION_RATES = {
    "2/n": lambda n: 2.0 / n,
    "1/n": lambda n: 1.0 / n,
    "0.5/n": lambda n: 0.5 / n,
}


def theoretical_runtime(n: float) -> float:
    """
    Theoretical expected runtime for (1+1) EA on OneMax.
    Formula: e*n*ln(n) - 1.89*n (from literature)
    """
    if n <= 0:
        return 0
    return math.e * n * math.log(n) - 1.89 * n


def run_single_trial(
    n: int,
    mutation_rate: float,
    seed: int,
    max_iterations: int = MAX_ITERATIONS
) -> Dict:
    """
    Run a single (1+1) EA trial on OneMax(n).
    
    Returns:
        Dict with keys: n, mutation_rate, seed, evals_to_optimum, iterations, success
    """
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    
    best_bitstring, iterations, _, _, fitness_evals, coords, fitness_history = OnePlusOneEA(
        fitness_fn=fitnessOnemax,
        bit_length=n,
        prob=mutation_rate,
        max_iterations=max_iterations,
    )
    
    elapsed = time.time() - start_time
    
    final_fitness = fitnessOnemax(best_bitstring)
    reached_optimum = final_fitness == n
    
    return {
        "n": n,
        "mutation_rate_desc": f"{mutation_rate:.6f}",
        "mutation_formula": None,  # Will be filled by caller
        "seed": seed,
        "fitness_evaluations": fitness_evals,
        "iterations": iterations,
        "final_fitness": final_fitness,
        "reached_optimum": reached_optimum,
        "elapsed_seconds": elapsed,
    }


def run_experiment(seeds: int = DEFAULT_SEEDS):
    """
    Run full scaling experiment across all problem sizes and mutation rates.
    
    Args:
        seeds: Number of independent runs per configuration
    
    Returns:
        Tuple of (results_list, aggregated_data)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    aggregated = {rate_name: {} for rate_name in MUTATION_RATES}
    
    print(f"Starting (1+1) EA OneMax scaling experiment")
    print(f"  Problem sizes: {PROBLEM_SIZES}")
    print(f"  Mutation rates: {list(MUTATION_RATES.keys())}")
    print(f"  Seeds per config: {seeds}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print()
    
    total_runs = len(PROBLEM_SIZES) * len(MUTATION_RATES) * seeds
    run_count = 0
    
    for n in PROBLEM_SIZES:
        for rate_name, rate_fn in MUTATION_RATES.items():
            mut_rate = rate_fn(n)
            
            # Run multiple seeds
            run_results = []
            for seed in range(seeds):
                run_count += 1
                trial = run_single_trial(n, mut_rate, seed, MAX_ITERATIONS)
                trial["mutation_formula"] = rate_name
                all_results.append(trial)
                run_results.append(trial)
                
                status = "✓" if trial["reached_optimum"] else "✗"
                print(
                    f"[{run_count:3d}/{total_runs}] n={n:4d}, μ={rate_name:6s}, "
                    f"seed={seed:2d}: evals={trial['fitness_evaluations']:8d}, "
                    f"iters={trial['iterations']:6d} {status}"
                )
            
            # Aggregate statistics
            evals = [r["fitness_evaluations"] for r in run_results if r["reached_optimum"]]
            iters = [r["iterations"] for r in run_results if r["reached_optimum"]]
            success_count = sum(1 for r in run_results if r["reached_optimum"])
            
            if evals:
                aggregated[rate_name][n] = {
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
                aggregated[rate_name][n] = {
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
    csv_file = OUTPUT_DIR / "onemax_mutation_scaling_results.csv"
    json_file = OUTPUT_DIR / "onemax_mutation_scaling_data.json"
    
    # Write detailed CSV
    fieldnames = [
        "n", "mutation_formula", "mutation_rate_desc", "seed",
        "fitness_evaluations", "iterations", "final_fitness",
        "reached_optimum", "elapsed_seconds"
    ]
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Saved raw results to {csv_file}")
    
    # Write aggregated JSON
    with open(json_file, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    print(f"Saved aggregated results to {json_file}")


def plot_results(aggregated: Dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "2/n": "#ef4444",
        "1/n": "#3b82f6",
        "0.5/n": "#10b981",
    }

    for rate_name, color in colors.items():
        if rate_name not in aggregated:
            continue

        data = aggregated[rate_name]

        sizes = []
        means = []
        stds = []

        for n in sorted(data.keys()):
            m = data[n]["mean_evals"]

            if np.isfinite(m):
                sizes.append(n)
                means.append(m)
                stds.append(data[n]["std_evals"])

        ax.errorbar(
            sizes,
            means,
            yerr=stds,
            label=f"Empirical: p={rate_name}",
            marker="o",
            markersize=8,
            color=color,
            linewidth=2.5,
            capsize=5,
            capthick=1.5,
            alpha=0.8,
        )

    theory_sizes = np.linspace(
        min(PROBLEM_SIZES) * 0.9,
        max(PROBLEM_SIZES) * 1.1,
        200,
    )

    theory_evals = [theoretical_runtime(n) for n in theory_sizes]

    ax.plot(
        theory_sizes,
        theory_evals,
        label=r"Theory: $e\,n\ln(n)-1.89n$",
        linestyle="--",
        linewidth=2.5,
        color="#64748b",
        alpha=0.7,
    )

    ax.set_xlabel("Problem Size (n)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fitness Evaluations to Optimum", fontsize=12, fontweight="bold")
    ax.set_title(
        "(1+1) EA on OneMax: Mutation Rate Comparison vs. Theoretical Runtime",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    plt.tight_layout()

    plot_file = OUTPUT_DIR / "onemax_mutation_scaling_comparison.png"

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if plot_file.exists():
        print(f"Saved plot to {plot_file}")
    else:
        print("WARNING: plot file was not created")

def print_summary(aggregated: Dict):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY: (1+1) EA on OneMax - Mutation Rate Comparison")
    print("="*80)
    
    for rate_name in ["2/n", "1/n", "0.5/n"]:
        if rate_name not in aggregated:
            continue
        
        print(f"\n{rate_name} Mutation Rate:")
        print("-" * 80)
        print(f"{'n':>6} | {'Mean Evals':>12} | {'Std Dev':>12} | {'Success %':>10} | Theory")
        print("-" * 80)
        
        data = aggregated[rate_name]
        for n in sorted(data.keys()):
            stats = data[n]
            theory = theoretical_runtime(n)
            theory_str = f"{theory:.0f}"
            
            print(
                f"{n:6d} | {stats['mean_evals']:12.0f} | {stats['std_evals']:12.0f} | "
                f"{stats['success_rate']*100:9.1f}% | {theory_str:>10}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="(1+1) EA OneMax scaling with mutation rate comparison"
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help=f"Number of independent runs per configuration (default: {DEFAULT_SEEDS})"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    all_results, aggregated = run_experiment(seeds=args.seeds)
    
    # Save results
    save_results(all_results, aggregated)
    
    # Plot
    plot_results(aggregated)
    
    # Print summary
    print_summary(aggregated)


if __name__ == "__main__":
    main()
