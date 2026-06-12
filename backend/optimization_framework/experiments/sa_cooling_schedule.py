# @author: Niclas Søe Irsbøl
"""
Simulated Annealing cooling schedule comparison.

Tests three cooling schedules (Fast, Medium, Slow) across three problems:
  - OneMax (bitstring, 100 bits)
  - LeadingOnes (bitstring, 100 bits)
  - TSP (berlin52)

Cooling schedules:
  - Fast:   T(t) = T0 * 0.90^t
  - Medium: T(t) = T0 * 0.99^t
  - Slow:   T(t) = T0 * 0.999^t

Configuration:
  - T0 = 1000 for all schedules
  - Max iterations per run: 10000 (bitstrings) or 100000 (TSP)
  - Seeds (independent runs): 15 per configuration
  - Metrics: Final fitness/cost, convergence curve, runtime

Output:
  - PNG plots: convergence curves per problem
  - JSON data: all trial results
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.algorithms.simulated_annealing import (
    simulated_annealing,
    simulated_annealingTSP,
)
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.problems.leadingones import fitnessLeadingOnes
from optimization_framework.problems.tsp import (
    fetch_tsp_instance,
    tour_cost,
    get_optimum,
)

# ============================================================
# CONFIG
# ============================================================

BITSTRING_LENGTH = 100
BITSTRING_MAX_ITERATIONS = 10000
TSP_MAX_ITERATIONS = 100000
DEFAULT_SEEDS = 15
OUTPUT_DIR = Path("output/sa_cooling_schedule")

# Cooling schedules: (name, alpha, T0)
COOLING_SCHEDULES = [
    ("Fast", 0.90, 1000),
    ("Medium", 0.99, 1000),
    ("Slow", 0.999, 1000),
]

# Problems: (name, type, config_dict)
PROBLEMS = [
    ("OneMax", "bitstring", {
        "fitness_fn": fitnessOnemax,
        "bit_length": BITSTRING_LENGTH,
        "max_iterations": BITSTRING_MAX_ITERATIONS,
    }),
    ("LeadingOnes", "bitstring", {
        "fitness_fn": fitnessLeadingOnes,
        "bit_length": BITSTRING_LENGTH,
        "max_iterations": BITSTRING_MAX_ITERATIONS,
    }),
    ("berlin52", "tsp", {
        "instance_name": "berlin52",
        "max_iterations": TSP_MAX_ITERATIONS,
    }),
]


# ============================================================
# BITSTRING EXPERIMENTS
# ============================================================

def run_sa_bitstring(fitness_fn, bit_length, cooling, T0, max_iterations, seed):
    """Run SA on a bitstring problem."""
    random.seed(seed)
    np.random.seed(seed)

    start = time.time()

    best, iterations, final_temp, _, fitness_evals, coords, fitness_history = simulated_annealing(
        fitness_fn=fitness_fn,
        bit_length=bit_length,
        cooling=cooling,
        T0=T0,
        max_iterations=max_iterations,
    )

    elapsed = time.time() - start
    final_fitness = fitness_fn(best)

    return {
        "final_fitness": final_fitness,
        "fitness_evaluations": fitness_evals,
        "iterations": iterations,
        "final_temperature": final_temp,
        "elapsed_seconds": elapsed,
        "fitness_history": fitness_history,
    }


# ============================================================
# TSP EXPERIMENTS
# ============================================================

def run_sa_tsp(instance_name, cooling, T0, max_iterations, seed):
    """Run SA on TSP."""
    random.seed(seed)
    np.random.seed(seed)

    name, problem, coords, nodes, distance_matrix = fetch_tsp_instance(instance_name)

    start = time.time()

    best_tour, iterations, final_temp, _, fitness_evals, tour_coords, cost_history = simulated_annealingTSP(
        distance_matrix=distance_matrix,
        city_coords=coords,
        cooling=cooling,
        T0=T0,
        max_iterations=max_iterations,
    )

    elapsed = time.time() - start
    final_cost = tour_cost(best_tour, distance_matrix)

    return {
        "final_cost": final_cost,
        "fitness_evaluations": fitness_evals,
        "iterations": iterations,
        "final_temperature": final_temp,
        "elapsed_seconds": elapsed,
        "cost_history": cost_history,
    }


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_by_evaluations(runs, max_points=200):
    """
    Average convergence curves on a shared fitness-evaluation x-axis.
    
    For bitstrings: each iteration = 1 evaluation
    For TSP: each iteration = 1 evaluation (one 2-opt per iteration)
    """
    series = []
    for run in runs:
        if "fitness_history" in run:
            history = run["fitness_history"]
        elif "cost_history" in run:
            history = run["cost_history"]
        else:
            continue

        if not history:
            continue

        values = np.asarray(history, dtype=float)
        evals = np.arange(len(values), dtype=float)
        series.append((evals, values))

    if not series:
        return [], [], []

    max_eval = max(float(evals[-1]) for evals, _ in series)

    # Create shared evaluation grid
    eval_grid = np.linspace(0, max_eval, min(max_points, int(max_eval) + 1))

    # Interpolate each run onto shared grid
    interpolated = []
    for evals, values in series:
        interp_values = np.interp(eval_grid, evals, values)
        interpolated.append(interp_values)

    # Compute mean and std
    interpolated = np.array(interpolated)
    mean_vals = np.mean(interpolated, axis=0)
    std_vals = np.std(interpolated, axis=0)

    return eval_grid, mean_vals, std_vals


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(num_seeds=DEFAULT_SEEDS):
    """Run full SA cooling schedule comparison."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_experiment_results = {}

    for problem_name, problem_type, config in PROBLEMS:

        print(f"\n{'='*60}")
        print(f"Problem: {problem_name} ({problem_type})")
        print(f"{'='*60}")

        problem_results = {}

        for schedule_name, alpha, T0 in COOLING_SCHEDULES:

            print(f"\nSchedule: {schedule_name} (α={alpha}, T0={T0})")

            runs = []

            for seed in range(num_seeds):

                print(f"  Seed {seed+1}/{num_seeds}...", end=" ", flush=True)

                try:
                    if problem_type == "bitstring":
                        result = run_sa_bitstring(
                            fitness_fn=config["fitness_fn"],
                            bit_length=config["bit_length"],
                            cooling=alpha,
                            T0=T0,
                            max_iterations=config["max_iterations"],
                            seed=seed,
                        )
                        metric_key = "final_fitness"
                        metric_val = result["final_fitness"]

                    else:  # TSP
                        result = run_sa_tsp(
                            instance_name=config["instance_name"],
                            cooling=alpha,
                            T0=T0,
                            max_iterations=config["max_iterations"],
                            seed=seed,
                        )
                        metric_key = "final_cost"
                        metric_val = result["final_cost"]

                    runs.append(result)
                    print(f"{metric_key}={metric_val:.2f}, time={result['elapsed_seconds']:.3f}s")

                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            if not runs:
                print(f"  All seeds failed for {schedule_name}")
                continue

            # Summary
            if problem_type == "bitstring":
                metrics = [r["final_fitness"] for r in runs]
                print(f"\n  Summary (Final Fitness):")
            else:
                metrics = [r["final_cost"] for r in runs]
                print(f"\n  Summary (Final Cost):")

            times = [r["elapsed_seconds"] for r in runs]

            print(f"    Mean: {np.mean(metrics):.2f} ± {np.std(metrics):.2f}")
            print(f"    Min/Max: {np.min(metrics):.2f} / {np.max(metrics):.2f}")
            print(f"    Time (mean): {np.mean(times):.4f}s")

            problem_results[schedule_name] = runs

        all_experiment_results[problem_name] = problem_results

        # Create plot for this problem
        print(f"\nGenerating convergence plot for {problem_name}...")
        create_convergence_plot(problem_name, problem_results, problem_type)

    # Save all results
    output_file = OUTPUT_DIR / "sa_cooling_schedule_data.json"
    with open(output_file, "w") as f:
        json.dump(all_experiment_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")


def create_convergence_plot(problem_name, problem_results, problem_type):
    """Create a convergence curve plot for a single problem."""

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"Fast": "#d62728", "Medium": "#ff7f0e", "Slow": "#1f77b4"}

    for schedule_name, runs in problem_results.items():

        eval_grid, mean_vals, std_vals = aggregate_by_evaluations(runs)

        if len(eval_grid) == 0:
            print(f"  Skipping {schedule_name} - no valid data")
            continue

        # For TSP, compute gap from optimum if available
        if problem_type == "tsp":
            optimum = get_optimum(problem_name)
            if optimum:
                gap = (np.mean(mean_vals) - optimum) / optimum * 100
                label = f"{schedule_name} (α={COOLING_SCHEDULES[[s[0] for s in COOLING_SCHEDULES].index(schedule_name)][1]}, avg={np.mean(mean_vals):.0f}, gap={gap:.1f}%)"
            else:
                label = f"{schedule_name} (α={COOLING_SCHEDULES[[s[0] for s in COOLING_SCHEDULES].index(schedule_name)][1]}, avg={np.mean(mean_vals):.0f})"
        else:
            label = f"{schedule_name} (α={COOLING_SCHEDULES[[s[0] for s in COOLING_SCHEDULES].index(schedule_name)][1]}, avg={np.mean(mean_vals):.2f})"

        ax.plot(eval_grid, mean_vals, label=label, color=colors[schedule_name], linewidth=2.5)
        ax.fill_between(eval_grid, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=colors[schedule_name])

    # Add optimum line for TSP
    if problem_type == "tsp":
        optimum = get_optimum(problem_name)
        if optimum:
            ax.axhline(y=optimum, color="green", linestyle="--", linewidth=2, label=f"Optimum: {optimum}", alpha=0.7)

    ax.set_xlabel("Fitness Evaluations", fontsize=12)
    if problem_type == "tsp":
        ax.set_ylabel("Tour Cost", fontsize=12)
    else:
        ax.set_ylabel("Fitness", fontsize=12)

    ax.set_title(f"Simulated Annealing - Cooling Schedule Comparison on {problem_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plot_file = OUTPUT_DIR / f"{problem_name}_cooling_schedule.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {plot_file}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SA cooling schedule comparison experiment")
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Number of independent runs per schedule (default: {DEFAULT_SEEDS})",
    )

    args = parser.parse_args()

    run_experiment(num_seeds=args.seeds)