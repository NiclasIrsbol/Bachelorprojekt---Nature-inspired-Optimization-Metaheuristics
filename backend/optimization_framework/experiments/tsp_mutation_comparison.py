"""
TSP mutation comparison: (1+1) EA on berlin52 with 2-opt vs 3-opt operators.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.operators import gaoperators
from optimization_framework.problems.tsp import (
    fetch_tsp_instance,
    tour_cost,
    get_optimum,
)

# ============================================================
# CONFIG
# ============================================================

MAX_ITERATIONS = 10000
DEFAULT_SEEDS = 20
OUTPUT_DIR = Path("output/tsp_mutation_comparison")


# ============================================================
# EA CORE
# ============================================================

def OnePlusOneEA_TSP_Generic(
    distance_matrix,
    city_coords,
    operator_name: str = "2opt",
    max_iterations: int = MAX_ITERATIONS,
):

    current = gaoperators.generate_random_ham_cycle(distance_matrix)
    current_cost = tour_cost(current, distance_matrix)

    best = current[:]
    best_cost = current_cost

    fitness_evaluations = 1
    cost_over_time = [best_cost]

    for _ in range(max_iterations):

        # mutation
        if operator_name == "2opt":
            neighbor = gaoperators.two_opt_mutation(current)

        elif operator_name == "3opt":
            if hasattr(gaoperators, "three_opt_mutation"):
                neighbor = gaoperators.three_opt_mutation(current, distance_matrix)
            else:
                raise NotImplementedError("3-opt not implemented in gaoperators")

        else:
            raise ValueError(f"Unknown operator: {operator_name}")

        neighbor_cost = tour_cost(neighbor, distance_matrix)
        fitness_evaluations += 1

        # (1+1) EA selection
        if neighbor_cost <= current_cost:
            current = neighbor
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        cost_over_time.append(best_cost)

    return best, fitness_evaluations, cost_over_time


# ============================================================
# SINGLE RUN
# ============================================================

def run_single_trial(
    distance_matrix,
    city_coords,
    operator_name,
    seed,
    instance_name,
    max_iterations,
):

    random.seed(seed)
    np.random.seed(seed)

    start = time.time()

    best_tour, fitness_evals, cost_history = OnePlusOneEA_TSP_Generic(
        distance_matrix,
        city_coords,
        operator_name,
        max_iterations,
    )

    elapsed = time.time() - start
    final_cost = tour_cost(best_tour, distance_matrix)

    return {
        "instance": instance_name,
        "operator": operator_name,
        "seed": seed,
        "fitness_evaluations": fitness_evals,
        "final_cost": final_cost,
        "elapsed_seconds": elapsed,
        "cost_history": cost_history,
    }


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(instance_name, num_seeds, max_iterations):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    name, problem, coords, nodes, distance_matrix = fetch_tsp_instance(instance_name)

    optimum = get_optimum(instance_name)

    print(f"\nInstance: {instance_name}")
    print(f"Cities: {len(nodes)}")
    print(f"Optimum: {optimum}\n")

    operators = ["2opt", "3opt"]
    all_results = {}

    for op in operators:

        print(f"Running operator: {op}")
        runs = []

        for seed in range(num_seeds):

            print(f"  seed {seed+1}/{num_seeds}")

            result = run_single_trial(
                distance_matrix,
                coords,
                op,
                seed,
                instance_name,
                max_iterations,
            )

            runs.append(result)
            print(f"    cost: {result['final_cost']:.2f} | time: {result['elapsed_seconds']:.4f}s")

        all_results[op] = runs

        # =========================
        # SUMMARY
        # =========================
        costs = [r["final_cost"] for r in runs]
        times = [r["elapsed_seconds"] for r in runs]

        print(f"\n{op} summary:")
        print(f"  mean cost: {np.mean(costs):.2f}")
        print(f"  std cost : {np.std(costs):.2f}")
        print(f"  min cost : {np.min(costs):.2f}")

        print(f"  mean time: {np.mean(times):.4f} sec")
        print(f"  std time : {np.std(times):.4f} sec")
        print(f"  min time : {np.min(times):.4f} sec")
        print(f"  max time : {np.max(times):.4f} sec")

    # ========================================================
    # SAVE DATA
    # ========================================================

    out_file = OUTPUT_DIR / f"{instance_name}_mutation_data.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved: {out_file}")

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"2opt": "blue", "3opt": "orange"}

    for op in operators:

        histories = [r["cost_history"] for r in all_results[op]]

        max_len = max(len(h) for h in histories)
        grid = np.arange(max_len)

        padded = np.array([
            np.pad(h, (0, max_len - len(h)), mode="edge")
            for h in histories
        ])

        mean = np.mean(padded, axis=0)
        std = np.std(padded, axis=0)

        ax.plot(grid, mean, label=op, color=colors[op])
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)

    if optimum:
        ax.axhline(optimum, linestyle="--", color="green", label="optimum")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Tour Cost")
    ax.set_title(f"(1+1) EA TSP Mutation Comparison - {instance_name}")
    ax.legend()
    ax.grid()

    ax.set_ylim(0, 30000)
    ax.set_yticks(np.arange(0, 30001, 5000))

    plot_file = OUTPUT_DIR / f"{instance_name}_mutation_plot.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()

    print(f"Plot saved: {plot_file}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="berlin52")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)

    args = parser.parse_args()

    run_experiment(
        args.instance,
        args.seeds,
        args.max_iterations,
    )