# @author: Andrej Kitanovski
"""
TSP experiment: framework (mu+lambda) EA on berlin52.

The TSP version uses tournament selection, order crossover, 2-opt mutation, and
plus-selection. Configurations are compared under approximately the same fitness
evaluation budget, because larger lambda values evaluate more tours per
generation.

Output (under output/scaling_experiments):
  - CSV: tsp_mu_lambda_berlin52_raw.csv
  - MD : tsp_mu_lambda_berlin52_summary.md
  - PNG: tsp_mu_lambda_berlin52_convergence.png

Usage:
    cd backend
    python3 -m optimization_framework.experiments.tsp_mu_lambda_berlin52
    # smoke run:
    python3 -m optimization_framework.experiments.tsp_mu_lambda_berlin52 \
        --seeds 2 --eval-budget 2000
"""

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.algorithms.mu_plus_lambda_EA import MuPlusLambdaEATSP
from optimization_framework.problems.tsp import fetch_tsp_instance, gap_percent, get_optimum, tour_cost


REPO_ROOT = Path(__file__).resolve().parents[3]

INSTANCE = "berlin52"
DEFAULT_SEEDS = 20
DEFAULT_EVAL_BUDGET = 80000
DEFAULT_TOURNAMENT_K = 3
OUTPUT_DIR = REPO_ROOT / "output/scaling_experiments"

EA_CONFIGS = {
    "(1+1)": (1, 1),
    "(5+20)": (5, 20),
    "(10+40)": (10, 40),
    "(20+80)": (20, 80),
}


def max_generations_for_budget(mu_size: int, lambda_size: int, eval_budget: int) -> int:
    """Convert a total evaluation budget into a generation budget."""
    remaining = max(0, eval_budget - mu_size)
    return max(1, remaining // lambda_size)


def evaluation_axis(mu_size: int, lambda_size: int, history_len: int) -> np.ndarray:
    """Return the fitness-evaluation x-axis for a cost history."""
    return np.array([mu_size + i * lambda_size for i in range(history_len)], dtype=float)


def run_single_trial(
    distance_matrix,
    city_coords,
    instance_name: str,
    config_name: str,
    mu_size: int,
    lambda_size: int,
    seed: int,
    eval_budget: int = DEFAULT_EVAL_BUDGET,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
) -> Dict:
    """Run one TSP trial for one (mu, lambda) configuration."""
    random.seed(seed)
    np.random.seed(seed)

    max_iterations = max_generations_for_budget(mu_size, lambda_size, eval_budget)
    start_time = time.time()
    best_tour, iterations, _, _population, fitness_evals, _tour_coords, cost_history = MuPlusLambdaEATSP(
        distance_matrix=distance_matrix,
        city_coords=city_coords,
        mu_size=mu_size,
        lambda_size=lambda_size,
        tournament_k=tournament_k,
        max_iterations=max_iterations,
    )
    elapsed = time.time() - start_time

    best_cost = tour_cost(best_tour, distance_matrix)
    gap = gap_percent(best_cost, instance_name)

    return {
        "instance": instance_name,
        "config": config_name,
        "mu": mu_size,
        "lambda": lambda_size,
        "tournament_k": tournament_k,
        "seed": seed,
        "eval_budget": eval_budget,
        "max_generations": max_iterations,
        "fitness_evaluations": fitness_evals,
        "iterations": iterations,
        "best_cost": best_cost,
        "gap_percent": gap,
        "hit_optimum": gap == 0.0 if gap is not None else False,
        "elapsed_seconds": elapsed,
        "cost_history": cost_history,
    }


def run_experiment(
    instance_name: str = INSTANCE,
    seeds: int = DEFAULT_SEEDS,
    eval_budget: int = DEFAULT_EVAL_BUDGET,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
) -> Tuple[List[Dict], Dict]:
    """Run all TSP (mu+lambda) configurations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name, _problem, city_coords, nodes, distance_matrix = fetch_tsp_instance(instance_name)
    optimum = get_optimum(name)

    print(f"Starting framework (mu+lambda) EA TSP experiment on {name}")
    print(f"  Cities: {len(nodes)}")
    print(f"  Known optimum: {optimum}")
    print(f"  Configs: {list(EA_CONFIGS.keys())}")
    print(f"  Seeds per config: {seeds}")
    print(f"  Evaluation budget: {eval_budget}")
    print()

    all_results: List[Dict] = []
    aggregated: Dict[str, Dict] = {}
    total_runs = len(EA_CONFIGS) * seeds
    run_count = 0

    for config_name, (mu_size, lambda_size) in EA_CONFIGS.items():
        runs = []
        for seed in range(seeds):
            run_count += 1
            trial = run_single_trial(
                distance_matrix=distance_matrix,
                city_coords=city_coords,
                instance_name=name,
                config_name=config_name,
                mu_size=mu_size,
                lambda_size=lambda_size,
                seed=seed,
                eval_budget=eval_budget,
                tournament_k=tournament_k,
            )
            runs.append(trial)
            all_results.append(trial)

            gap = trial["gap_percent"]
            gap_text = "" if gap is None else f", gap={gap:.2f}%"
            print(
                f"[{run_count:3d}/{total_runs}] cfg={config_name:7s}, seed={seed:2d}: "
                f"cost={trial['best_cost']:.0f}{gap_text}, evals={trial['fitness_evaluations']}"
            )

        costs = [r["best_cost"] for r in runs]
        gaps = [r["gap_percent"] for r in runs if r["gap_percent"] is not None]
        aggregated[config_name] = {
            "mu": mu_size,
            "lambda": lambda_size,
            "trials": len(runs),
            "eval_budget": eval_budget,
            "max_generations": max_generations_for_budget(mu_size, lambda_size, eval_budget),
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "min_cost": float(np.min(costs)),
            "max_cost": float(np.max(costs)),
            "mean_gap_percent": float(np.mean(gaps)) if gaps else None,
            "std_gap_percent": float(np.std(gaps)) if gaps else None,
            "hit_rate": sum(1 for r in runs if r["hit_optimum"]) / len(runs),
        }

    print(f"\nExperiment complete. {run_count} trials run.")
    return all_results, aggregated


def save_results(all_results: List[Dict], aggregated: Dict):
    """Save raw CSV and summary Markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = OUTPUT_DIR / "tsp_mu_lambda_berlin52_raw.csv"
    summary_file = OUTPUT_DIR / "tsp_mu_lambda_berlin52_summary.md"

    fieldnames = [
        "instance",
        "config",
        "mu",
        "lambda",
        "tournament_k",
        "seed",
        "eval_budget",
        "max_generations",
        "fitness_evaluations",
        "iterations",
        "best_cost",
        "gap_percent",
        "hit_optimum",
        "elapsed_seconds",
    ]

    with open(raw_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"Saved raw results to {raw_file}")

    with open(summary_file, "w") as f:
        f.write("# TSP (mu+lambda) EA on berlin52\n\n")
        f.write("| config | mu | lambda | eval budget | mean cost | std | min cost | mean gap % | hit rate |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for config_name, stats in aggregated.items():
            gap = stats["mean_gap_percent"]
            gap_text = "" if gap is None else f"{gap:.2f}"
            f.write(
                f"| {config_name} | {stats['mu']} | {stats['lambda']} | "
                f"{stats['eval_budget']} | {stats['mean_cost']:.1f} | "
                f"{stats['std_cost']:.1f} | {stats['min_cost']:.1f} | "
                f"{gap_text} | {stats['hit_rate']:.2f} |\n"
            )
    print(f"Saved summary to {summary_file}")


def _mean_curve(histories: List[List[float]]) -> np.ndarray:
    """Pad histories with their last value and return mean curve."""
    max_len = max(len(h) for h in histories)
    padded = np.array([np.pad(h, (0, max_len - len(h)), mode="edge") for h in histories])
    return np.mean(padded, axis=0)


def plot_results(all_results: List[Dict], aggregated: Dict, optimum=None):
    """Plot mean best-cost convergence against fitness evaluations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "(1+1)": "#10b981",
        "(5+20)": "#3b82f6",
        "(10+40)": "#f59e0b",
        "(20+80)": "#ef4444",
    }

    for config_name, stats in aggregated.items():
        runs = [r for r in all_results if r["config"] == config_name]
        histories = [r["cost_history"] for r in runs]
        mean = _mean_curve(histories)
        x = evaluation_axis(stats["mu"], stats["lambda"], len(mean))
        ax.plot(
            x,
            mean,
            label=(
                f"{config_name} "
                f"(mean={stats['mean_cost']:.0f}, gap={stats['mean_gap_percent']:.2f}%)"
                if stats["mean_gap_percent"] is not None
                else f"{config_name} (mean={stats['mean_cost']:.0f})"
            ),
            color=colors.get(config_name),
            linewidth=2.5,
        )

    if optimum is not None:
        ax.axhline(optimum, linestyle="--", color="green", label=f"Optimum: {optimum}")

    ax.set_xlabel("Fitness Evaluations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tour Cost", fontsize=12, fontweight="bold")
    ax.set_title(
        "Framework (mu+lambda) EA on berlin52: Population Size Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend()
    fig.tight_layout()

    plot_file = OUTPUT_DIR / "tsp_mu_lambda_berlin52_convergence.png"
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_file}")


def print_summary(aggregated: Dict):
    print("\n" + "=" * 88)
    print("SUMMARY: Framework (mu+lambda) EA on berlin52")
    print("=" * 88)
    print(
        f"{'config':>10} | {'mu':>4} | {'lambda':>6} | {'mean cost':>10} | "
        f"{'std':>8} | {'min':>8} | {'gap %':>8} | {'hit %':>7}"
    )
    print("-" * 88)
    for config_name, stats in aggregated.items():
        gap = stats["mean_gap_percent"]
        gap_text = "" if gap is None else f"{gap:8.2f}"
        print(
            f"{config_name:>10} | {stats['mu']:4d} | {stats['lambda']:6d} | "
            f"{stats['mean_cost']:10.1f} | {stats['std_cost']:8.1f} | "
            f"{stats['min_cost']:8.1f} | {gap_text} | {stats['hit_rate']*100:6.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Framework (mu+lambda) EA on berlin52 under equal evaluation budget"
    )
    parser.add_argument("--instance", default=INSTANCE)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--eval-budget", type=int, default=DEFAULT_EVAL_BUDGET)
    parser.add_argument("--tournament-k", type=int, default=DEFAULT_TOURNAMENT_K)
    args = parser.parse_args()

    all_results, aggregated = run_experiment(
        instance_name=args.instance,
        seeds=args.seeds,
        eval_budget=args.eval_budget,
        tournament_k=args.tournament_k,
    )
    save_results(all_results, aggregated)
    optimum = get_optimum(args.instance)
    plot_results(all_results, aggregated, optimum=optimum)
    print_summary(aggregated)


if __name__ == "__main__":
    main()
