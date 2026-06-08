"""
Crossover-comparison experiment for the GA-style (μ+λ) variant.

Runs the recombination-based ``MuPlusLambdaGA`` with each bitstring crossover
operator on OneMax and LeadingOnes, over several seeds, and reports which
crossover reaches the optimum in the fewest fitness evaluations (and how often).

Crossover types compared:
  - single_point : classic one-cut-point crossover
  - two_point    : swap the middle segment between two cut points
  - three_point  : k-point crossover with k=3 (alternating segments)
  - four_point   : k-point crossover with k=4 (alternating segments)
  - uniform      : each position independently inherited from either parent

Metric: fitness evaluations to reach the optimum (primary, fair across configs
since each generation costs λ evaluations) and success rate. Runs that hit the
generation cap without reaching the optimum are excluded from the mean-evals
statistic but counted in the success rate.

Output (under output/crossover_comparison):
  - CSV : crossover_comparison_results.csv             (one row per run)
  - CSV : crossover_comparison_curve_points.csv        (mean convergence curves)
  - JSON: crossover_comparison_curve_points.json       (same curve data as JSON)
  - PNG : crossover_comparison_<problem>.png           (mean evals per crossover, bars)
  - PNG : crossover_convergence_<problem>.png          (mean fitness over evaluations)

Usage:
    cd backend
    python3 -m optimization_framework.experiments.crossover_comparison
    # smoke run (tiny):
    python3 -m optimization_framework.experiments.crossover_comparison \
        --problems onemax --bit-length 30 --seeds 3 --max-iterations 5000
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

from optimization_framework.algorithms.mu_plus_lambda_EA import MuPlusLambdaGA
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.problems.leadingones import fitnessLeadingOnes


FITNESS_FNS = {
    "onemax": fitnessOnemax,
    "leadingones": fitnessLeadingOnes,
}

CROSSOVER_TYPES = ["single_point", "two_point", "three_point", "four_point", "uniform"]

DEFAULT_PROBLEMS = ["onemax", "leadingones"]
DEFAULT_BIT_LENGTH = 100
DEFAULT_SEEDS = 50
DEFAULT_MU = 5
DEFAULT_LAMBDA = 25
DEFAULT_MAX_ITERATIONS = 200000
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "output/crossover_comparison"

COLORS = {
    "single_point": "#3b82f6",
    "two_point": "#10b981",
    "three_point": "#f59e0b",
    "four_point": "#8b5cf6",
    "uniform": "#ef4444",
}


def run_single_trial(
    problem: str,
    crossover_type: str,
    bit_length: int,
    seed: int,
    mu_size: int = DEFAULT_MU,
    lambda_size: int = DEFAULT_LAMBDA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Dict:
    """Run a single GA-style (μ+λ) trial with a given crossover type."""
    fitness_fn = FITNESS_FNS[problem]

    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    best, iterations, _, _, fitness_evals, coords, history = MuPlusLambdaGA(
        fitness_fn=fitness_fn,
        bit_length=bit_length,
        mu_size=mu_size,
        lambda_size=lambda_size,
        mutation_prob=1.0 / bit_length,
        crossover_type=crossover_type,
        max_iterations=max_iterations,
    )
    elapsed = time.time() - start_time

    final_fitness = best["fitness"] if isinstance(best, dict) else fitness_fn(best)
    reached_optimum = final_fitness == bit_length

    return {
        "problem": problem,
        "crossover": crossover_type,
        "bit_length": bit_length,
        "mu": mu_size,
        "lambda": lambda_size,
        "seed": seed,
        "fitness_evaluations": fitness_evals,
        "iterations": iterations,
        "final_fitness": final_fitness,
        "reached_optimum": reached_optimum,
        "elapsed_seconds": elapsed,
        "fitness_history": history,
    }


def run_experiment(
    problems: List[str] = None,
    crossover_types: List[str] = None,
    bit_length: int = DEFAULT_BIT_LENGTH,
    seeds: int = DEFAULT_SEEDS,
    mu_size: int = DEFAULT_MU,
    lambda_size: int = DEFAULT_LAMBDA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
):
    """Run the full crossover comparison; returns (all_results, aggregated)."""
    if problems is None:
        problems = DEFAULT_PROBLEMS
    if crossover_types is None:
        crossover_types = CROSSOVER_TYPES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    aggregated = {p: {} for p in problems}

    print("Starting crossover-comparison experiment (GA-style (μ+λ))")
    print(f"  Problems: {problems}")
    print(f"  Crossovers: {crossover_types}")
    print(f"  n={bit_length}, μ={mu_size}, λ={lambda_size}, seeds={seeds}")
    print(f"  Max generations: {max_iterations}")
    print()

    total = len(problems) * len(crossover_types) * seeds
    count = 0

    for problem in problems:
        for crossover_type in crossover_types:
            runs = []
            for seed in range(seeds):
                count += 1
                trial = run_single_trial(
                    problem, crossover_type, bit_length, seed,
                    mu_size, lambda_size, max_iterations,
                )
                all_results.append(trial)
                runs.append(trial)

                status = "✓" if trial["reached_optimum"] else "✗"
                print(
                    f"[{count:3d}/{total}] {problem:12s} | {crossover_type:12s} | "
                    f"seed={seed:2d}: evals={trial['fitness_evaluations']:8d} {status}"
                )

            evals = [r["fitness_evaluations"] for r in runs if r["reached_optimum"]]
            success = sum(1 for r in runs if r["reached_optimum"])
            aggregated[problem][crossover_type] = {
                "mean_evals": float(np.mean(evals)) if evals else float("inf"),
                "std_evals": float(np.std(evals)) if evals else 0.0,
                "success_rate": success / len(runs),
                "trials": len(runs),
            }

    print(f"\nExperiment complete. {count} trials run.")
    return all_results, aggregated


def save_results(all_results: List[Dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = OUTPUT_DIR / "crossover_comparison_results.csv"
    fieldnames = [
        "problem", "crossover", "bit_length", "mu", "lambda", "seed",
        "fitness_evaluations", "iterations", "final_fitness",
        "reached_optimum", "elapsed_seconds",
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved raw results to {csv_file}")


def build_curve_rows(all_results: List[Dict], max_points: int = 250) -> List[Dict]:
    """Aggregate run histories into mean convergence curves.

    History index 0 is after the initial population has been evaluated. For a
    fixed (mu, lambda) configuration, generation i corresponds to approximately
    ``mu + i * lambda`` fitness evaluations.
    """
    grouped = {}
    for row in all_results:
        if "fitness_history" not in row:
            continue
        key = (row["problem"], row["crossover"])
        grouped.setdefault(key, []).append(row)

    curve_rows = []
    for (problem, crossover_type), runs in grouped.items():
        max_len = max((len(r["fitness_history"]) for r in runs), default=0)
        if max_len == 0:
            continue

        point_count = min(max_points, max_len)
        generation_grid = np.linspace(0, max_len - 1, point_count, dtype=int)
        generation_grid = sorted(set(int(g) for g in generation_grid))

        mu_size = runs[0]["mu"]
        lambda_size = runs[0]["lambda"]
        bit_length = runs[0]["bit_length"]

        for generation in generation_grid:
            values = [
                r["fitness_history"][min(generation, len(r["fitness_history"]) - 1)]
                for r in runs
            ]
            curve_rows.append(
                {
                    "problem": problem,
                    "crossover": crossover_type,
                    "generation": generation,
                    "fitness_evaluations": int(mu_size + generation * lambda_size),
                    "mean_fitness": float(np.mean(values)),
                    "std_fitness": float(np.std(values)),
                    "min_fitness": float(np.min(values)),
                    "max_fitness": float(np.max(values)),
                    "bit_length": bit_length,
                    "trials": len(runs),
                }
            )

    return curve_rows


def save_curve_points(all_results: List[Dict], max_points: int = 250):
    """Save averaged convergence curves as CSV and JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_rows = build_curve_rows(all_results, max_points=max_points)

    csv_file = OUTPUT_DIR / "crossover_comparison_curve_points.csv"
    json_file = OUTPUT_DIR / "crossover_comparison_curve_points.json"
    fieldnames = [
        "problem",
        "crossover",
        "generation",
        "fitness_evaluations",
        "mean_fitness",
        "std_fitness",
        "min_fitness",
        "max_fitness",
        "bit_length",
        "trials",
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)
    print(f"Saved convergence curve points to {csv_file}")

    with open(json_file, "w") as f:
        json.dump(curve_rows, f, indent=2)
    print(f"Saved convergence curve points to {json_file}")

    return curve_rows


def plot_results(aggregated: Dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for problem, by_cx in aggregated.items():
        crossovers = list(by_cx.keys())
        means = [by_cx[c]["mean_evals"] for c in crossovers]
        stds = [by_cx[c]["std_evals"] for c in crossovers]
        success = [by_cx[c]["success_rate"] for c in crossovers]

        # Replace inf (never reached optimum) with 0 height; annotate as DNF.
        plot_means = [m if np.isfinite(m) else 0.0 for m in means]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        bars = ax.bar(
            crossovers, plot_means, yerr=stds,
            color=[COLORS.get(c, "#888") for c in crossovers],
            capsize=5, alpha=0.85, edgecolor="black", linewidth=0.6,
        )
        for bar, m, s in zip(bars, means, success):
            label = "DNF" if not np.isfinite(m) else f"{s*100:.0f}% ok"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label, ha="center", va="bottom", fontsize=9,
            )

        ax.set_ylabel("Mean Fitness Evaluations to Optimum", fontsize=11, fontweight="bold")
        ax.set_xlabel("Crossover Type", fontsize=11, fontweight="bold")
        ax.set_title(
            f"GA-style (μ+λ) on {problem}: Crossover Comparison",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
        fig.tight_layout()
        path = OUTPUT_DIR / f"crossover_comparison_{problem}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved plot to {path}")


def plot_convergence_curves(all_results: List[Dict], max_points: int = 250):
    """Plot mean best fitness over fitness evaluations for each crossover."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_rows = build_curve_rows(all_results, max_points=max_points)

    by_problem = {}
    for row in curve_rows:
        by_problem.setdefault(row["problem"], {}).setdefault(row["crossover"], []).append(row)

    for problem, by_crossover in by_problem.items():
        fig, ax = plt.subplots(figsize=(11, 6.5))
        bit_length = None

        for crossover_type, rows in by_crossover.items():
            rows = sorted(rows, key=lambda r: r["fitness_evaluations"])
            x = np.array([r["fitness_evaluations"] for r in rows], dtype=float)
            mean = np.array([r["mean_fitness"] for r in rows], dtype=float)
            std = np.array([r["std_fitness"] for r in rows], dtype=float)
            bit_length = rows[0]["bit_length"]
            color = COLORS.get(crossover_type, "#888")

            ax.plot(
                x,
                mean,
                label=crossover_type,
                color=color,
                linewidth=2.3,
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

        if bit_length is not None:
            ax.axhline(
                bit_length,
                linestyle="--",
                color="#111827",
                linewidth=1.2,
                label=f"Optimum: {bit_length}",
            )

        ax.set_xlabel("Fitness Evaluations", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Best Fitness", fontsize=11, fontweight="bold")
        ax.set_title(
            f"GA-style (μ+λ) on {problem}: Crossover Convergence",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend()
        fig.tight_layout()

        path = OUTPUT_DIR / f"crossover_convergence_{problem}.png"
        fig.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved convergence plot to {path}")


def print_summary(aggregated: Dict):
    print("\n" + "=" * 72)
    print("SUMMARY: Crossover comparison (GA-style (μ+λ))")
    print("=" * 72)
    for problem, by_cx in aggregated.items():
        print(f"\n{problem}:")
        print("-" * 72)
        print(f"{'crossover':>14} | {'mean evals':>12} | {'std':>10} | {'success %':>9}")
        print("-" * 72)
        finite = {c: v for c, v in by_cx.items() if np.isfinite(v["mean_evals"])}
        best_cx = min(finite, key=lambda c: finite[c]["mean_evals"]) if finite else None
        for c, v in by_cx.items():
            mark = "  <-- best" if c == best_cx else ""
            mean_str = f"{v['mean_evals']:12.0f}" if np.isfinite(v["mean_evals"]) else f"{'DNF':>12}"
            print(
                f"{c:>14} | {mean_str} | {v['std_evals']:10.0f} | "
                f"{v['success_rate']*100:8.1f}%{mark}"
            )
        if best_cx:
            print(f"  Best crossover for {problem}: {best_cx}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare crossover operators for the GA-style (μ+λ) variant."
    )
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS,
                        choices=DEFAULT_PROBLEMS)
    parser.add_argument("--crossovers", nargs="+", default=CROSSOVER_TYPES,
                        choices=CROSSOVER_TYPES)
    parser.add_argument("--bit-length", type=int, default=DEFAULT_BIT_LENGTH)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--mu", type=int, default=DEFAULT_MU)
    parser.add_argument("--lambda-size", type=int, default=DEFAULT_LAMBDA)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--curve-points",
        type=int,
        default=250,
        help="Maximum number of points in saved convergence curves",
    )
    args = parser.parse_args()

    all_results, aggregated = run_experiment(
        problems=args.problems,
        crossover_types=args.crossovers,
        bit_length=args.bit_length,
        seeds=args.seeds,
        mu_size=args.mu,
        lambda_size=args.lambda_size,
        max_iterations=args.max_iterations,
    )
    save_results(all_results)
    save_curve_points(all_results, max_points=args.curve_points)
    plot_results(aggregated)
    plot_convergence_curves(all_results, max_points=args.curve_points)
    print_summary(aggregated)


if __name__ == "__main__":
    main()
