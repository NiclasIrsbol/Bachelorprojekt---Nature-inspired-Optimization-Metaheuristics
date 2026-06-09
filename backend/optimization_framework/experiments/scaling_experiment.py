"""Scaling experiment: problem size vs. empirical runtime.

Sweeps a bitstring problem across several problem sizes (n = bit length),
runs each algorithm over multiple seeds, and plots how the number of
fitness evaluations needed to reach the optimum grows with n. Fitness
evaluations are the default metric because they are fairer than iterations when
comparing single-solution, population-based, and colony-based algorithms.

It produces two variants per problem (like the reference):
  - "small": n = 10, 20, ..., 100
  - "large": n = 100, 200, ..., 1000

All algorithms are capped via ``--max-iterations`` so large sizes stay feasible.
Runs that hit the cap without reaching the optimum are still included in the
averages and marked in the CSV through ``reached_optimum``.

Reuses the seeded single-run helper from ``batch_runner`` for reproducibility.

Outputs (under the output dir):
  - ``scaling_results.csv``            one row per run
  - ``scaling_<problem>_<sizeset>.png`` mean metric vs. size per algorithm

Usage:
    cd backend
    # both presets, all algorithms
    python -m optimization_framework.experiments.scaling_experiment
    # report-sized bitstring comparison
    python -m optimization_framework.experiments.scaling_experiment --preset small --seeds 30
    # custom sizes
    python -m optimization_framework.experiments.scaling_experiment --sizes 10 50 100 200
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimization_framework.experiments.batch_runner import _run_bitstring

BITSTRING_PROBLEMS = ["onemax", "leadingones"]
ALGORITHMS = ["(1+1) EA", "(μ+λ) EA", "Simulated Annealing", "MMAS-ACO", "P-ACO"]

SMALL_SIZES = list(range(10, 101, 10))      # 10, 20, ..., 100
LARGE_SIZES = list(range(100, 1001, 100))   # 100, 200, ..., 1000

DEFAULT_SEEDS = 20
DEFAULT_MAX_ITERATIONS = 150000             # matches the reference cap
DEFAULT_OUTPUT_DIR = "output/algorithm_comparison_scaling"
SCALING_RESULTS_FILE = "scaling_results.csv"

RESULT_FIELDNAMES = [
    "problem",
    "algorithm",
    "size_set",
    "size",
    "seed",
    "iterations",
    "fitness_evaluations",
    "reached_optimum",
]

from optimization_framework.experiments.theory_curves import CURVES_BY_PROBLEM


def run_scaling(
    problems,
    algorithms,
    size_sets,
    seeds,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    extra_params = {"max_iterations": max_iterations} if max_iterations else None
    rows = []
    for set_name, sizes in size_sets.items():
        for problem in problems:
            if problem == "tsp":
                print("Skipping 'tsp': fixed iterations, size-scaling is not meaningful here.")
                continue
            for algorithm in algorithms:
                for size in sizes:
                    for seed in seeds:
                        print(f"[{set_name}] {problem} | {algorithm} | n={size} | seed={seed}")
                        row, _curve = _run_bitstring(
                            problem, algorithm, size, seed, extra_params=extra_params
                        )
                        rows.append(
                            {
                                "problem": problem,
                                "algorithm": algorithm,
                                "size_set": set_name,
                                "size": size,
                                "seed": seed,
                                "iterations": row["iterations"],
                                "fitness_evaluations": row["fitness_evaluations"],
                                "reached_optimum": row["reached_optimum"],
                            }
                        )

    _write_results(rows, output_dir)
    return rows


def _write_results(rows, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SCALING_RESULTS_FILE
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} runs to {path}")


def _plot(rows, y_metric, show_theory, log_log, output_dir):
    # (size_set, problem) -> algorithm -> size -> {"vals": [...], "opt": [...]}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"vals": [], "opt": []})))
    for r in rows:
        cell = data[(r["size_set"], r["problem"])][r["algorithm"]][r["size"]]
        cell["vals"].append(r[y_metric])
        cell["opt"].append(bool(r["reached_optimum"]))

    ylabel = "average iterations" if y_metric == "iterations" else "average fitness evaluations"

    for (set_name, problem), algos in sorted(data.items()):
        fig, ax = plt.subplots(figsize=(8, 5))

        for algo in sorted(algos.keys()):
            size_map = algos[algo]
            # origin point (n=0 -> 0 iterations), like the reference tables
            xs, means, stds = [0], [0.0], [0.0]
            for size in sorted(size_map.keys()):
                vals = np.array(size_map[size]["vals"], dtype=float)
                xs.append(size)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            xs = np.array(xs)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(xs, means, marker="o", label=algo)
            ax.fill_between(xs, means - stds, means + stds, alpha=0.15)

        if show_theory and problem in CURVES_BY_PROBLEM:
            all_sizes = sorted(
                {
                    r["size"]
                    for r in rows
                    if r["size_set"] == set_name and r["problem"] == problem
                }
            )
            xs = np.array([0] + all_sizes, dtype=float)
            for i, (label, fn) in enumerate(CURVES_BY_PROBLEM[problem]):
                ys = np.array([0.0] + [fn(n) for n in all_sizes])
                ax.plot(
                    xs,
                    ys,
                    linestyle="--",
                    color="black" if i == 0 else "#666666",
                    alpha=0.75,
                    linewidth=1.5,
                    label=label,
                )

        ax.set_title(f"{problem} ({set_name}): {ylabel} vs. problem size")
        ax.set_xlabel("problem size n (bit length)")
        ax.set_ylabel(ylabel)
        if log_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        path = Path(output_dir) / f"scaling_{problem}_{set_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote scaling plot to {path}")


def _resolve_size_sets(args):
    if args.sizes:
        return {"custom": sorted(args.sizes)}
    size_sets = {}
    if args.preset in ("small", "both"):
        size_sets["small"] = SMALL_SIZES
    if args.preset in ("large", "both"):
        size_sets["large"] = LARGE_SIZES
    return size_sets


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot problem size vs. empirical runtime scaling."
    )
    parser.add_argument("--problems", nargs="+", default=BITSTRING_PROBLEMS)
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument(
        "--preset",
        choices=["small", "large", "both"],
        default="both",
        help="Size preset: small (10..100), large (100..1000), or both.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Custom problem sizes (overrides --preset).",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Iteration cap applied to all algorithms (0 disables the cap).",
    )
    parser.add_argument(
        "--y-metric",
        choices=["iterations", "fitness_evaluations"],
        default="fitness_evaluations",
        help="Comparison metric. Default is fitness_evaluations, which is fair "
        "across algorithms because population- and colony-based methods do "
        "many evaluations per iteration.",
    )
    parser.add_argument(
        "--no-theory",
        action="store_true",
        help="Disable exact theoretical reference curves.",
    )
    parser.add_argument(
        "--log-log",
        action="store_true",
        help="Use log-log axes (useful for polynomial growth).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = _parse_args()
    seeds = list(range(args.seeds))
    size_sets = _resolve_size_sets(args)
    rows = run_scaling(
        problems=args.problems,
        algorithms=args.algorithms,
        size_sets=size_sets,
        seeds=seeds,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
    )
    _plot(rows, args.y_metric, not args.no_theory, args.log_log, args.output_dir)


if __name__ == "__main__":
    main()
