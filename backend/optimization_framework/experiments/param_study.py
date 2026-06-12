# @author: Andrej Kitanovski
"""Parameter sensitivity sweeps for thesis figures.

Runs small factorial sweeps (few seeds) and writes CSV + box/bar plots under
``output/param_study/``. Includes a uniform random-search baseline on bitstrings
for context when comparing SA cooling or MMAS evaporation rate.

Usage:
    cd backend
    python -m optimization_framework.experiments.param_study
    python -m optimization_framework.experiments.param_study --study sa-cooling --seeds 10
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimization_framework.experiments.batch_runner import (
    _run_bitstring,
    _run_tsp,
    _seed_everything,
)
from optimization_framework.problems import tsp as tsp_mod
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.problems.leadingones import fitnessLeadingOnes
from optimization_framework.operators import gaoperators

DEFAULT_OUTPUT = "output/param_study"
DEFAULT_SEEDS = 10
DEFAULT_BIT_LENGTH = 50


def _random_search(fitness_fn, bit_length, max_iterations=100000):
    """Uniform random restarts: sample a full bitstring each step."""
    best = gaoperators.generateSingleBitstring(bit_length)
    best_fit = fitness_fn(best)
    evals = 1
    iterations = 0
    while best_fit < bit_length and iterations < max_iterations:
        iterations += 1
        candidate = gaoperators.generateSingleBitstring(bit_length)
        fit = fitness_fn(candidate)
        evals += 1
        if fit > best_fit:
            best, best_fit = candidate, fit
    return {
        "iterations": iterations,
        "fitness_evaluations": evals,
        "reached_optimum": best_fit == bit_length,
    }


def _write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def _boxplot_by_param(rows, param_key, y_key, title, out_path):
    groups = defaultdict(list)
    for r in rows:
        groups[r[param_key]].append(r[y_key])
    labels = sorted(groups.keys(), key=lambda x: (isinstance(x, str), x))
    data = [groups[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=[str(k) for k in labels])
    ax.set_title(title)
    ax.set_ylabel(y_key.replace("_", " "))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def study_sa_cooling(problem, bit_length, seeds, output_dir, coolings):
    fitness = fitnessOnemax if problem == "onemax" else fitnessLeadingOnes
    rows = []
    for cooling in coolings:
        for seed in range(seeds):
            _seed_everything(seed)
            row, _ = _run_bitstring(
                problem,
                "Simulated Annealing",
                bit_length,
                seed,
                extra_params={"cooling": cooling, "max_iterations": 100000},
            )
            rows.append(
                {
                    "study": "sa_cooling",
                    "problem": problem,
                    "cooling": cooling,
                    "seed": seed,
                    "fitness_evaluations": row["fitness_evaluations"],
                    "reached_optimum": row["reached_optimum"],
                }
            )
        for seed in range(seeds):
            _seed_everything(seed + 10000)
            rs = _random_search(fitness, bit_length)
            rows.append(
                {
                    "study": "sa_cooling",
                    "problem": problem,
                    "cooling": "random_baseline",
                    "seed": seed,
                    "fitness_evaluations": rs["fitness_evaluations"],
                    "reached_optimum": rs["reached_optimum"],
                }
            )
    out = Path(output_dir)
    fields = ["study", "problem", "cooling", "seed", "fitness_evaluations", "reached_optimum"]
    _write_csv(rows, out / f"sa_cooling_{problem}.csv", fields)
    _boxplot_by_param(
        rows,
        "cooling",
        "fitness_evaluations",
        f"SA cooling sweep ({problem}, n={bit_length})",
        out / f"sa_cooling_{problem}.png",
    )


def study_mmas_rho(problem, bit_length, seeds, output_dir, rhos):
    rows = []
    for rho in rhos:
        for seed in range(seeds):
            row, _ = _run_bitstring(
                problem,
                "ACO",
                bit_length,
                seed,
                extra_params={"rho": rho, "max_iterations": 100000},
            )
            rows.append(
                {
                    "study": "mmas_rho",
                    "problem": problem,
                    "rho": rho,
                    "seed": seed,
                    "fitness_evaluations": row["fitness_evaluations"],
                    "reached_optimum": row["reached_optimum"],
                }
            )
    out = Path(output_dir)
    fields = ["study", "problem", "rho", "seed", "fitness_evaluations", "reached_optimum"]
    _write_csv(rows, out / f"mmas_rho_{problem}.csv", fields)
    _boxplot_by_param(
        rows,
        "rho",
        "fitness_evaluations",
        f"MMAS ρ sweep ({problem}, n={bit_length})",
        out / f"mmas_rho_{problem}.png",
    )


def study_mmas_alpha_beta(tsp_instance, seeds, output_dir, alphas, betas, max_iterations=2000):
    """α/β sweep on TSP (bitstring MMAS has no α/β parameters)."""
    name, _problem, coords, _nodes, dist = tsp_mod.fetch_tsp_instance(tsp_instance)
    rows = []
    for alpha in alphas:
        for beta in betas:
            for seed in range(seeds):
                row, _ = _run_tsp(
                    "ACO",
                    name,
                    dist,
                    coords,
                    {"alpha": alpha, "beta": beta, "max_iterations": max_iterations},
                    seed,
                )
                rows.append(
                    {
                        "study": "mmas_alpha_beta",
                        "instance": name,
                        "alpha": alpha,
                        "beta": beta,
                        "seed": seed,
                        "fitness_evaluations": row["fitness_evaluations"],
                        "gap_percent": row["gap_percent"],
                    }
                )
    out = Path(output_dir)
    fields = ["study", "instance", "alpha", "beta", "seed", "fitness_evaluations", "gap_percent"]
    _write_csv(rows, out / f"mmas_alpha_beta_{name}.csv", fields)
    # heatmap of mean evals
    means = defaultdict(list)
    for r in rows:
        means[(r["alpha"], r["beta"])].append(r["fitness_evaluations"])
    alphas_s = sorted(alphas)
    betas_s = sorted(betas)
    grid = np.zeros((len(alphas_s), len(betas_s)))
    for i, a in enumerate(alphas_s):
        for j, b in enumerate(betas_s):
            grid[i, j] = np.mean(means[(a, b)])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_xticks(range(len(betas_s)), labels=[str(b) for b in betas_s])
    ax.set_yticks(range(len(alphas_s)), labels=[str(a) for a in alphas_s])
    ax.set_xlabel("β")
    ax.set_ylabel("α")
    ax.set_title(f"MMAS α/β mean evals (TSP {name})")
    fig.colorbar(im, ax=ax, label="mean fitness evaluations")
    fig.tight_layout()
    fig.savefig(out / f"mmas_alpha_beta_{name}.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out / f'mmas_alpha_beta_{name}.png'}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sensitivity sweeps.")
    parser.add_argument(
        "--study",
        choices=["all", "sa-cooling", "mmas-rho", "mmas-alpha-beta"],
        default="all",
    )
    parser.add_argument("--problem", default="onemax", choices=["onemax", "leadingones"])
    parser.add_argument("--bit-length", type=int, default=DEFAULT_BIT_LENGTH)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tsp-instance",
        default="burma14.tsp",
        help="TSPLIB instance for α/β sweep (requires tsplib clone).",
    )
    args = parser.parse_args()

    if args.study in ("all", "sa-cooling"):
        study_sa_cooling(
            args.problem,
            args.bit_length,
            args.seeds,
            args.output_dir,
            coolings=[0.9, 0.95, 0.99, 0.999],
        )
    if args.study in ("all", "mmas-rho"):
        study_mmas_rho(
            args.problem,
            args.bit_length,
            args.seeds,
            args.output_dir,
            rhos=[0.01, 0.05, 0.1, 0.3],
        )
    if args.study in ("all", "mmas-alpha-beta"):
        study_mmas_alpha_beta(
            args.tsp_instance,
            args.seeds,
            args.output_dir,
            alphas=[0.5, 1, 2],
            betas=[1, 2, 5],
        )


if __name__ == "__main__":
    main()
