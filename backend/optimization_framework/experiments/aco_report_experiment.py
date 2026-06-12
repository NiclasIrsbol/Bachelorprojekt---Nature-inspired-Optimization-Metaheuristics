# @author: Andrej Kitanovski
"""
Full ACO report experiment: MMAS and P-ACO on OneMax, LeadingOnes and berlin52.

This script is intended as the single source of truth for the ACO report figures
and tables. The default ``report`` preset is a balanced run: it still performs a
parameter study, but avoids the very large full grid unless explicitly requested.
It runs over:
  - several bitstring lengths for OneMax and LeadingOnes
  - several MMAS evaporation rates
  - several P-ACO archive sizes and q0 values
  - berlin52 TSP with additional heuristic-weight beta settings

The output is designed to be copied directly into the report:
  - raw CSV with one row per run
  - summary CSV with means/stds/success rates
  - Markdown tables for the best parameter choices
  - JSON summary for reproducibility
  - PNG graphs for scaling, parameter comparison and TSP convergence

Usage:
    cd backend
    python3 -m optimization_framework.experiments.aco_report_experiment

    # Fast verification run, same script and output structure:
    python3 -m optimization_framework.experiments.aco_report_experiment --preset quick

    # Old large grid:
    python3 -m optimization_framework.experiments.aco_report_experiment --preset full
"""

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.algorithms.ant_optimization_problem import (
    ant_colony_optimization,
    ant_colony_optimizationTSP,
    population_based_aco,
    population_based_acoTSP,
)
from optimization_framework.problems.onemax import fitnessOnemax
from optimization_framework.problems.leadingones import fitnessLeadingOnes
from optimization_framework.problems import tsp


# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "output/aco_report_experiment"

DEFAULT_PRESET = "report"

FULL_SEEDS = 15
REPORT_SEEDS = 5
QUICK_SEEDS = 2
DEFAULT_SEEDS = REPORT_SEEDS

FULL_TSP_SEEDS = 15
REPORT_TSP_SEEDS = 15
QUICK_TSP_SEEDS = 2
DEFAULT_TSP_SEEDS = REPORT_TSP_SEEDS

FULL_ONEMAX_SIZES = [100, 200, 300, 400, 500]
REPORT_ONEMAX_SIZES = [100, 300, 500]
QUICK_ONEMAX_SIZES = [50, 100]
ONEMAX_SIZES = REPORT_ONEMAX_SIZES

FULL_LEADINGONES_SIZES = [100, 200, 300, 400, 500]
REPORT_LEADINGONES_SIZES = [100, 300, 500]
QUICK_LEADINGONES_SIZES = [50, 100]
LEADINGONES_SIZES = REPORT_LEADINGONES_SIZES

FULL_ONEMAX_MAX_ITERATIONS = 50000
REPORT_ONEMAX_MAX_ITERATIONS = 50000
QUICK_ONEMAX_MAX_ITERATIONS = 10000
ONEMAX_MAX_ITERATIONS = REPORT_ONEMAX_MAX_ITERATIONS

FULL_LEADINGONES_MAX_ITERATIONS = 200000
REPORT_LEADINGONES_MAX_ITERATIONS = 100000
QUICK_LEADINGONES_MAX_ITERATIONS = 30000
LEADINGONES_MAX_ITERATIONS = REPORT_LEADINGONES_MAX_ITERATIONS

TSP_INSTANCE = "berlin52"
FULL_TSP_MAX_ITERATIONS = 5000
REPORT_TSP_MAX_ITERATIONS = 3000
QUICK_TSP_MAX_ITERATIONS = 300
TSP_MAX_ITERATIONS = REPORT_TSP_MAX_ITERATIONS
TSP_ALPHA = 1
FULL_TSP_BETAS = [2, 3]
REPORT_TSP_BETAS = [2, 3]
QUICK_TSP_BETAS = [3]
TSP_BETAS = REPORT_TSP_BETAS

FULL_MMAS_RHOS = [0.05, 0.10, 0.50]
REPORT_MMAS_RHOS = [0.05, 0.10, 0.50]
QUICK_MMAS_RHOS = [0.10, 0.50]
MMAS_RHOS = REPORT_MMAS_RHOS
MMAS_NUM_ANTS = [10]

FULL_PACO_ARCHIVE_SIZES = [5, 10, 20]
REPORT_PACO_ARCHIVE_SIZES = [10, 20]
QUICK_PACO_ARCHIVE_SIZES = [10, 20]
PACO_ARCHIVE_SIZES = REPORT_PACO_ARCHIVE_SIZES

FULL_TSP_PACO_ARCHIVE_SIZES = [5, 10, 20]
REPORT_TSP_PACO_ARCHIVE_SIZES = [5, 10, 20]
QUICK_TSP_PACO_ARCHIVE_SIZES = [10, 20]
TSP_PACO_ARCHIVE_SIZES = REPORT_TSP_PACO_ARCHIVE_SIZES
PACO_NUM_ANTS = [30]
FULL_PACO_Q0_VALUES = [0.7, 0.9]
REPORT_PACO_Q0_VALUES = [0.7, 0.9]
QUICK_PACO_Q0_VALUES = [0.7]
PACO_Q0_VALUES = REPORT_PACO_Q0_VALUES

EXPERIMENT_PRESETS = {
    "quick": {
        "seeds": QUICK_SEEDS,
        "tsp_seeds": QUICK_TSP_SEEDS,
        "onemax_sizes": QUICK_ONEMAX_SIZES,
        "leadingones_sizes": QUICK_LEADINGONES_SIZES,
        "onemax_max_iterations": QUICK_ONEMAX_MAX_ITERATIONS,
        "leadingones_max_iterations": QUICK_LEADINGONES_MAX_ITERATIONS,
        "tsp_max_iterations": QUICK_TSP_MAX_ITERATIONS,
        "mmas_rhos": QUICK_MMAS_RHOS,
        "mmas_num_ants": MMAS_NUM_ANTS,
        "paco_archive_sizes": QUICK_PACO_ARCHIVE_SIZES,
        "tsp_paco_archive_sizes": QUICK_TSP_PACO_ARCHIVE_SIZES,
        "paco_num_ants": PACO_NUM_ANTS,
        "paco_q0_values": QUICK_PACO_Q0_VALUES,
        "tsp_betas": QUICK_TSP_BETAS,
    },
    "report": {
        "seeds": REPORT_SEEDS,
        "tsp_seeds": REPORT_TSP_SEEDS,
        "onemax_sizes": REPORT_ONEMAX_SIZES,
        "leadingones_sizes": REPORT_LEADINGONES_SIZES,
        "onemax_max_iterations": REPORT_ONEMAX_MAX_ITERATIONS,
        "leadingones_max_iterations": REPORT_LEADINGONES_MAX_ITERATIONS,
        "tsp_max_iterations": REPORT_TSP_MAX_ITERATIONS,
        "mmas_rhos": REPORT_MMAS_RHOS,
        "mmas_num_ants": MMAS_NUM_ANTS,
        "paco_archive_sizes": REPORT_PACO_ARCHIVE_SIZES,
        "tsp_paco_archive_sizes": REPORT_TSP_PACO_ARCHIVE_SIZES,
        "paco_num_ants": PACO_NUM_ANTS,
        "paco_q0_values": REPORT_PACO_Q0_VALUES,
        "tsp_betas": REPORT_TSP_BETAS,
    },
    "full": {
        "seeds": FULL_SEEDS,
        "tsp_seeds": FULL_TSP_SEEDS,
        "onemax_sizes": FULL_ONEMAX_SIZES,
        "leadingones_sizes": FULL_LEADINGONES_SIZES,
        "onemax_max_iterations": FULL_ONEMAX_MAX_ITERATIONS,
        "leadingones_max_iterations": FULL_LEADINGONES_MAX_ITERATIONS,
        "tsp_max_iterations": FULL_TSP_MAX_ITERATIONS,
        "mmas_rhos": FULL_MMAS_RHOS,
        "mmas_num_ants": MMAS_NUM_ANTS,
        "paco_archive_sizes": FULL_PACO_ARCHIVE_SIZES,
        "tsp_paco_archive_sizes": FULL_TSP_PACO_ARCHIVE_SIZES,
        "paco_num_ants": PACO_NUM_ANTS,
        "paco_q0_values": FULL_PACO_Q0_VALUES,
        "tsp_betas": FULL_TSP_BETAS,
    },
}

BITSTRING_PROBLEMS = {
    "onemax": {
        "title": "OneMax",
        "fitness_fn": fitnessOnemax,
        "default_sizes": ONEMAX_SIZES,
        "default_max_iterations": ONEMAX_MAX_ITERATIONS,
    },
    "leadingones": {
        "title": "LeadingOnes",
        "fitness_fn": fitnessLeadingOnes,
        "default_sizes": LEADINGONES_SIZES,
        "default_max_iterations": LEADINGONES_MAX_ITERATIONS,
    },
}

RAW_FIELDS = [
    "problem",
    "problem_type",
    "n",
    "instance",
    "algorithm",
    "config_label",
    "seed",
    "rho",
    "archive_size",
    "num_ants",
    "q0",
    "alpha",
    "beta",
    "max_iterations",
    "fitness_evaluations",
    "iterations",
    "final_fitness",
    "reached_optimum",
    "best_cost",
    "gap_percent",
    "elapsed_seconds",
]

SUMMARY_FIELDS = [
    "problem",
    "problem_type",
    "n",
    "instance",
    "algorithm",
    "config_label",
    "rho",
    "archive_size",
    "num_ants",
    "q0",
    "alpha",
    "beta",
    "trials",
    "success_rate",
    "mean_evals_success",
    "std_evals_success",
    "min_evals_success",
    "max_evals_success",
    "mean_final_fitness",
    "std_final_fitness",
    "mean_cost",
    "std_cost",
    "min_cost",
    "max_cost",
    "mean_gap_percent",
    "std_gap_percent",
    "min_gap_percent",
]


# ============================================================
# CONFIG BUILDERS
# ============================================================

def build_mmas_bitstring_configs(rhos: Iterable[float], ant_counts: Iterable[int]) -> List[Dict]:
    configs = []
    for rho in rhos:
        for num_ants in ant_counts:
            configs.append({
                "algorithm": "MMAS",
                "config_label": f"rho={rho:g}, ants={num_ants}",
                "rho": float(rho),
                "num_ants": int(num_ants),
            })
    return configs


def build_mmas_tsp_configs(
    rhos: Iterable[float],
    ant_counts: Iterable[int],
    betas: Iterable[float],
    alpha: float = TSP_ALPHA,
) -> List[Dict]:
    configs = []
    for rho in rhos:
        for num_ants in ant_counts:
            for beta in betas:
                configs.append({
                    "algorithm": "MMAS",
                    "config_label": f"rho={rho:g}, ants={num_ants}, beta={beta:g}",
                    "rho": float(rho),
                    "num_ants": int(num_ants),
                    "alpha": float(alpha),
                    "beta": float(beta),
                })
    return configs


def build_paco_bitstring_configs(
    archive_sizes: Iterable[int],
    ant_counts: Iterable[int],
    q0_values: Iterable[float],
) -> List[Dict]:
    configs = []
    for archive_size in archive_sizes:
        for num_ants in ant_counts:
            for q0 in q0_values:
                configs.append({
                    "algorithm": "P-ACO",
                    "config_label": f"K={archive_size}, ants={num_ants}, q0={q0:g}",
                    "archive_size": int(archive_size),
                    "num_ants": int(num_ants),
                    "q0": float(q0),
                })
    return configs


def build_paco_tsp_configs(
    archive_sizes: Iterable[int],
    ant_counts: Iterable[int],
    q0_values: Iterable[float],
    betas: Iterable[float],
    alpha: float = TSP_ALPHA,
) -> List[Dict]:
    configs = []
    for archive_size in archive_sizes:
        for num_ants in ant_counts:
            for q0 in q0_values:
                for beta in betas:
                    configs.append({
                        "algorithm": "P-ACO",
                        "config_label": (
                            f"K={archive_size}, ants={num_ants}, "
                            f"q0={q0:g}, beta={beta:g}"
                        ),
                        "archive_size": int(archive_size),
                        "num_ants": int(num_ants),
                        "q0": float(q0),
                        "alpha": float(alpha),
                        "beta": float(beta),
                    })
    return configs


# ============================================================
# RUN HELPERS
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _blank_row() -> Dict:
    return {field: "" for field in RAW_FIELDS}


def run_bitstring_trial(
    problem_key: str,
    n: int,
    config: Dict,
    seed: int,
    max_iterations: int,
) -> Dict:
    problem = BITSTRING_PROBLEMS[problem_key]
    fitness_fn = problem["fitness_fn"]

    seed_everything(seed)
    start = time.time()

    if config["algorithm"] == "MMAS":
        best, iterations, _, _, fitness_evals, _coords, _history = ant_colony_optimization(
            fitness_fn=fitness_fn,
            bit_length=n,
            rho=config["rho"],
            num_ants=config["num_ants"],
            max_iterations=max_iterations,
        )
    elif config["algorithm"] == "P-ACO":
        best, iterations, _, _, fitness_evals, _coords, _history = population_based_aco(
            fitness_fn=fitness_fn,
            bit_length=n,
            archive_size=config["archive_size"],
            num_ants=config["num_ants"],
            q0=config["q0"],
            max_iterations=max_iterations,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")

    elapsed = time.time() - start
    final_fitness = fitness_fn(best) if isinstance(best, str) else best["fitness"]

    row = _blank_row()
    row.update({
        "problem": problem_key,
        "problem_type": "bitstring",
        "n": n,
        "algorithm": config["algorithm"],
        "config_label": config["config_label"],
        "seed": seed,
        "rho": config.get("rho", ""),
        "archive_size": config.get("archive_size", ""),
        "num_ants": config["num_ants"],
        "q0": config.get("q0", ""),
        "max_iterations": max_iterations,
        "fitness_evaluations": fitness_evals,
        "iterations": iterations,
        "final_fitness": final_fitness,
        "reached_optimum": final_fitness == n,
        "elapsed_seconds": elapsed,
    })
    return row


def run_tsp_trial(
    instance_name: str,
    distance_matrix,
    city_coords,
    config: Dict,
    seed: int,
    max_iterations: int,
) -> Tuple[Dict, List[float]]:
    seed_everything(seed)
    optimum = tsp.get_optimum(instance_name)
    start = time.time()

    if config["algorithm"] == "MMAS":
        best_tour, iterations, _, _, fitness_evals, _coords, history = ant_colony_optimizationTSP(
            distance_matrix,
            city_coords,
            rho=config["rho"],
            num_ants=config["num_ants"],
            alpha=config["alpha"],
            beta=config["beta"],
            optimum=optimum,
            max_iterations=max_iterations,
        )
    elif config["algorithm"] == "P-ACO":
        best_tour, iterations, _, _, fitness_evals, _coords, history = population_based_acoTSP(
            distance_matrix,
            city_coords,
            archive_size=config["archive_size"],
            num_ants=config["num_ants"],
            q0=config["q0"],
            alpha=config["alpha"],
            beta=config["beta"],
            max_iterations=max_iterations,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")

    elapsed = time.time() - start
    best_cost = tsp.tour_cost(best_tour, distance_matrix)

    row = _blank_row()
    row.update({
        "problem": "tsp",
        "problem_type": "tsp",
        "n": len(distance_matrix),
        "instance": instance_name,
        "algorithm": config["algorithm"],
        "config_label": config["config_label"],
        "seed": seed,
        "rho": config.get("rho", ""),
        "archive_size": config.get("archive_size", ""),
        "num_ants": config["num_ants"],
        "q0": config.get("q0", ""),
        "alpha": config.get("alpha", ""),
        "beta": config.get("beta", ""),
        "max_iterations": max_iterations,
        "fitness_evaluations": fitness_evals,
        "iterations": iterations,
        "best_cost": best_cost,
        "gap_percent": tsp.gap_percent(best_cost, instance_name),
        "elapsed_seconds": elapsed,
    })
    return row, history


# ============================================================
# AGGREGATION
# ============================================================

def _to_float(value):
    if value == "" or value is None:
        return None
    return float(value)


def _mean(values):
    return float(np.mean(values)) if values else ""


def _std(values):
    return float(np.std(values)) if values else ""


def _min(values):
    return float(np.min(values)) if values else ""


def _max(values):
    return float(np.max(values)) if values else ""


def _summary_key(row: Dict):
    return (
        row["problem"],
        row["problem_type"],
        row["n"],
        row["instance"],
        row["algorithm"],
        row["config_label"],
        row["rho"],
        row["archive_size"],
        row["num_ants"],
        row["q0"],
        row["alpha"],
        row["beta"],
    )


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[_summary_key(row)].append(row)

    summary = []
    for key, group in groups.items():
        (
            problem,
            problem_type,
            n,
            instance,
            algorithm,
            config_label,
            rho,
            archive_size,
            num_ants,
            q0,
            alpha,
            beta,
        ) = key

        out = {field: "" for field in SUMMARY_FIELDS}
        out.update({
            "problem": problem,
            "problem_type": problem_type,
            "n": n,
            "instance": instance,
            "algorithm": algorithm,
            "config_label": config_label,
            "rho": rho,
            "archive_size": archive_size,
            "num_ants": num_ants,
            "q0": q0,
            "alpha": alpha,
            "beta": beta,
            "trials": len(group),
        })

        if problem_type == "bitstring":
            successes = [r for r in group if bool(r["reached_optimum"])]
            evals = [float(r["fitness_evaluations"]) for r in successes]
            final_fitness = [float(r["final_fitness"]) for r in group]
            out.update({
                "success_rate": len(successes) / len(group),
                "mean_evals_success": _mean(evals),
                "std_evals_success": _std(evals),
                "min_evals_success": _min(evals),
                "max_evals_success": _max(evals),
                "mean_final_fitness": _mean(final_fitness),
                "std_final_fitness": _std(final_fitness),
            })
        else:
            costs = [float(r["best_cost"]) for r in group]
            gaps = [_to_float(r["gap_percent"]) for r in group]
            gaps = [g for g in gaps if g is not None]
            optimum = tsp.get_optimum(instance)
            optimal_runs = [c for c in costs if optimum is not None and c <= optimum]
            out.update({
                "success_rate": len(optimal_runs) / len(group) if optimum else "",
                "mean_cost": _mean(costs),
                "std_cost": _std(costs),
                "min_cost": _min(costs),
                "max_cost": _max(costs),
                "mean_gap_percent": _mean(gaps),
                "std_gap_percent": _std(gaps),
                "min_gap_percent": _min(gaps),
            })

        summary.append(out)

    return sorted(summary, key=lambda r: (
        r["problem"],
        r["algorithm"],
        str(r["config_label"]),
        int(r["n"]) if r["n"] != "" else 0,
    ))


def select_best_configs(summary_rows: List[Dict]) -> List[Dict]:
    """Select the best config per problem and algorithm for report discussion."""
    by_problem_algo = defaultdict(list)
    for row in summary_rows:
        by_problem_algo[(row["problem"], row["algorithm"])].append(row)

    best_rows = []
    for (problem, algorithm), rows in sorted(by_problem_algo.items()):
        if rows[0]["problem_type"] == "bitstring":
            max_n = max(int(r["n"]) for r in rows)
            candidates = [r for r in rows if int(r["n"]) == max_n]

            def bit_key(r):
                success = float(r["success_rate"] or 0.0)
                mean_evals = float(r["mean_evals_success"]) if r["mean_evals_success"] != "" else math.inf
                final_fitness = float(r["mean_final_fitness"] or 0.0)
                return (-success, mean_evals, -final_fitness)

            best = min(candidates, key=bit_key)
            reason = (
                f"largest n={max_n}: success={float(best['success_rate']):.2f}, "
                f"mean evals={_fmt(best['mean_evals_success'])}"
            )
        else:
            def tsp_key(r):
                mean_gap = float(r["mean_gap_percent"]) if r["mean_gap_percent"] != "" else math.inf
                mean_cost = float(r["mean_cost"]) if r["mean_cost"] != "" else math.inf
                return (mean_gap, mean_cost)

            best = min(rows, key=tsp_key)
            reason = (
                f"mean gap={_fmt(best['mean_gap_percent'])}%, "
                f"mean cost={_fmt(best['mean_cost'])}"
            )

        best_entry = dict(best)
        best_entry["selection_reason"] = reason
        best_rows.append(best_entry)

    return best_rows


# ============================================================
# OUTPUT
# ============================================================

def write_csv(rows: List[Dict], path: Path, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {path}")


def write_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved JSON: {path}")


def _fmt(value, digits=2):
    if value == "" or value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "inf"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.{digits}f}"


def write_markdown_table(path: Path, rows: List[Dict], columns: List[Tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("| " + " | ".join(header for header, _key in columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            vals = []
            for _header, key in columns:
                value = row.get(key, "")
                vals.append(_fmt(value) if isinstance(value, (float, int)) else str(value))
            f.write("| " + " | ".join(vals) + " |\n")
    print(f"Saved table: {path}")


def save_outputs(
    raw_rows: List[Dict],
    summary_rows: List[Dict],
    best_rows: List[Dict],
    output_dir: Path,
):
    write_csv(raw_rows, output_dir / "aco_report_runs.csv", RAW_FIELDS)
    write_csv(summary_rows, output_dir / "aco_report_summary.csv", SUMMARY_FIELDS)

    best_fields = SUMMARY_FIELDS + ["selection_reason"]
    write_csv(best_rows, output_dir / "aco_report_best_configs.csv", best_fields)

    write_json({
        "raw_rows": len(raw_rows),
        "summary": summary_rows,
        "best_configs": best_rows,
    }, output_dir / "aco_report_results.json")

    write_markdown_table(
        output_dir / "aco_report_best_configs.md",
        best_rows,
        [
            ("Problem", "problem"),
            ("Algorithm", "algorithm"),
            ("Best config", "config_label"),
            ("Success", "success_rate"),
            ("Mean evals", "mean_evals_success"),
            ("Mean cost", "mean_cost"),
            ("Mean gap %", "mean_gap_percent"),
            ("Reason", "selection_reason"),
        ],
    )

    compact_summary = [
        r for r in summary_rows
        if r["problem_type"] == "tsp" or str(r["n"]) in {"100", "500"}
    ]
    write_markdown_table(
        output_dir / "aco_report_summary_compact.md",
        compact_summary,
        [
            ("Problem", "problem"),
            ("n", "n"),
            ("Algorithm", "algorithm"),
            ("Config", "config_label"),
            ("Success", "success_rate"),
            ("Mean evals", "mean_evals_success"),
            ("Mean fitness", "mean_final_fitness"),
            ("Mean cost", "mean_cost"),
            ("Mean gap %", "mean_gap_percent"),
        ],
    )


# ============================================================
# PLOTTING
# ============================================================

def _config_color_map(configs: List[str]):
    cmap = plt.cm.get_cmap("tab20", max(len(configs), 1))
    return {cfg: cmap(i) for i, cfg in enumerate(configs)}


def plot_bitstring_parameter_scaling(summary_rows: List[Dict], output_dir: Path):
    for problem in ("onemax", "leadingones"):
        for algorithm in ("MMAS", "P-ACO"):
            rows = [
                r for r in summary_rows
                if r["problem"] == problem and r["algorithm"] == algorithm
            ]
            if not rows:
                continue

            configs = sorted({r["config_label"] for r in rows})
            colors = _config_color_map(configs)

            fig, ax = plt.subplots(figsize=(12, 7))
            plotted = False
            for config_label in configs:
                data = [r for r in rows if r["config_label"] == config_label]
                xs, means, stds = [], [], []
                for row in sorted(data, key=lambda r: int(r["n"])):
                    if row["mean_evals_success"] == "":
                        continue
                    xs.append(int(row["n"]))
                    means.append(float(row["mean_evals_success"]))
                    stds.append(float(row["std_evals_success"] or 0.0))
                if xs:
                    plotted = True
                    ax.errorbar(
                        xs,
                        means,
                        yerr=stds,
                        label=config_label,
                        marker="o",
                        linewidth=2.0,
                        capsize=4,
                        color=colors[config_label],
                        alpha=0.85,
                    )

            if not plotted:
                plt.close(fig)
                continue

            title_problem = BITSTRING_PROBLEMS[problem]["title"]
            ax.set_title(f"{algorithm} parameter study on {title_problem}", fontsize=14)
            ax.set_xlabel("Bitstring length (n)", fontsize=12)
            ax.set_ylabel("Fitness evaluations to optimum", fontsize=12)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.legend(fontsize=8)
            path = output_dir / f"{problem}_{algorithm.lower().replace('-', '')}_parameter_scaling.png"
            fig.tight_layout()
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {path}")


def plot_best_bitstring_comparison(summary_rows: List[Dict], best_rows: List[Dict], output_dir: Path):
    for problem in ("onemax", "leadingones"):
        best_by_algo = {
            r["algorithm"]: r["config_label"]
            for r in best_rows
            if r["problem"] == problem
        }
        if not best_by_algo:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = {"MMAS": "#1f77b4", "P-ACO": "#d62728"}
        plotted = False
        for algorithm, config_label in best_by_algo.items():
            rows = [
                r for r in summary_rows
                if r["problem"] == problem
                and r["algorithm"] == algorithm
                and r["config_label"] == config_label
            ]
            xs, means, stds = [], [], []
            for row in sorted(rows, key=lambda r: int(r["n"])):
                if row["mean_evals_success"] == "":
                    continue
                xs.append(int(row["n"]))
                means.append(float(row["mean_evals_success"]))
                stds.append(float(row["std_evals_success"] or 0.0))
            if xs:
                plotted = True
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    label=f"{algorithm}: {config_label}",
                    marker="o",
                    linewidth=2.5,
                    capsize=5,
                    color=colors.get(algorithm),
                    alpha=0.85,
                )

        if not plotted:
            plt.close(fig)
            continue

        title_problem = BITSTRING_PROBLEMS[problem]["title"]
        ax.set_title(f"Best MMAS vs P-ACO configurations on {title_problem}", fontsize=14)
        ax.set_xlabel("Bitstring length (n)", fontsize=12)
        ax.set_ylabel("Fitness evaluations to optimum", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=9)
        path = output_dir / f"{problem}_best_aco_comparison.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {path}")


def plot_tsp_parameter_bars(summary_rows: List[Dict], output_dir: Path):
    for algorithm in ("MMAS", "P-ACO"):
        rows = [
            r for r in summary_rows
            if r["problem"] == "tsp" and r["algorithm"] == algorithm
        ]
        if not rows:
            continue

        rows = sorted(rows, key=lambda r: float(r["mean_gap_percent"] or math.inf))
        labels = [r["config_label"] for r in rows]
        means = [float(r["mean_gap_percent"]) for r in rows]
        stds = [float(r["std_gap_percent"] or 0.0) for r in rows]

        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 6))
        xs = np.arange(len(rows))
        ax.bar(xs, means, yerr=stds, capsize=4, color="#1f77b4" if algorithm == "MMAS" else "#d62728")
        ax.set_xticks(xs, labels=labels, rotation=45, ha="right")
        ax.set_ylabel("Mean gap above optimum (%)", fontsize=12)
        ax.set_title(f"{algorithm} parameter study on berlin52", fontsize=14)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
        path = output_dir / f"tsp_{algorithm.lower().replace('-', '')}_parameter_gap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {path}")


def _mean_curve(curves: List[List[float]], num_ants: int, max_points=250):
    if not curves:
        return np.array([]), np.array([]), np.array([])

    series = []
    for curve in curves:
        values = np.asarray(curve, dtype=float)
        evals = np.arange(len(values), dtype=float) * float(num_ants)
        series.append((evals, values))

    max_eval = max(float(e[-1]) for e, _ in series)
    grid = np.linspace(0.0, max_eval, min(max_points, int(max_eval) + 1))
    stacked = []
    for evals, values in series:
        stacked.append(np.interp(grid, evals, values))
    arr = np.asarray(stacked)
    return grid, arr.mean(axis=0), arr.std(axis=0)


def plot_tsp_best_convergence(
    curves: Dict[Tuple[str, str], List[List[float]]],
    config_num_ants: Dict[Tuple[str, str], int],
    best_rows: List[Dict],
    output_dir: Path,
):
    best_tsp = [r for r in best_rows if r["problem"] == "tsp"]
    if not best_tsp:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"MMAS": "#1f77b4", "P-ACO": "#d62728"}
    for row in best_tsp:
        key = (row["algorithm"], row["config_label"])
        x, mean, std = _mean_curve(curves.get(key, []), config_num_ants.get(key, 1))
        if len(x) == 0:
            continue
        ax.plot(x, mean, label=f"{row['algorithm']}: {row['config_label']}", color=colors.get(row["algorithm"]), linewidth=2.5)
        ax.fill_between(x, mean - std, mean + std, color=colors.get(row["algorithm"]), alpha=0.18)

    optimum = tsp.get_optimum(TSP_INSTANCE)
    if optimum:
        ax.axhline(optimum, linestyle="--", color="green", linewidth=2.0, alpha=0.8, label=f"Optimum: {optimum}")

    ax.set_title("Best ACO configurations on berlin52", fontsize=14)
    ax.set_xlabel("Fitness evaluations", fontsize=12)
    ax.set_ylabel("Tour cost", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    path = output_dir / "tsp_best_aco_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def save_curve_points(
    curves: Dict[Tuple[str, str], List[List[float]]],
    config_num_ants: Dict[Tuple[str, str], int],
    output_dir: Path,
):
    rows = []
    for key, group_curves in sorted(curves.items()):
        algorithm, config_label = key
        x, mean, std = _mean_curve(group_curves, config_num_ants[key])
        for evals, mean_cost, std_cost in zip(x, mean, std):
            rows.append({
                "problem": "tsp",
                "instance": TSP_INSTANCE,
                "algorithm": algorithm,
                "config_label": config_label,
                "evaluations": int(round(evals)),
                "mean_cost": float(mean_cost),
                "std_cost": float(std_cost),
            })

    if rows:
        write_csv(
            rows,
            output_dir / "aco_report_tsp_curve_points.csv",
            ["problem", "instance", "algorithm", "config_label", "evaluations", "mean_cost", "std_cost"],
        )


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    seeds: int = DEFAULT_SEEDS,
    tsp_seeds: int = None,
    onemax_sizes: List[int] = None,
    leadingones_sizes: List[int] = None,
    onemax_max_iterations: int = ONEMAX_MAX_ITERATIONS,
    leadingones_max_iterations: int = LEADINGONES_MAX_ITERATIONS,
    tsp_max_iterations: int = TSP_MAX_ITERATIONS,
    mmas_rhos: List[float] = None,
    mmas_num_ants: List[int] = None,
    paco_archive_sizes: List[int] = None,
    tsp_paco_archive_sizes: List[int] = None,
    paco_num_ants: List[int] = None,
    paco_q0_values: List[float] = None,
    tsp_betas: List[float] = None,
    output_dir: Path = OUTPUT_DIR,
    skip_tsp: bool = False,
    preset_name: str = DEFAULT_PRESET,
    dry_run: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    onemax_sizes = onemax_sizes or ONEMAX_SIZES
    leadingones_sizes = leadingones_sizes or LEADINGONES_SIZES
    tsp_seeds = tsp_seeds if tsp_seeds is not None else seeds
    mmas_rhos = mmas_rhos or MMAS_RHOS
    mmas_num_ants = mmas_num_ants or MMAS_NUM_ANTS
    paco_archive_sizes = paco_archive_sizes or PACO_ARCHIVE_SIZES
    tsp_paco_archive_sizes = tsp_paco_archive_sizes or paco_archive_sizes
    paco_num_ants = paco_num_ants or PACO_NUM_ANTS
    paco_q0_values = paco_q0_values or PACO_Q0_VALUES
    tsp_betas = tsp_betas or TSP_BETAS

    mmas_bit_configs = build_mmas_bitstring_configs(mmas_rhos, mmas_num_ants)
    paco_bit_configs = build_paco_bitstring_configs(paco_archive_sizes, paco_num_ants, paco_q0_values)
    bit_configs = mmas_bit_configs + paco_bit_configs

    mmas_tsp_configs = build_mmas_tsp_configs(mmas_rhos, mmas_num_ants, tsp_betas)
    paco_tsp_configs = build_paco_tsp_configs(tsp_paco_archive_sizes, paco_num_ants, paco_q0_values, tsp_betas)
    tsp_configs = mmas_tsp_configs + paco_tsp_configs

    raw_rows = []
    tsp_curves = defaultdict(list)
    tsp_config_num_ants = {}

    print("\n" + "=" * 78)
    print("ACO REPORT EXPERIMENT")
    print("=" * 78)
    print(f"Preset: {preset_name}")
    print(f"Bitstring seeds per configuration: {seeds}")
    print(f"TSP seeds per configuration: {tsp_seeds}")
    print(f"OneMax sizes: {onemax_sizes}")
    print(f"LeadingOnes sizes: {leadingones_sizes}")
    print(f"MMAS configs: {[c['config_label'] for c in mmas_bit_configs]}")
    print(f"P-ACO configs: {[c['config_label'] for c in paco_bit_configs]}")
    if not skip_tsp:
        print(f"TSP instance: {TSP_INSTANCE}; beta values: {tsp_betas}")
        print(f"TSP P-ACO archive sizes: {tsp_paco_archive_sizes}")
    print("=" * 78)

    bit_plan = [
        ("onemax", onemax_sizes, onemax_max_iterations),
        ("leadingones", leadingones_sizes, leadingones_max_iterations),
    ]

    total_bit_runs = sum(len(sizes) * len(bit_configs) * seeds for _problem, sizes, _max_i in bit_plan)
    total_tsp_runs = 0 if skip_tsp else len(tsp_configs) * tsp_seeds
    total_runs = total_bit_runs + total_tsp_runs
    run_count = 0
    print(f"Planned runs: bitstring={total_bit_runs}, tsp={total_tsp_runs}, total={total_runs}")

    if dry_run:
        print("Dry run only; no trials executed.")
        return [], [], []

    for problem_key, sizes, max_iterations in bit_plan:
        title = BITSTRING_PROBLEMS[problem_key]["title"]
        print(f"\n--- {title}: bit-length and parameter study ---")
        for n in sizes:
            for config in bit_configs:
                for seed in range(seeds):
                    run_count += 1
                    row = run_bitstring_trial(problem_key, n, config, seed, max_iterations)
                    raw_rows.append(row)
                    status = "ok" if row["reached_optimum"] else "dnf"
                    print(
                        f"[{run_count:4d}/{total_runs}] {title:11s} n={n:4d} "
                        f"{config['algorithm']:5s} {config['config_label']:24s} "
                        f"seed={seed:2d}: evals={int(row['fitness_evaluations']):8d} {status}"
                    )

    if not skip_tsp:
        print(f"\n--- TSP: {TSP_INSTANCE} parameter study ---")
        try:
            instance_name, _problem, coords, _nodes, dm = tsp.fetch_tsp_instance(TSP_INSTANCE)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load {TSP_INSTANCE}; skipping TSP ({exc})")
        else:
            for config in tsp_configs:
                key = (config["algorithm"], config["config_label"])
                tsp_config_num_ants[key] = config["num_ants"]
                for seed in range(tsp_seeds):
                    run_count += 1
                    row, history = run_tsp_trial(instance_name, dm, coords, config, seed, tsp_max_iterations)
                    raw_rows.append(row)
                    tsp_curves[key].append(history)
                    gap = _fmt(row["gap_percent"])
                    print(
                        f"[{run_count:4d}/{total_runs}] {instance_name:9s} "
                        f"{config['algorithm']:5s} {config['config_label']:34s} "
                        f"seed={seed:2d}: cost={float(row['best_cost']):8.1f}, gap={gap}%"
                    )

    summary_rows = aggregate_rows(raw_rows)
    best_rows = select_best_configs(summary_rows)

    save_outputs(raw_rows, summary_rows, best_rows, output_dir)
    plot_bitstring_parameter_scaling(summary_rows, output_dir)
    plot_best_bitstring_comparison(summary_rows, best_rows, output_dir)
    plot_tsp_parameter_bars(summary_rows, output_dir)
    plot_tsp_best_convergence(tsp_curves, tsp_config_num_ants, best_rows, output_dir)
    save_curve_points(tsp_curves, tsp_config_num_ants, output_dir)

    print("\n" + "=" * 78)
    print("BEST CONFIGURATIONS")
    print("=" * 78)
    for row in best_rows:
        print(
            f"{row['problem']:12s} | {row['algorithm']:5s} | "
            f"{row['config_label']:34s} | {row['selection_reason']}"
        )

    print("\n" + "=" * 78)
    print(f"Finished. Output directory: {output_dir}")
    print("=" * 78)

    return raw_rows, summary_rows, best_rows


def _copy_preset_value(value):
    return list(value) if isinstance(value, list) else value


def _arg_or_preset(args, preset_name: str, key: str):
    value = getattr(args, key)
    if value is None:
        value = EXPERIMENT_PRESETS[preset_name][key]
    return _copy_preset_value(value)


def main():
    parser = argparse.ArgumentParser(description="Full MMAS/P-ACO report experiment")
    parser.add_argument(
        "--preset",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        default=DEFAULT_PRESET,
        help=(
            "quick = fast verification, report = balanced default, "
            "full = old large grid"
        ),
    )
    parser.add_argument("--full", action="store_true", help="Shortcut for --preset full")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned grid and exit")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--tsp-seeds", type=int, default=None)
    parser.add_argument("--onemax-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--leadingones-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--onemax-max-iterations", type=int, default=None)
    parser.add_argument("--leadingones-max-iterations", type=int, default=None)
    parser.add_argument("--tsp-max-iterations", type=int, default=None)
    parser.add_argument("--mmas-rhos", nargs="+", type=float, default=None)
    parser.add_argument("--mmas-num-ants", nargs="+", type=int, default=None)
    parser.add_argument("--paco-archive-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--tsp-paco-archive-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--paco-num-ants", nargs="+", type=int, default=None)
    parser.add_argument("--paco-q0-values", nargs="+", type=float, default=None)
    parser.add_argument("--tsp-betas", nargs="+", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-tsp", action="store_true")
    args = parser.parse_args()

    preset_name = "full" if args.full else args.preset

    run_experiment(
        seeds=_arg_or_preset(args, preset_name, "seeds"),
        tsp_seeds=_arg_or_preset(args, preset_name, "tsp_seeds"),
        onemax_sizes=_arg_or_preset(args, preset_name, "onemax_sizes"),
        leadingones_sizes=_arg_or_preset(args, preset_name, "leadingones_sizes"),
        onemax_max_iterations=_arg_or_preset(args, preset_name, "onemax_max_iterations"),
        leadingones_max_iterations=_arg_or_preset(args, preset_name, "leadingones_max_iterations"),
        tsp_max_iterations=_arg_or_preset(args, preset_name, "tsp_max_iterations"),
        mmas_rhos=_arg_or_preset(args, preset_name, "mmas_rhos"),
        mmas_num_ants=_arg_or_preset(args, preset_name, "mmas_num_ants"),
        paco_archive_sizes=_arg_or_preset(args, preset_name, "paco_archive_sizes"),
        tsp_paco_archive_sizes=_arg_or_preset(args, preset_name, "tsp_paco_archive_sizes"),
        paco_num_ants=_arg_or_preset(args, preset_name, "paco_num_ants"),
        paco_q0_values=_arg_or_preset(args, preset_name, "paco_q0_values"),
        tsp_betas=_arg_or_preset(args, preset_name, "tsp_betas"),
        output_dir=args.output_dir,
        skip_tsp=args.skip_tsp,
        preset_name=preset_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
