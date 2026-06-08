"""
ACO comparison experiment (for evaluation): best-rho MMAS vs best-K P-ACO.

This is the dedicated head-to-head comparison used when evaluating the two ACO
variants. For each problem it first sweeps MMAS over rho and P-ACO over archive
size K, automatically selects the best-performing configuration of each variant,
and then plots those two best configurations against each other:

  - OneMax / LeadingOnes : best MMAS(rho*) vs best P-ACO(K*), average fitness
    evaluations to the optimum vs problem size n. "Best" = lowest mean
    evaluations (report_aco_utils.best_config_by_mean_evals).
  - berlin52 (TSP)       : best MMAS(rho*) vs best P-ACO(K*) by mean gap %,
    tour-length convergence vs evaluations, with the optimum line.

It reuses the per-problem sweep scripts so the comparison always reflects the
same implementations.

Defaults are modest so the comparison runs quickly; scale up with the CLI flags
for final evaluation figures.

Output (under output/report_aco):
  - PNG : comparison_onemax.png, comparison_leadingones.png, comparison_tsp_berlin52.png
  - JSON: comparison_summary.json

Usage:
    cd backend
    python3 -m optimization_framework.experiments.legacy_aco.aco_comparison
    python3 -m optimization_framework.experiments.legacy_aco.aco_comparison \
        --bitstring-sizes 100 200 300 --seeds 5
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimization_framework.experiments.legacy_aco import (
    mmas_onemax_rho,
    mmas_leadingones_rho,
    paco_onemax_archive,
    paco_leadingones_archive,
    mmas_tsp_rho_berlin52,
    paco_tsp_archive_berlin52,
)
from optimization_framework.experiments.legacy_aco.report_aco_utils import (
    best_config_by_mean_evals, write_json,
)
from optimization_framework.problems import tsp

OUTPUT_DIR = Path("output/report_aco")
MMAS_COLOR = "#3b82f6"
PACO_COLOR = "#ef4444"

# Modest comparison defaults (override via CLI).
DEFAULT_BITSTRING_SIZES = [100, 200, 300, 400, 500]
DEFAULT_SEEDS = 5
DEFAULT_BITSTRING_MAX_ITERATIONS = 100000
DEFAULT_TSP_MAX_ITERATIONS = 2000


def _best_tsp_by_gap(aggregated):
    """Return the param (rho or K) with the lowest mean gap %."""
    best_param, best_gap = None, None
    for param, stats in aggregated.items():
        gap = stats.get("mean_gap")
        if gap is None:
            continue
        if best_gap is None or gap < best_gap:
            best_gap, best_param = gap, param
    return best_param


def _plot_bitstring(problem, mmas_agg, mmas_rho, paco_agg, paco_k, out_path):
    fig, ax = plt.subplots(figsize=(12, 7))

    for agg, param, color, label in (
        (mmas_agg, mmas_rho, MMAS_COLOR, f"MMAS (ρ = {mmas_rho})"),
        (paco_agg, paco_k, PACO_COLOR, f"P-ACO (K = {paco_k})"),
    ):
        if param is None or param not in agg:
            continue
        xs, means, stds = [], [], []
        for n in sorted(agg[param].keys()):
            m = agg[param][n]["mean_evals"]
            if np.isfinite(m):
                xs.append(n); means.append(m); stds.append(agg[param][n]["std_evals"])
        if xs:
            ax.errorbar(xs, means, yerr=stds, label=label, marker="o", markersize=8,
                        color=color, linewidth=2.5, capsize=5, alpha=0.85)

    title_problem = "OneMax" if problem == "onemax" else "LeadingOnes"
    ax.set_xlabel("Bitstring Length (n)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Evaluations to Optimum", fontsize=12, fontweight="bold")
    ax.set_title(f"MMAS vs P-ACO on {title_problem} (best configurations)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved plot to {out_path}")


def _mean_curve(curve_list, scale_x=1):
    if not curve_list:
        return np.array([]), np.array([])
    max_len = max(len(c) for c in curve_list)
    padded = []
    for c in curve_list:
        if len(c) < max_len:
            c = list(c) + [c[-1]] * (max_len - len(c))
        padded.append(c)
    mean = np.array(padded, dtype=float).mean(axis=0)
    x = np.arange(len(mean)) * scale_x
    return x, mean


def run_bitstring_comparison(problem, mmas_mod, paco_mod, sizes, seeds, max_iterations):
    print(f"\n=== Comparison: {problem} ===")
    _, mmas_agg = mmas_mod.run_experiment(seeds=seeds, sizes=sizes, max_iterations=max_iterations)
    _, paco_agg = paco_mod.run_experiment(seeds=seeds, sizes=sizes, max_iterations=max_iterations)

    best_rho = best_config_by_mean_evals(mmas_agg)
    best_k = best_config_by_mean_evals(paco_agg)
    print(f"  Best MMAS rho = {best_rho}; best P-ACO K = {best_k}")

    out = OUTPUT_DIR / f"comparison_{problem}.png"
    _plot_bitstring(problem, mmas_agg, best_rho, paco_agg, best_k, out)

    return {
        "problem": problem,
        "best_mmas_rho": best_rho,
        "best_paco_archive": best_k,
        "mmas_summary": mmas_agg.get(best_rho, {}),
        "paco_summary": paco_agg.get(best_k, {}),
    }


def run_tsp_comparison(seeds, max_iterations, num_ants):
    print("\n=== Comparison: TSP berlin52 ===")
    mmas_results, mmas_agg, mmas_curves = mmas_tsp_rho_berlin52.run_experiment(
        seeds=seeds, num_ants=num_ants, max_iterations=max_iterations)
    paco_results, paco_agg, paco_curves = paco_tsp_archive_berlin52.run_experiment(
        seeds=seeds, num_ants=num_ants, max_iterations=max_iterations)

    if not mmas_results or not paco_results:
        print("  TSP comparison skipped (berlin52 unavailable).")
        return None

    best_rho = _best_tsp_by_gap(mmas_agg)
    best_k = _best_tsp_by_gap(paco_agg)
    optimum = tsp.get_optimum("berlin52")
    print(f"  Best MMAS rho = {best_rho}; best P-ACO K = {best_k}")

    fig, ax = plt.subplots(figsize=(12, 7))
    if best_rho is not None:
        x, mean = _mean_curve(mmas_curves[best_rho], scale_x=num_ants)
        ax.plot(x, mean, label=f"MMAS (ρ = {best_rho})", color=MMAS_COLOR, linewidth=2.5, alpha=0.85)
    if best_k is not None:
        x, mean = _mean_curve(paco_curves[best_k], scale_x=num_ants)
        ax.plot(x, mean, label=f"P-ACO (K = {best_k})", color=PACO_COLOR, linewidth=2.5, alpha=0.85)
    if optimum is not None:
        ax.axhline(optimum, linestyle="--", linewidth=2.0, color="#64748b", alpha=0.8,
                   label=f"Optimum = {optimum}")

    ax.set_xlabel("Evaluations", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tour Length", fontsize=12, fontweight="bold")
    ax.set_title("MMAS vs P-ACO on berlin52 (best configurations)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right")
    out = OUTPUT_DIR / "comparison_tsp_berlin52.png"
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved plot to {out}")

    return {
        "problem": "tsp_berlin52",
        "best_mmas_rho": best_rho,
        "best_paco_archive": best_k,
        "mmas_summary": mmas_agg.get(best_rho, {}),
        "paco_summary": paco_agg.get(best_k, {}),
    }


def main():
    p = argparse.ArgumentParser(description="MMAS vs P-ACO comparison (best rho vs best K)")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--bitstring-sizes", nargs="+", type=int, default=None)
    p.add_argument("--bitstring-max-iterations", type=int, default=DEFAULT_BITSTRING_MAX_ITERATIONS)
    p.add_argument("--tsp-max-iterations", type=int, default=DEFAULT_TSP_MAX_ITERATIONS)
    p.add_argument("--tsp-num-ants", type=int, default=paco_tsp_archive_berlin52.NUM_ANTS)
    p.add_argument("--skip-tsp", action="store_true")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sizes = args.bitstring_sizes if args.bitstring_sizes else DEFAULT_BITSTRING_SIZES

    summary = []
    summary.append(run_bitstring_comparison(
        "onemax", mmas_onemax_rho, paco_onemax_archive, sizes, args.seeds, args.bitstring_max_iterations))
    summary.append(run_bitstring_comparison(
        "leadingones", mmas_leadingones_rho, paco_leadingones_archive, sizes, args.seeds, args.bitstring_max_iterations))

    if not args.skip_tsp:
        tsp_entry = run_tsp_comparison(args.seeds, args.tsp_max_iterations, args.tsp_num_ants)
        if tsp_entry:
            summary.append(tsp_entry)

    write_json(summary, OUTPUT_DIR / "comparison_summary.json")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (best configurations)")
    print("=" * 70)
    for entry in summary:
        print(f"  {entry['problem']:16s} | best MMAS ρ = {entry['best_mmas_rho']} | "
              f"best P-ACO K = {entry['best_paco_archive']}")


if __name__ == "__main__":
    main()
