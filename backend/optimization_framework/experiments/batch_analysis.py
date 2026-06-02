"""Statistical analysis and plotting for batch experiment results.

Reads ``raw_results.csv`` and ``curves.json`` produced by ``batch_runner.py``
and writes, under the same output directory:

  - ``summary.csv``            aggregate stats (mean/std/best/worst/median +
                               success rate) per (group, algorithm)
  - ``wilcoxon_<group>.csv``   pairwise Wilcoxon rank-sum p-values
  - ``boxplot_<group>.png``    distribution of the primary metric per algorithm
  - ``convergence_<group>.png``mean convergence curve per algorithm (+/- std)

Primary metric:
  - bitstring problems: fitness evaluations to optimum (lower is better)
  - TSP: best tour cost (lower is better)

Groups: bitstring problems are grouped by problem name; TSP runs are grouped
per instance (``tsp_<instance>``) because cost scales differ between instances.

Usage:
    cd backend
    python -m optimization_framework.experiments.batch_analysis
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DEFAULT_OUTPUT_DIR = "output/batch"
RAW_RESULTS_FILE = "raw_results.csv"
CURVES_FILE = "curves.json"
SUMMARY_FILE = "summary.csv"

SUMMARY_FIELDNAMES = [
    "group",
    "problem",
    "instance",
    "algorithm",
    "metric",
    "count",
    "mean",
    "std",
    "best",
    "worst",
    "median",
    "success_rate",
]


def _metric_column(problem):
    """Primary comparison metric column for a problem (lower is better)."""
    return "fitness_evaluations" if problem != "tsp" else "best_value"


def _group_label(problem, instance):
    return f"tsp_{instance}" if problem == "tsp" else problem


def _load_rows(output_dir):
    raw_path = Path(output_dir) / RAW_RESULTS_FILE
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run batch_runner.py first."
        )

    rows = []
    with open(raw_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "problem": r["problem"],
                    "algorithm": r["algorithm"],
                    "instance": r["instance"],
                    "seed": int(r["seed"]),
                    "best_value": float(r["best_value"]),
                    "reached_optimum": r["reached_optimum"] == "True",
                    "has_optimum_flag": r["reached_optimum"] != "",
                    "iterations": float(r["iterations"]),
                    "fitness_evaluations": float(r["fitness_evaluations"]),
                    "runtime_seconds": float(r["runtime_seconds"]),
                }
            )
    return rows


def _load_curves(output_dir):
    curves_path = Path(output_dir) / CURVES_FILE
    if not curves_path.exists():
        return []
    with open(curves_path) as f:
        return json.load(f)


def _grouped(rows):
    """group_label -> (problem, instance, {algorithm: [rows...]})."""
    groups = defaultdict(lambda: {"problem": None, "instance": "", "algos": defaultdict(list)})
    for r in rows:
        label = _group_label(r["problem"], r["instance"])
        g = groups[label]
        g["problem"] = r["problem"]
        g["instance"] = r["instance"]
        g["algos"][r["algorithm"]].append(r)
    return groups


def _write_summary(groups, output_dir):
    out_path = Path(output_dir) / SUMMARY_FILE
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()

        for label, g in sorted(groups.items()):
            metric = _metric_column(g["problem"])
            for algo, algo_rows in sorted(g["algos"].items()):
                values = np.array([r[metric] for r in algo_rows], dtype=float)
                has_flag = any(r["has_optimum_flag"] for r in algo_rows)
                if has_flag:
                    n_opt = sum(1 for r in algo_rows if r["reached_optimum"])
                    success_rate = n_opt / len(algo_rows)
                else:
                    success_rate = ""

                writer.writerow(
                    {
                        "group": label,
                        "problem": g["problem"],
                        "instance": g["instance"],
                        "algorithm": algo,
                        "metric": metric,
                        "count": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "best": float(np.min(values)),
                        "worst": float(np.max(values)),
                        "median": float(np.median(values)),
                        "success_rate": success_rate,
                    }
                )
    print(f"Wrote summary to {out_path}")


def _write_wilcoxon(groups, output_dir):
    for label, g in sorted(groups.items()):
        metric = _metric_column(g["problem"])
        algos = sorted(g["algos"].keys())
        data = {a: np.array([r[metric] for r in g["algos"][a]], dtype=float) for a in algos}

        out_path = Path(output_dir) / f"wilcoxon_{label}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"Wilcoxon rank-sum p-values ({metric})"] + algos)
            for a in algos:
                row = [a]
                for b in algos:
                    if a == b:
                        row.append("")
                        continue
                    try:
                        _stat, pval = stats.ranksums(data[a], data[b])
                        row.append(f"{pval:.4g}")
                    except Exception:
                        row.append("nan")
                writer.writerow(row)
        print(f"Wrote Wilcoxon table to {out_path}")


def _plot_boxplots(groups, output_dir):
    for label, g in sorted(groups.items()):
        metric = _metric_column(g["problem"])
        algos = sorted(g["algos"].keys())
        data = [np.array([r[metric] for r in g["algos"][a]], dtype=float) for a in algos]

        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(algos)), 5))
        ax.boxplot(data, labels=algos, showmeans=True)
        ax.set_title(f"{label}: distribution of {metric}")
        ax.set_ylabel(f"{metric} (lower is better)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        out_path = Path(output_dir) / f"boxplot_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote box plot to {out_path}")


def _pad_curves(curve_list):
    """Pad each curve to the max length using its final value, then stack."""
    max_len = max(len(c) for c in curve_list)
    padded = []
    for c in curve_list:
        if len(c) < max_len:
            c = list(c) + [c[-1]] * (max_len - len(c))
        padded.append(c)
    return np.array(padded, dtype=float)


def _plot_convergence(groups, curves, output_dir):
    # index curves by (label, algorithm) -> list of curves
    by_group_algo = defaultdict(lambda: defaultdict(list))
    for entry in curves:
        if not entry.get("curve"):
            continue
        label = _group_label(entry["problem"], entry.get("instance", ""))
        by_group_algo[label][entry["algorithm"]].append(entry["curve"])

    for label, g in sorted(groups.items()):
        algo_curves = by_group_algo.get(label)
        if not algo_curves:
            continue
        metric = _metric_column(g["problem"])
        ylabel = "best fitness" if g["problem"] != "tsp" else "best cost"

        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in sorted(algo_curves.keys()):
            stacked = _pad_curves(algo_curves[algo])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
            x = np.arange(len(mean))
            ax.plot(x, mean, label=algo)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)

        ax.set_title(f"{label}: mean convergence (+/- std)")
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        out_path = Path(output_dir) / f"convergence_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote convergence plot to {out_path} (metric for stats: {metric})")


def analyze(output_dir=DEFAULT_OUTPUT_DIR):
    rows = _load_rows(output_dir)
    curves = _load_curves(output_dir)
    groups = _grouped(rows)

    _write_summary(groups, output_dir)
    _write_wilcoxon(groups, output_dir)
    _plot_boxplots(groups, output_dir)
    _plot_convergence(groups, curves, output_dir)


def _parse_args():
    parser = argparse.ArgumentParser(description="Analyze batch experiment results.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing batch results (and where plots are written).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    analyze(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
