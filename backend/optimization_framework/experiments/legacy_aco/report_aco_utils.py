"""Shared helpers for the report-structured ACO experiments (sections 8.4, 8.5).

Used by the per-problem MMAS/P-ACO scaling scripts and the comparison script to
avoid duplicating seeding, aggregation, and file-writing logic.
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def aggregate_evals(run_results: List[Dict]) -> Dict:
    """Aggregate a list of trial dicts into mean/std/min/max fitness evaluations.

    The mean is taken over runs that reached the optimum only (consistent with
    the other scaling scripts); success rate is reported separately.
    """
    evals = [r["fitness_evaluations"] for r in run_results if r["reached_optimum"]]
    success = sum(1 for r in run_results if r["reached_optimum"])
    if evals:
        return {
            "mean_evals": float(np.mean(evals)),
            "std_evals": float(np.std(evals)),
            "min_evals": float(np.min(evals)),
            "max_evals": float(np.max(evals)),
            "success_rate": success / len(run_results),
            "trials": len(run_results),
        }
    return {
        "mean_evals": float("inf"),
        "std_evals": 0.0,
        "min_evals": float("inf"),
        "max_evals": 0.0,
        "success_rate": 0.0,
        "trials": len(run_results),
    }


def write_csv(rows: List[Dict], path: Path, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved raw results to {path}")


def write_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved aggregated results to {path}")


def best_config_by_mean_evals(aggregated: Dict):
    """Given aggregated[param][n] -> stats, return the param value with the lowest
    mean evaluations at the largest size where it is finite (tie-break: lowest
    average finite mean across sizes). Returns None if nothing finite."""
    best_param, best_key = None, None
    for param, by_n in aggregated.items():
        finite = [(n, s["mean_evals"]) for n, s in by_n.items() if np.isfinite(s["mean_evals"])]
        if not finite:
            continue
        largest_n = max(n for n, _ in finite)
        val_at_largest = dict(finite)[largest_n]
        avg = float(np.mean([v for _, v in finite]))
        key = (val_at_largest, avg)
        if best_key is None or key < best_key:
            best_key, best_param = key, param
    return best_param
