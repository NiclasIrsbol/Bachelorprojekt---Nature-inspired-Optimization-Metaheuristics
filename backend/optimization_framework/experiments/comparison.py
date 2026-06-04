"""In-memory aggregation for the in-app comparison feature.

Reuses the seeded single-run helpers from ``batch_runner`` to produce, without
writing any files:
  - ``compare``: per-algorithm statistics + averaged convergence curve for one
    problem at a fixed size (convergence comparison).
  - ``scaling``: averaged metric vs. problem size, one series per algorithm
    (performance/scaling comparison; bitstring only).

These are consumed by the FastAPI /compare and /scaling endpoints.
"""

import numpy as np

from optimization_framework.experiments.batch_runner import _run_bitstring, _run_tsp
from optimization_framework.experiments.theory_curves import theory_series
from optimization_framework.problems import tsp as tsp_problem


def _aggregate_by_evaluations(runs, max_points=200):
    """Average several convergence runs on a shared fitness-evaluation x-axis.

    ``runs`` is a list of ``(curve, evals_per_iter)`` pairs, where ``curve`` is
    the best-fitness-over-time list (one entry per iteration, index 0 being the
    initial point) and ``evals_per_iter`` is that run's fitness evaluations
    divided by its iteration count.

    Each run's curve index ``i`` is mapped to ``i * evals_per_iter`` actual
    fitness evaluations - this is exact for algorithms with a constant number of
    evaluations per iteration ((1+1) EA = 1, (mu+lambda) EA = lambda, ACO/P-ACO
    = ants per iteration). Runs are then forward-filled onto a shared evaluation
    grid (a finished run holds its final, converged value) and averaged. This is
    fairer than averaging by iteration index, because algorithms that spend many
    evaluations per iteration are placed correctly on the x-axis.
    """
    series = []
    for curve, epi in runs:
        if not curve:
            continue
        values = np.asarray(curve, dtype=float)
        evals = np.arange(len(values), dtype=float) * float(epi)
        series.append((evals, values))

    if not series:
        return []

    max_eval = max(float(evals[-1]) for evals, _ in series)
    if max_eval <= 0:
        # All runs converged at evaluation 0 (degenerate); report a single point.
        mean_value = float(np.mean([values[-1] for _, values in series]))
        return [{"evaluations": 0, "value": mean_value}]

    grid = np.linspace(0.0, max_eval, max_points)

    stacked = []
    for evals, values in series:
        # Forward-fill: value at grid x is the best fitness reached by x evals.
        # searchsorted(..., "right") - 1 gives the index of the last eval <= x.
        idx = np.searchsorted(evals, grid, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        stacked.append(values[idx])

    mean_curve = np.mean(np.vstack(stacked), axis=0)
    return [
        {"evaluations": int(round(x)), "value": float(v)}
        for x, v in zip(grid, mean_curve)
    ]


def _stats(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "best": float(arr.min()),
        "worst": float(arr.max()),
        "median": float(np.median(arr)),
    }


def compare(problem, algorithms, seeds, bit_length=50, max_iterations=100000,
            tsp_instance=None):
    """Convergence comparison: run each algorithm `seeds` times at a fixed size
    and return per-algorithm stats + averaged convergence curve."""
    seed_list = list(range(seeds))
    extra = {"max_iterations": max_iterations} if max_iterations else None

    tsp_data = None
    if problem == "tsp":
        name, _problem, coords, _nodes, dist = tsp_problem.fetch_tsp_instance(tsp_instance)
        tsp_data = (name, coords, dist)

    result_algos = []
    for algo in algorithms:
        metric_vals, runs, reached = [], [], []
        for seed in seed_list:
            if problem == "tsp":
                name, coords, dist = tsp_data
                tsp_params = {"max_iterations": max_iterations} if max_iterations else {}
                row, curve = _run_tsp(algo, name, dist, coords, tsp_params, seed)
                metric_vals.append(row["best_value"])
            else:
                row, curve = _run_bitstring(problem, algo, bit_length, seed, extra_params=extra)
                metric_vals.append(row["fitness_evaluations"])
                reached.append(bool(row["reached_optimum"]))
            epi = row["fitness_evaluations"] / max(row["iterations"], 1)
            runs.append((curve, epi))

        entry = {"algorithm": algo, **_stats(metric_vals),
                 "curve": _aggregate_by_evaluations(runs)}
        if problem == "tsp":
            gap = tsp_problem.gap_percent(entry["mean"], tsp_data[0])
            entry["gap_percent"] = None if gap is None else round(gap, 3)
        else:
            entry["success_rate"] = sum(reached) / len(reached)
        result_algos.append(entry)

    out = {
        "problem": problem,
        "metric": "best_cost" if problem == "tsp" else "fitness_evaluations",
        "algorithms": result_algos,
    }
    if problem == "tsp":
        out["tsp_instance"] = tsp_data[0]
        out["optimum"] = tsp_problem.get_optimum(tsp_data[0])
    else:
        out["bit_length"] = bit_length
    return out


def scaling(problem, algorithms, max_size, steps, seeds,
            max_iterations=150000, y_metric="fitness_evaluations"):
    """Scaling comparison (bitstring only): averaged metric vs. problem size."""
    if problem == "tsp":
        raise ValueError("scaling comparison is only supported for bitstring problems")

    sizes = list(range(steps, max_size + 1, steps))
    seed_list = list(range(seeds))
    extra = {"max_iterations": max_iterations} if max_iterations else None

    series = []
    for algo in algorithms:
        points = [{"size": 0, "value": 0.0}]  # origin
        for size in sizes:
            vals = []
            for seed in seed_list:
                row, _curve = _run_bitstring(problem, algo, size, seed, extra_params=extra)
                vals.append(row[y_metric])
            points.append({"size": size, "value": float(np.mean(vals))})
        series.append({"algorithm": algo, "points": points})

    return {
        "problem": problem,
        "y_metric": y_metric,
        "series": series,
        "theory": theory_series(problem, sizes),
    }
