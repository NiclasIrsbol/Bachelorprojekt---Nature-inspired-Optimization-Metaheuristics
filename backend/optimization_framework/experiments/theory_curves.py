# @author: Andrej Kitanovski
"""Exact expected-runtime curves for thesis scaling overlays."""

import math

E = math.e


def onemax_ea_expected_evals(n: float) -> float:
    """Expected fitness evaluations for (1+1) EA on OneMax: e·n·ln n − 1.89n."""
    if n < 2:
        return 0.0
    return E * n * math.log(n) - 1.89 * n


def leadingones_expected_evals(n: float) -> float:
    """Expected fitness evaluations for (1+1) EA on LeadingOnes: ≈ 0.859 n²."""
    return 0.859 * n * n


def rls_expected_evals(n: float) -> float:
    """RLS / SA with T₀=0 (single-bit flips, accept-if-not-worse): ≈ n ln n."""
    if n < 2:
        return 0.0
    return n * math.log(n)


# (label, function) pairs keyed by problem name
CURVES_BY_PROBLEM = {
    "onemax": [
        ("(1+1) EA theory", onemax_ea_expected_evals),
        ("RLS / SA (T₀=0)", rls_expected_evals),
    ],
    "leadingones": [
        ("LeadingOnes theory", leadingones_expected_evals),
    ],
}


def theory_series(problem: str, sizes: list[int]) -> list[dict]:
    """Build theory overlay series for API / plots."""
    curves = CURVES_BY_PROBLEM.get(problem, [])
    out = []
    for label, fn in curves:
        points = [{"size": 0, "value": 0.0}]
        for n in sizes:
            points.append({"size": n, "value": float(fn(n))})
        out.append({"label": label, "points": points})
    return out
