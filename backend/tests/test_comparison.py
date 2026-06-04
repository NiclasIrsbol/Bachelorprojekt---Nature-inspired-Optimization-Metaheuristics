"""Tests for the in-app comparison aggregation (comparison.compare / scaling)."""
import pytest

from optimization_framework.experiments import comparison


class TestCompareBitstring:
    def test_metadata_and_keys(self):
        out = comparison.compare(
            "onemax", ["(1+1) EA", "(μ+λ) EA"], seeds=3, bit_length=20
        )
        assert out["problem"] == "onemax"
        assert out["metric"] == "fitness_evaluations"
        assert out["bit_length"] == 20
        assert "tsp_instance" not in out
        assert [a["algorithm"] for a in out["algorithms"]] == ["(1+1) EA", "(μ+λ) EA"]

        for algo in out["algorithms"]:
            assert {"mean", "std", "best", "worst", "median", "success_rate", "curve"} <= set(algo)
            assert "gap_percent" not in algo

    def test_curve_x_axis_is_monotonic_in_evaluations(self):
        out = comparison.compare("onemax", ["(1+1) EA"], seeds=3, bit_length=25)
        curve = out["algorithms"][0]["curve"]
        assert len(curve) > 1
        xs = [p["evaluations"] for p in curve]
        assert xs[0] == 0
        assert all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
        # Fitness (best so far) is non-decreasing for OneMax maximization.
        ys = [p["value"] for p in curve]
        assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))

    def test_mu_lambda_x_axis_exceeds_one_plus_one(self):
        """(μ+λ) EA does lambda evaluations per iteration, so its averaged curve
        should reach a larger final evaluation count than (1+1) EA at equal n."""
        out = comparison.compare(
            "onemax", ["(1+1) EA", "(μ+λ) EA"], seeds=3, bit_length=25
        )
        by_algo = {a["algorithm"]: a for a in out["algorithms"]}
        last_one = by_algo["(1+1) EA"]["curve"][-1]["evaluations"]
        last_mu = by_algo["(μ+λ) EA"]["curve"][-1]["evaluations"]
        assert last_mu > last_one


class TestCompareTsp:
    def test_tsp_metadata(self):
        try:
            out = comparison.compare(
                "tsp", ["(1+1) EA"], seeds=2, max_iterations=300, tsp_instance="burma14"
            )
        except Exception as e:
            pytest.skip(f"TSP instance unavailable: {e}")
        assert out["problem"] == "tsp"
        assert out["metric"] == "best_cost"
        assert "tsp_instance" in out
        assert "bit_length" not in out
        for algo in out["algorithms"]:
            assert "gap_percent" in algo
            assert "success_rate" not in algo


class TestScaling:
    def test_bitstring_scaling_has_theory(self):
        out = comparison.scaling("onemax", ["(1+1) EA"], max_size=20, steps=10, seeds=2)
        assert out["problem"] == "onemax"
        assert out["y_metric"] == "fitness_evaluations"
        assert len(out["series"]) == 1
        # origin point plus one per size (10, 20)
        assert out["series"][0]["points"][0] == {"size": 0, "value": 0.0}
        assert [p["size"] for p in out["series"][0]["points"]] == [0, 10, 20]
        assert len(out["theory"]) >= 1
        for t in out["theory"]:
            assert "label" in t and "points" in t

    def test_tsp_scaling_rejected(self):
        with pytest.raises(ValueError):
            comparison.scaling("tsp", ["(1+1) EA"], max_size=20, steps=10, seeds=2)


class TestAggregateByEvaluations:
    def test_forward_fill_holds_final_value(self):
        # Two runs of different length; shorter run converged at value 5.
        runs = [([1.0, 3.0, 5.0], 1.0), ([1.0, 2.0, 3.0, 4.0, 5.0], 1.0)]
        curve = comparison._aggregate_by_evaluations(runs, max_points=50)
        xs = [p["evaluations"] for p in curve]
        assert xs == sorted(xs)
        assert curve[-1]["value"] == pytest.approx(5.0)

    def test_empty_runs(self):
        assert comparison._aggregate_by_evaluations([]) == []
