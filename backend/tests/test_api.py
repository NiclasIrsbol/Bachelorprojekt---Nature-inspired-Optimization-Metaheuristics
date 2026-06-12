# @author: Andrej Kitanovski

"""Endpoint tests for the FastAPI layer using Starlette's TestClient.

Report §5.1.2 / §5.3.3 state the API can be tested separately from the frontend;
these cover the main endpoints without a running server. Skipped automatically if
``httpx`` (TestClient's dependency) is not installed.
"""

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from optimization_framework.api import app  # noqa: E402

client = TestClient(app)


def test_run_onemax():
    resp = client.post("/run", json={
        "problem": "onemax",
        "algorithm": "(1+1) EA",
        "params": {"bit_length": 12, "max_iterations": 3000},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["problem"] == "onemax"
    assert body["fitness_evaluations"] >= 1


def test_compare_returns_per_algorithm_stats():
    resp = client.post("/compare", json={
        "problem": "onemax",
        "algorithms": ["(1+1) EA", "P-ACO"],
        "seeds": 2,
        "bit_length": 10,
        "max_iterations": 3000,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert {a["algorithm"] for a in body["algorithms"]} == {"(1+1) EA", "P-ACO"}
    assert all("curve" in a and "success_rate" in a for a in body["algorithms"])


def test_scaling_includes_theory():
    resp = client.post("/scaling", json={
        "problem": "onemax",
        "algorithms": ["(1+1) EA"],
        "max_size": 20,
        "steps": 10,
        "seeds": 1,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["series"] and body["theory"]


def test_scaling_rejects_tsp():
    resp = client.post("/scaling", json={"problem": "tsp"})
    # comparison.scaling raises; the endpoint maps it to an {"error": ...} body.
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_export_csv_after_a_run():
    client.post("/run", json={
        "problem": "onemax", "algorithm": "(1+1) EA",
        "params": {"bit_length": 10, "max_iterations": 2000},
    })
    resp = client.get("/export-csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_tsp_instances_listing():
    resp = client.get("/tsp-instances")
    assert resp.status_code == 200
    body = resp.json()
    if "error" in body:
        pytest.skip(f"TSP instances unavailable: {body['error']}")
    assert isinstance(body["instances"], list) and body["instances"]
    assert {"name", "num_cities"} <= set(body["instances"][0])
