from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from optimization_framework.experiments.run_experiment import main
from optimization_framework.experiments import comparison
from optimization_framework.problems import tsp
import json
import csv
import io
from datetime import datetime

DEFAULT_ALGORITHMS = ["(1+1) EA", "(μ+λ) EA", "Simulated Annealing", "ACO", "P-ACO"]

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_FILE = Path("output/latest_run.json")

class RunRequest(BaseModel):
    problem: str = "onemax"
    algorithm: str = "(μ+λ) EA"
    params: dict = {}
    tsp_instance: str = None  # Optional: specific TSP instance name

@app.get("/latest-run")
def get_latest_run():
    if not OUTPUT_FILE.exists():
        return {"error": "No experiment run yet"}

    return FileResponse(OUTPUT_FILE)

@app.get("/tsp-instances")
def get_tsp_instances():
    """Get list of available TSP instances with metadata."""
    try:
        metadata = tsp.get_tsp_instances_metadata()
        return {"instances": metadata}
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/run")
def run_experiment(request: RunRequest):
    result = main(
        problem_name=request.problem,
        algorithm_name=request.algorithm,
        params=request.params,
        tsp_instance=request.tsp_instance
    )
    return result


class CompareRequest(BaseModel):
    problem: str = "onemax"
    algorithms: List[str] = DEFAULT_ALGORITHMS
    seeds: int = 20
    bit_length: int = 50
    max_iterations: int = 100000
    tsp_instance: Optional[str] = None


@app.post("/compare")
def compare_experiment(request: CompareRequest):
    """Convergence comparison: per-algorithm stats + averaged convergence curves."""
    # Guardrails to keep live computation responsive.
    seeds = max(1, min(request.seeds, 100))
    bit_length = max(2, min(request.bit_length, 2000))
    algorithms = request.algorithms or DEFAULT_ALGORITHMS
    try:
        return comparison.compare(
            problem=request.problem,
            algorithms=algorithms,
            seeds=seeds,
            bit_length=bit_length,
            max_iterations=request.max_iterations,
            tsp_instance=request.tsp_instance,
        )
    except Exception as e:
        return {"error": str(e)}


class ScalingRequest(BaseModel):
    problem: str = "onemax"
    algorithms: List[str] = DEFAULT_ALGORITHMS
    max_size: int = 100
    steps: int = 10
    seeds: int = 10
    max_iterations: int = 150000
    y_metric: str = "fitness_evaluations"


@app.post("/scaling")
def scaling_experiment(request: ScalingRequest):
    """Performance/scaling comparison: averaged metric vs. problem size (bitstring)."""
    seeds = max(1, min(request.seeds, 100))
    max_size = max(2, min(request.max_size, 1000))
    steps = max(1, min(request.steps, max_size))
    algorithms = request.algorithms or DEFAULT_ALGORITHMS
    try:
        return comparison.scaling(
            problem=request.problem,
            algorithms=algorithms,
            max_size=max_size,
            steps=steps,
            seeds=seeds,
            max_iterations=request.max_iterations,
            y_metric=request.y_metric,
        )
    except Exception as e:
        return {"error": str(e)}

@app.get("/export-csv")
def export_csv():
    """Export the latest run results as CSV."""
    if not OUTPUT_FILE.exists():
        return {"error": "No experiment run yet"}
    
    try:
        with open(OUTPUT_FILE) as f:
            data = json.load(f)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write metadata section
        writer.writerow(["Experiment Results"])
        writer.writerow([])
        writer.writerow(["Metadata"])
        writer.writerow(["Problem", data.get("problem", "")])
        writer.writerow(["Algorithm", data.get("algorithm", "")])
        writer.writerow(["Iterations", data.get("iterations", "")])
        writer.writerow(["Fitness Evaluations", data.get("fitness_evaluations", "")])
        writer.writerow(["Theoretical Runtime", data.get("theoretical_runtime", "")])
        
        if data.get("problem") == "tsp":
            writer.writerow(["TSP Instance", data.get("tsp_instance", "")])
            writer.writerow(["Number of Cities", data.get("num_cities", "")])
            writer.writerow(["Best Cost", data.get("best_cost", "")])
        
        writer.writerow([])
        
        # Write population/best solution
        if data.get("history") and len(data["history"]) > 0:
            population = data["history"][-1].get("Population", {})
            if population:
                writer.writerow(["Population Data"])
                for idx, (key, individual) in enumerate(population.items()):
                    if data.get("problem") == "tsp":
                        writer.writerow([f"{key}", f"Cost: {individual.get('cost', '')}", f"Tour length: {len(individual.get('tour', []))}"])
                    else:
                        writer.writerow([f"{key}", f"Fitness: {individual.get('fitness', '')}", f"Solution: {individual.get('bit', '')}"])
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.csv"
        
        # Return as downloadable file
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        return {"error": str(e)}
