from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

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


@app.get("/latest-run")
def get_latest_run():
    if not OUTPUT_FILE.exists():
        return {"error": "No experiment run yet"}

    return FileResponse(OUTPUT_FILE)


@app.post("/run")
def run_experiment(request: RunRequest):
    from optimization_framework.experiments.run_experiment import main

    result = main(request.problem)
    return result
