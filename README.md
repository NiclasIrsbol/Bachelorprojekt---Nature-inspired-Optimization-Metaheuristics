# Nature-Inspired Optimization Metaheuristics

A comprehensive framework for visualizing, evaluating, and comparing nature-inspired optimization algorithms on various problem domains. This project implements multiple metaheuristic algorithms and benchmarks them on classic optimization problems with an interactive web interface.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [API Overview](#api-overview)
- [Running Experiments](#running-experiments)
- [Testing](#testing)
- [Features](#features)
- [References](#references)

## Overview

This project provides a unified framework for studying nature-inspired optimization metaheuristics. It includes both a Python backend (REST API) and a React frontend for interactive exploration and visualization of algorithm behavior on multiple optimization problems.

**Key capabilities:**
- Run metaheuristic algorithms on various optimization problems
- Track fitness/cost evolution over time
- Visualize algorithm performance and search behavior
- Compare multiple algorithms on identical problem instances
- Parameterize algorithms for performance tuning
- Benchmark on real-world TSP instances (TSPLIB)

## Architecture

### Backend (Python)
- **Framework:** FastAPI + Uvicorn
- **Structure:** Modular design with separate algorithm and problem modules
- **Key Components:**
  - `algorithms/`: Algorithm implementations
  - `problems/`: Problem definitions and test functions
  - `operators/`: Genetic operators and mutation strategies
  - `experiments/`: Experiment runner and visualization utilities
  - `api.py`: REST API endpoints

### Frontend (React + TypeScript)
- **Framework:** React 19 with Vite
- **Visualization:** Recharts for fitness tracking, custom canvas for TSP tours
- **Features:**
  - Interactive algorithm selection
  - Real-time parameter adjustment
  - Live fitness/cost tracking
  - TSP tour visualization
  - Experiment history

## Project Structure

```
.
├── README.md                          # Project documentation
├── backend/                           # Python backend
│   ├── pyproject.toml                 # Python dependencies
│   ├── pytest.ini                     # Pytest configuration
│   ├── optimization_framework/        # Main package
│   │   ├── api.py                     # FastAPI endpoints
│   │   ├── algorithms/                # Algorithm implementations
│   │   │   ├── one_plus_one_EA.py
│   │   │   ├── mu_plus_lambda_EA.py
│   │   │   ├── simulated_annealing.py
│   │   │   └── ant_optimization_problem.py
│   │   ├── operators/                 # Genetic operators
│   │   │   └── gaoperators.py
│   │   ├── problems/                  # Problem definitions
│   │   │   ├── onemax.py
│   │   │   ├── leadingones.py
│   │   │   ├── tsp.py
│   │   │   └── tsplib/                # TSPLIB instances (80+ files)
│   │   └── experiments/               # Experiment utilities
│   │       ├── run_experiment.py
│   │       └── tsp_visualization.py
│   ├── tests/                         # Test suite (60+ tests)
│   │   ├── test_onemax.py
│   │   ├── test_leadingones.py
│   │   ├── test_SA.py
│   │   └── test_ACO.py
│   └── output/                        # Experiment results
│       └── latest_run.json
├── frontend/                          # React + TypeScript frontend
│   ├── bachelorproject/
│   │   ├── package.json
│   │   ├── index.html
│   │   ├── vite.config.ts
│   │   ├── tsconfig.json
│   │   ├── src/
│   │   │   ├── App.tsx
│   │   │   ├── main.tsx
│   │   │   └── components/            # React components
│   │   └── public/                    # Static assets
└── output/                            # Shared results
    └── latest_run.json

```

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Git

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e ".[test]"
```

### TSPLIB Setup (Required for TSP Algorithms)

Before running any TSP-related experiments, you must clone the TSPLIB repository:

```bash
# Navigate to the problems directory
cd backend/optimization_framework/problems

# Clone the TSPLIB repository
git clone https://github.com/mastqe/tsplib.git

# Verify installation
# You should now have a tsplib/ directory with .tsp files
ls tsplib/ | head
```

**Note:** The TSP algorithms will not work without the TSPLIB instances. The cloned repository provides 80+ real-world TSP instances ranging from 14 to 85,900 cities.

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend/bachelorproject

# Install dependencies
npm install
```

## Getting Started

### Prerequisites Check

**Before running experiments, ensure:**
1. Backend virtual environment is activated
2. Frontend dependencies are installed
3. **TSPLIB repository is cloned** (required for TSP experiments)

### Running the Full Application

#### 1. Start the Backend API (from `backend/` directory)

```bash
# Activate virtual environment first
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# Start the API server
python -m uvicorn optimization_framework.api:app --reload
```

The API will be available at `http://localhost:8000`

#### 2. Start the Frontend (from `frontend/bachelorproject/` directory)

```bash
npx vite
```

The frontend will be available at `http://localhost:5173`

#### 3. Access the Application

Open your browser and navigate to `http://localhost:5173`

### Running a Single Experiment (CLI)

```bash
# From backend directory
python -m optimization_framework.experiments.run_experiment onemax "(μ+λ) EA"
python -m optimization_framework.experiments.run_experiment tsp "Simulated Annealing"
```

## API Overview

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. **POST /run** - Run an Experiment
Execute an optimization algorithm on a specified problem.

**Request Body:**
```json
{
  "problem": "onemax",           // or "leadingones", "tsp"
  "algorithm": "(1+1) EA",       // Algorithm name
  "params": {                    // Optional parameters
    "bit_length": 20,
    "max_iterations": 10000
  },
  "tsp_instance": "a280.tsp"    // For TSP: specific instance
}
```

**Response:**
```json
{
  "problem": "onemax",
  "algorithm": "(1+1) EA",
  "iterations": 145,
  "fitness_evaluations": 2891,
  "best_solution": "11111111111111111111",
  "best_fitness": 20,
  "fitness_over_time": [
    {"generation": 0, "fitness": 8},
    {"generation": 1, "fitness": 9},
    ...
  ]
}
```

#### 2. **GET /latest-run** - Get Last Result
Retrieve the results of the most recent experiment.

**Response:** Same as `/run` response

#### 3. **GET /tsp-instances** - List TSP Instances
Get available TSPLIB instances with metadata.

**Response:**
```json
{
  "instances": [
    {
      "name": "a280.tsp",
      "cities": 280,
      "optimal": 2579.35
    },
    ...
  ]
}
```

#### 4. **GET /export-csv** - Export Results
Download experiment results in CSV format.

**Response:** CSV file with metadata and fitness history

### Algorithm Parameters

#### (1+1) EA
```json
{
  "bit_length": 20,              // Bitstring length
  "max_iterations": 10000,       // Max iterations
  "prob": 0.05                   // Mutation probability
}
```

#### (μ+λ) EA
```json
{
  "bit_length": 20,
  "mu_size": 20,                 // Parent population
  "lambda_size": 40,             // Offspring per generation
  "tournament_k": 3,             // Tournament size
  "max_iterations": 5000
}
```

#### Simulated Annealing
```json
{
  "bit_length": 20,
  "cooling": 0.99,               // Cooling rate (0.99-0.9995)
  "T0": 100.0,                   // Initial temperature
  "max_iterations": 100000
}
```

#### MMAS ACO
```json
{
  "bit_length": 100,
  "rho": 0.1,                    // Evaporation rate
  "alpha": 1,                    // Pheromone weight
  "beta": 2,                     // Heuristic weight
  "max_iterations": 1000
}
```

#### P-ACO
```json
{
  "bit_length": 100,
  "archive_size": 10,            // Best solutions to maintain
  "num_ants": 30,                // Ants per iteration
  "q0": 0.9,                     // Greedy probability
  "max_iterations": 1000
}
```

## Running Experiments

### Via Frontend
1. Select a problem (OneMax, LeadingOnes, or TSP)
2. Choose an algorithm
3. Adjust parameters (optional)
4. Click "Run Experiment"
5. View real-time results and visualizations

### Via API
```bash
# Example: Run (μ+λ) EA on OneMax
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "onemax",
    "algorithm": "(μ+λ) EA",
    "params": {"bit_length": 20, "mu_size": 20, "lambda_size": 40}
  }'

# Example: Run SA on TSP
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "tsp",
    "algorithm": "Simulated Annealing",
    "params": {"cooling": 0.995, "T0": 500},
    "tsp_instance": "berlin52.tsp"
  }'
```

### Via CLI
```bash
cd backend
python -m optimization_framework.experiments.run_experiment onemax "(μ+λ) EA"
python -m optimization_framework.experiments.run_experiment tsp ACO
```

## Testing

### Run All Tests
```bash
cd backend
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/test_onemax.py -v
python -m pytest tests/test_SA.py -v
```

### Test Coverage
The project includes 60+ tests covering:
- Convergence validation
- Return type/structure validation
- Parameter sensitivity
- Edge cases
- Solution validity
- Progress tracking (fitness/cost history)
- Population structure validation

Test files:
- `test_onemax.py` - OneMax problem and (1+1) EA, (μ+λ) EA tests
- `test_leadingones.py` - LeadingOnes problem and algorithm tests
- `test_SA.py` - Simulated Annealing tests
- `test_ACO.py` - ACO and P-ACO tests

**Last Updated:** May 2026
