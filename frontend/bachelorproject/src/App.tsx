import { useEffect, useState, useCallback } from "react";
import "./App.css";
import ControlPanel, { type ExperimentType } from "./components/ControlPanel";
import FitnessChart from "./components/FitnessChart";
import MetricsPanel from "./components/MetricsPanel";
import PopulationPanel from "./components/PopulationPanel";
import ParametersPanel, { getDefaultParams } from "./components/ParametersPanel";
import TSPInstanceSelector from "./components/TSPInstanceSelector";
import ComparisonPanel, {
  type CompareData,
  type ScalingData,
} from "./components/ComparisonPanel";
import ComparisonSettings from "./components/ComparisonSettings";

const API_BASE = "http://localhost:8000";

const ALL_ALGORITHMS = [
  "(1+1) EA",
  "(μ+λ) EA",
  "Simulated Annealing",
  "MMAS-ACO",
  "P-ACO",
];

type Theme = "dark" | "light";

interface Individual {
  bit?: string;
  fitness?: number;
  tour?: number[];
  cost?: number;
}

interface Population {
  [key: string]: Individual;
}

interface Generation {
  Population: Population;
}

interface ExperimentData {
  problem: string;
  algorithm: string;
  iterations: number;
  temp?: number;
  fitness_evaluations: number;
  theoretical_runtime: string;
  history: Generation[];
  coords?: { x: number; y: number }[];
  fitness_over_time?: { generation: number; fitness: number }[];
  tsp_instance?: string;
  num_cities?: number;
  best_cost?: number;
  best_tour?: number[];
  city_coords?: Record<string, [number, number]>;
  tour_map_image?: string;
}

function getStoredTheme(): Theme {
  return (localStorage.getItem("theme") as Theme) ?? "dark";
}

export default function App() {
  const [data, setData] = useState<ExperimentData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(getStoredTheme);
  const [algorithm, setAlgorithm] = useState("(μ+λ) EA");
  const [problem, setProblem] = useState("onemax");
  const [params, setParams] = useState<Record<string, number | string>>(() => getDefaultParams("(μ+λ) EA"));
  const [tspInstance, setTspInstance] = useState<string | null>(null);

  const [experimentType, setExperimentType] = useState<ExperimentType>("single");
  const [compareData, setCompareData] = useState<CompareData | null>(null);
  const [scalingData, setScalingData] = useState<ScalingData | null>(null);
  const [seeds, setSeeds] = useState(20);
  const [bitLength, setBitLength] = useState(50);
  const [maxIterations, setMaxIterations] = useState(50000);
  const [maxSize, setMaxSize] = useState(100);
  const [steps, setSteps] = useState(10);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    if (problem === "tsp" && experimentType === "scaling") {
      setExperimentType("convergence");
    }
  }, [problem, experimentType]);

  const runExperiment = useCallback(() => {
    setLoading(true);
    setError(null);
    const payload: any = { problem, algorithm, params };
    if (problem === "tsp" && tspInstance) {
      payload.tsp_instance = tspInstance;
    }
    fetch(`${API_BASE}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: ExperimentData) => {
        if ("error" in json) {
          setError((json as unknown as { error: string }).error);
          setData(null);
        } else {
          setData(json);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Could not connect to backend. Is the server running?");
        setLoading(false);
      });
  }, [problem, algorithm, params, tspInstance]);

  const runComparison = useCallback(() => {
    setLoading(true);
    setError(null);
    const payload: any = {
      problem,
      algorithms: ALL_ALGORITHMS,
      seeds,
    };
    if (problem === "tsp") {
      payload.max_iterations = maxIterations;
      if (tspInstance) payload.tsp_instance = tspInstance;
    } else {
      payload.bit_length = bitLength;
    }
    fetch(`${API_BASE}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: any) => {
        if (json && json.error) {
          setError(json.error);
          setCompareData(null);
        } else {
          setCompareData(json as CompareData);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Could not connect to backend. Is the server running?");
        setLoading(false);
      });
  }, [problem, seeds, bitLength, maxIterations, tspInstance]);

  const runScaling = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/scaling`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        problem,
        algorithms: ALL_ALGORITHMS,
        max_size: maxSize,
        steps,
        seeds,
        y_metric: "fitness_evaluations",
      }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: any) => {
        if (json && json.error) {
          setError(json.error);
          setScalingData(null);
        } else {
          setScalingData(json as ScalingData);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Could not connect to backend. Is the server running?");
        setLoading(false);
      });
  }, [problem, maxSize, steps, seeds]);

  const handleRun = useCallback(() => {
    if (experimentType === "convergence") runComparison();
    else if (experimentType === "scaling") runScaling();
    else runExperiment();
  }, [experimentType, runComparison, runScaling, runExperiment]);

  const toggleTheme = () =>
    setTheme((t) => (t === "dark" ? "light" : "dark"));

  const generation =
    data && data.history.length > 0
      ? data.history[data.history.length - 1]
      : null;
  const population = generation?.Population ?? null;

  return (
    <div className="page">
      <div className="topBar">
        <div className="topBarLeft">
          <h1 className="title">Optimization Dashboard</h1>
          <p className="subtitle">
            Nature-inspired Optimization Metaheuristics
          </p>
        </div>
        <button
          className="themeToggle"
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          {theme === "dark" ? "\u2600" : "\u263E"}
        </button>
      </div>

      <ControlPanel
        problem={problem}
        algorithm={algorithm}
        experimentType={experimentType}
        onProblemChange={setProblem}
        onAlgorithmChange={setAlgorithm}
        onExperimentTypeChange={setExperimentType}
        onRun={handleRun}
        loading={loading}
      />

      {experimentType !== "single" && (
        <ComparisonSettings
          experimentType={experimentType}
          problem={problem}
          seeds={seeds}
          bitLength={bitLength}
          maxIterations={maxIterations}
          maxSize={maxSize}
          steps={steps}
          onSeedsChange={setSeeds}
          onBitLengthChange={setBitLength}
          onMaxIterationsChange={setMaxIterations}
          onMaxSizeChange={setMaxSize}
          onStepsChange={setSteps}
        />
      )}

      {experimentType === "single" && (
        <div className={problem === "tsp" ? "paramAndTspWrapper" : ""}>
          <ParametersPanel algorithm={algorithm} problem={problem} params={params} onChange={setParams} />
          {problem === "tsp" && (
            <TSPInstanceSelector
              selectedInstance={tspInstance}
              onInstanceChange={setTspInstance}
            />
          )}
        </div>
      )}

      {experimentType === "convergence" && problem === "tsp" && (
        <TSPInstanceSelector selectedInstance={tspInstance} onInstanceChange={setTspInstance} />
      )}

      {loading ? (
        <div className="loadingWrap">
          <div className="spinner" />
        </div>
      ) : error ? (
        <div className="emptyState">
          <h2>Error</h2>
          <p>{error}</p>
        </div>
      ) : experimentType === "convergence" && compareData ? (
        <ComparisonPanel
          mode="convergence"
          compareData={compareData}
          selectedAlgorithm={algorithm}
        />
      ) : experimentType === "scaling" && scalingData ? (
        <ComparisonPanel
          mode="scaling"
          scalingData={scalingData}
          selectedAlgorithm={algorithm}
        />
      ) : experimentType === "single" && population ? (
        <div className="grid">
          <MetricsPanel
            population={population}
            algorithm={data!.algorithm}
            problem={data!.problem}
            iterations={data!.iterations}
            temp={data!.temp ?? 0}
            fitnessEvaluations={data!.fitness_evaluations}
            theoreticalRuntime={data!.theoretical_runtime}
            tspInstance={data!.tsp_instance}
            numCities={data!.num_cities}
            bestCost={data!.best_cost}
          />
          <FitnessChart
            population={population}
            problem={data!.problem}
            coords={data!.coords}
            fitnessOverTime={data!.fitness_over_time}
            cityCoords={data!.city_coords}
            bestTour={data!.best_tour}
            tourMapImage={data!.tour_map_image}
          />
          <PopulationPanel population={population} problem={data!.problem} />
        </div>
      ) : (
        <div className="emptyState">
          <h2>No Experiment Data</h2>
          <p>
            {experimentType === "single"
              ? "Select a problem and click Run to see results."
              : "Configure settings above and click Run to compare algorithms."}
          </p>
        </div>
      )}
    </div>
  );
}
