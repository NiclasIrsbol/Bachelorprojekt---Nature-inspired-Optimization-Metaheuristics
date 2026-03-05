import { useEffect, useState, useCallback } from "react";
import "./App.css";
import ControlPanel from "./components/ControlPanel";
import FitnessChart from "./components/FitnessChart";
import MetricsPanel from "./components/MetricsPanel";
import PopulationPanel from "./components/PopulationPanel";
import ParametersPanel, { getDefaultParams } from "./components/ParametersPanel";

const API_BASE = "http://localhost:8000";

type Theme = "dark" | "light";

interface Individual {
  bit: string;
  fitness: number;
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
  temp: number;
  fitness_evaluations: number;
  theoretical_runtime: string;
  history: Generation[];
  coords?: { x: number; y: number }[];
  fitness_over_time?: { generation: number; fitness: number }[];
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
  const [params, setParams] = useState<Record<string, number>>(() => getDefaultParams("(μ+λ) EA"));

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const runExperiment = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ problem, algorithm, params }),
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
  }, [problem, algorithm, params]);

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
        onProblemChange={setProblem}
        onAlgorithmChange={setAlgorithm}
        onRun={runExperiment}
        loading={loading}
      />

      <ParametersPanel algorithm={algorithm} params={params} onChange={setParams} />

      {loading ? (
        <div className="loadingWrap">
          <div className="spinner" />
        </div>
      ) : error ? (
        <div className="emptyState">
          <h2>No Data Available</h2>
          <p>{error}</p>
        </div>
      ) : population ? (
        <div className="grid">
          <MetricsPanel
            population={population}
            algorithm={data!.algorithm}
            problem={data!.problem}
            iterations={data!.iterations}
            temp={data!.temp}
            fitnessEvaluations={data!.fitness_evaluations}
            theoreticalRuntime={data!.theoretical_runtime}
          />
          <FitnessChart population={population} coords={data!.coords} fitnessOverTime={data!.fitness_over_time} />
          <PopulationPanel population={population} />
        </div>
      ) : (
        <div className="emptyState">
          <h2>No Experiment Data</h2>
          <p>Select a problem and click Run to see results.</p>
        </div>
      )}
    </div>
  );
}
