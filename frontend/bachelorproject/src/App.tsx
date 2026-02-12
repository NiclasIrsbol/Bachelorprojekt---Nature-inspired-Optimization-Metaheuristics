import { useEffect, useState, useCallback } from "react";
import "./App.css";
import ControlPanel from "./components/ControlPanel";
import FitnessChart from "./components/FitnessChart";
import MetricsPanel from "./components/MetricsPanel";
import PopulationPanel from "./components/PopulationPanel";

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
  history: Generation[];
}

function getStoredTheme(): Theme {
  return (localStorage.getItem("theme") as Theme) ?? "dark";
}

export default function App() {
  const [data, setData] = useState<ExperimentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(getStoredTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const fetchData = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/latest-run`)
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
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const toggleTheme = () =>
    setTheme((t) => (t === "dark" ? "light" : "dark"));

  if (loading && !data) {
    return (
      <div className="page">
        <div className="loadingWrap">
          <div className="spinner" />
        </div>
      </div>
    );
  }

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

      <ControlPanel onRun={fetchData} loading={loading} />

      {error ? (
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
          />
          <FitnessChart population={population} />
          <PopulationPanel population={population} />
        </div>
      ) : (
        <div className="emptyState">
          <h2>No Experiment Data</h2>
          <p>Run an experiment to see results here.</p>
        </div>
      )}
    </div>
  );
}
