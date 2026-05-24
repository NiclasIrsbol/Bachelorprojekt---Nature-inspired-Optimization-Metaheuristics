interface Individual {
  bit?: string;
  fitness?: number;
  tour?: number[];
  cost?: number;
}

interface Population {
  [key: string]: Individual;
}

interface MetricsPanelProps {
  population: Population;
  algorithm: string;
  problem: string;
  temp?: number;
  iterations: number;
  fitnessEvaluations: number;
  theoreticalRuntime: string;
  tspInstance?: string;
  numCities?: number;
  bestCost?: number;
}

export default function MetricsPanel({
  population,
  algorithm,
  problem,
  temp,
  iterations,
  fitnessEvaluations,
  theoreticalRuntime,
  tspInstance,
  numCities,
  bestCost,
}: MetricsPanelProps) {
  const entries = Object.values(population);
  const isTsp = problem === "tsp";

  const handleDownloadCSV = async () => {
    try {
      const response = await fetch("http://localhost:8000/export-csv");
      if (!response.ok) throw new Error("Failed to download");
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const contentDisposition = response.headers.get("content-disposition");
      const filename = contentDisposition?.split("filename=")[1]?.replace(/"/g, "") || "results.csv";
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Error downloading CSV:", err);
      alert("Failed to download CSV");
    }
  };

  const metrics: { label: string; value: string | number }[] = [
    { label: "Algorithm", value: algorithm },
    { label: "Problem", value: isTsp ? `TSP (${tspInstance ?? "?"})` : problem },
  ];

  if (isTsp) {
    metrics.push({ label: "Cities", value: numCities ?? "?" });
    metrics.push({ label: "Best Cost", value: bestCost?.toLocaleString() ?? "?" });
  } else {
    const fitnesses = entries.map((e) => e.fitness ?? 0);
    const best = Math.max(...fitnesses);
    metrics.push({ label: "Best Fitness", value: best });
  }

  if (algorithm === "Simulated Annealing" && temp !== undefined) {
    metrics.push({ label: "Temperature", value: temp });
  }
  metrics.push({ label: "Iterations", value: iterations });
  metrics.push({ label: "Fitness Evaluations", value: fitnessEvaluations.toLocaleString() });
  metrics.push({ label: "Theoretical Complexity", value: theoreticalRuntime });

  return (
    <div className="card">
      <div className="cardHeader">
        <h3 className="cardTitle">Metrics</h3>
        <button className="downloadButton" onClick={handleDownloadCSV} title="Download results as CSV">
          📋
        </button>
      </div>
      <div className="metricsGrid">
        {metrics.map((m) => (
          <div className="metric" key={m.label}>
            <span className="metricLabel">{m.label}</span>
            <p className="metricValue">{m.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
