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
  temp: number;
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

  const metrics: { label: string; value: string | number }[] = [
    { label: "Algorithm", value: algorithm },
    { label: "Problem", value: isTsp ? `TSP (${tspInstance ?? "?"})` : problem },
  ];

  if (isTsp) {
    metrics.push({ label: "Cities", value: numCities ?? "?" });
    metrics.push({ label: "Best Cost", value: bestCost?.toLocaleString() ?? "?" });
    if (entries.length > 1) {
      const costs = entries.map((e) => e.cost ?? 0);
      const avg = costs.reduce((a, b) => a + b, 0) / costs.length;
      metrics.push({ label: "Average Cost", value: avg.toFixed(0) });
    }
  } else {
    const fitnesses = entries.map((e) => e.fitness ?? 0);
    const best = Math.max(...fitnesses);
    const avg = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    const bitLength = entries[0]?.bit?.length ?? 0;

    metrics.push({ label: "Population Size", value: entries.length });
    metrics.push({ label: "Bitstring Length", value: bitLength });
    metrics.push({ label: "Best Fitness", value: best });
    metrics.push({ label: "Average Fitness", value: avg.toFixed(2) });
  }

  metrics.push({ label: "Temperature", value: temp });
  metrics.push({ label: "Iterations", value: iterations });
  metrics.push({ label: "Fitness Evaluations", value: fitnessEvaluations.toLocaleString() });
  metrics.push({ label: "Theoretical Complexity", value: theoreticalRuntime });

  return (
    <div className="card">
      <div className="cardHeader">
        <h3 className="cardTitle">Metrics</h3>
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
