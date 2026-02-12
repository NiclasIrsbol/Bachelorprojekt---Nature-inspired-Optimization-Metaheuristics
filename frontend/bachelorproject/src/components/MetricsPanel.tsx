interface Individual {
  bit: string;
  fitness: number;
}

interface Population {
  [key: string]: Individual;
}

interface MetricsPanelProps {
  population: Population;
  algorithm: string;
  problem: string;
}

export default function MetricsPanel({
  population,
  algorithm,
  problem,
}: MetricsPanelProps) {
  const entries = Object.values(population);
  const fitnesses = entries.map((e) => e.fitness);

  const best = Math.max(...fitnesses);
  const worst = Math.min(...fitnesses);
  const avg = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
  const bitLength = entries[0]?.bit.length ?? 0;

  const metrics = [
    { label: "Algorithm", value: algorithm },
    { label: "Problem", value: problem },
    { label: "Population Size", value: entries.length },
    { label: "Bitstring Length", value: bitLength },
    { label: "Best Fitness", value: best },
    { label: "Average Fitness", value: avg.toFixed(2) },
    { label: "Worst Fitness", value: worst },
  ];

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
