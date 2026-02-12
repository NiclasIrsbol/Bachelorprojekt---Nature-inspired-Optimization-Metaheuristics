interface Individual {
  bit: string;
  fitness: number;
}

interface Population {
  [key: string]: Individual;
}

interface PopulationPanelProps {
  population: Population;
}

export default function PopulationPanel({ population }: PopulationPanelProps) {
  const entries = Object.entries(population);
  const sorted = [...entries].sort(([, a], [, b]) => b.fitness - a.fitness);
  const [, best] = sorted[0];

  return (
    <div className="card">
      <div className="cardHeader">
        <h3 className="cardTitle">Best Solution</h3>
      </div>

      <div className="solutionBox">
        <div className="solutionMeta">
          <span className="pill pillGreen">
            Fitness: {best.fitness}
          </span>
          <span className="pill mono">{best.bit.length} bits</span>
        </div>
        <pre className="solutionMono">{best.bit.split("").join(" ")}</pre>
      </div>

      <div className="cardHeader" style={{ marginTop: 16 }}>
        <h3 className="cardTitle">Population</h3>
        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
          sorted by fitness
        </span>
      </div>

      <div className="popList">
        {sorted.map(([key, ind], i) => (
          <div className="popItem" key={key}>
            <span className="popIndex">#{i + 1}</span>
            <span className="popBit">{ind.bit}</span>
            <span className="popFitness">{ind.fitness}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
