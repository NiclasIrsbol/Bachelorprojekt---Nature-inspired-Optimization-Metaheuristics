interface Individual {
  bit?: string;
  fitness?: number;
  tour?: number[];
  cost?: number;
}

interface Population {
  [key: string]: Individual;
}

interface PopulationPanelProps {
  population: Population;
  problem: string;
}

function formatTour(tour: number[]): string {
  return tour.join(" → ") + " → " + tour[0];
}

export default function PopulationPanel({ population, problem }: PopulationPanelProps) {
  const entries = Object.entries(population);
  const isTsp = problem === "tsp";

  if (entries.length === 0) {
    return (
      <div className="card">
        <div className="cardHeader">
          <h3 className="cardTitle">Population</h3>
        </div>
        <p style={{ color: "var(--text-muted)", fontStyle: "italic", padding: "1rem" }}>
          Single-individual algorithm — no population to display
        </p>
      </div>
    );
  }

  if (isTsp) {
    const sorted = [...entries].sort(
      ([, a], [, b]) => (a.cost ?? Infinity) - (b.cost ?? Infinity)
    );
    const [, best] = sorted[0];

    return (
      <div className="card">
        <div className="cardHeader">
          <h3 className="cardTitle">Best Tour</h3>
        </div>

        <div className="solutionBox">
          <div className="solutionMeta">
            <span className="pill pillGreen">
              Cost: {best.cost?.toLocaleString()}
            </span>
            <span className="pill mono">{best.tour?.length} cities</span>
          </div>
          <pre className="solutionMono" style={{ wordBreak: "break-all", whiteSpace: "pre-wrap" }}>
            {best.tour ? formatTour(best.tour) : "—"}
          </pre>
        </div>

        {sorted.length > 1 && (
          <>
            <div className="cardHeader" style={{ marginTop: 16 }}>
              <h3 className="cardTitle">Population</h3>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                sorted by cost
              </span>
            </div>
            <div className="popList">
              {sorted.map(([key, ind], i) => (
                <div className="popItem" key={key}>
                  <span className="popIndex">#{i + 1}</span>
                  <span className="popBit" style={{ fontSize: "0.75rem" }}>
                    {ind.tour ? ind.tour.slice(0, 10).join("→") + (ind.tour.length > 10 ? "…" : "") : "—"}
                  </span>
                  <span className="popFitness">{ind.cost?.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    );
  }

  const sorted = [...entries].sort(([, a], [, b]) => (b.fitness ?? 0) - (a.fitness ?? 0));
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
          <span className="pill mono">{best.bit?.length ?? 0} bits</span>
        </div>
        <pre className="solutionMono">{best.bit?.split("").join(" ")}</pre>
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
