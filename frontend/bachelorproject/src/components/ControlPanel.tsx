export type ExperimentType = "single" | "convergence" | "scaling";

interface ControlPanelProps {
  problem: string;
  algorithm: string;
  experimentType: ExperimentType;
  onProblemChange: (p: string) => void;
  onAlgorithmChange: (a: string) => void;
  onExperimentTypeChange: (t: ExperimentType) => void;
  onRun: () => void;
  loading: boolean;
}

export default function ControlPanel({
  problem,
  algorithm,
  experimentType,
  onProblemChange,
  onAlgorithmChange,
  onExperimentTypeChange,
  onRun,
  loading,
}: ControlPanelProps) {

  return (
    <div className="card controlPanel">
      <div className="controlRow">
        <div className="field">
          <span className="label">Experiment</span>
          <select
            className="select"
            value={experimentType}
            onChange={(e) => onExperimentTypeChange(e.target.value as ExperimentType)}
          >
            <option value="single">Single run</option>
            <option value="convergence">Convergence comparison</option>
            <option value="scaling" disabled={problem === "tsp"}>
              Performance comparison (scaling)
            </option>
          </select>
        </div>

        <div className="field">
          <span className="label">Algorithm</span>
          <select
            className="select"
            value={algorithm}
            onChange={(e) => onAlgorithmChange(e.target.value)}
          >
            <option value="(μ+λ) EA">(μ+λ) EA</option>
            <option value="(1+1) EA">(1+1) EA</option>
            <option value="Simulated Annealing">Simulated annealing</option>
            <option value="ACO">Max-min ant system</option>
            <option value="P-ACO">Population-Based ACO</option>
          </select>
        </div>

        <div className="field">
          <span className="label">Problem</span>
          <select
            className="select"
            value={problem}
            onChange={(e) => onProblemChange(e.target.value)}
          >
            <option value="onemax">OneMax</option>
            <option value="leadingones">LeadingOnes</option>
            <option value="tsp">TSP</option>
          </select>
        </div>

        <button
          className="button"
          onClick={onRun}
          disabled={loading}
        >
          {loading ? "Loading..." : "Run"}
        </button>
      </div>
      {experimentType !== "single" && (
        <p className="subtitle" style={{ marginTop: 8 }}>
          {experimentType === "convergence"
            ? "Runs all algorithms over multiple seeds at a fixed size; the selected algorithm is highlighted."
            : "Averages each algorithm over multiple seeds across increasing problem sizes (bitstring only)."}
        </p>
      )}
    </div>
  );
}
