interface ControlPanelProps {
  problem: string;
  algorithm: string;
  onProblemChange: (p: string) => void;
  onAlgorithmChange: (a: string) => void;
  onRun: () => void;
  loading: boolean;
}

export default function ControlPanel({
  problem,
  algorithm,
  onProblemChange,
  onAlgorithmChange,
  onRun,
  loading,
}: ControlPanelProps) {

  return (
    <div className="card controlPanel">
      <div className="controlRow">
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
            <option value="ACO">Ant Colony Optimization</option>
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
    </div>
  );
}
