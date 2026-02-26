import { useState } from "react";

interface ControlPanelProps {
  onRun: (problem: string, algorithm: string) => void;
  loading: boolean;
}

export default function ControlPanel({ onRun, loading }: ControlPanelProps) {
  const [problem, setProblem] = useState("onemax");
  const [algorithm, setAlgorithm] = useState("(μ+λ) EA");

  return (
    <div className="card controlPanel">
      <div className="controlRow">
        <div className="field">
          <span className="label">Algorithm</span>
          <select
            className="select"
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
          >
            <option value="(μ+λ) EA">(μ+λ) EA</option>
            <option value="(1+1) EA">(1+1) EA</option>
          </select>
        </div>

        <div className="field">
          <span className="label">Problem</span>
          <select
            className="select"
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
          >
            <option value="onemax">OneMax</option>
            <option value="leadingones">LeadingOnes</option>
          </select>
        </div>

        <button
          className="button"
          onClick={() => onRun(problem, algorithm)}
          disabled={loading}
        >
          {loading ? "Loading..." : "Run"}
        </button>
      </div>
    </div>
  );
}
