import { useState } from "react";

interface ControlPanelProps {
  onRun: (problem: string) => void;
  loading: boolean;
}

export default function ControlPanel({ onRun, loading }: ControlPanelProps) {
  const [problem, setProblem] = useState("onemax");

  return (
    <div className="card controlPanel">
      <div className="controlRow">
        <div className="field">
          <span className="label">Algorithm</span>
          <select className="select" defaultValue="ga">
            <option value="ga">Genetic Algorithm</option>
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
          onClick={() => onRun(problem)}
          disabled={loading}
        >
          {loading ? "Loading..." : "Run"}
        </button>
      </div>
    </div>
  );
}
