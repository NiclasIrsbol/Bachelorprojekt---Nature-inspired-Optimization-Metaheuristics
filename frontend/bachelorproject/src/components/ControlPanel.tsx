interface ControlPanelProps {
  onRun: () => void;
  loading: boolean;
}

export default function ControlPanel({ onRun, loading }: ControlPanelProps) {
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
          <select className="select" defaultValue="onemax">
            <option value="onemax">OneMax</option>
          </select>
        </div>

        <button className="button" onClick={onRun} disabled={loading}>
          {loading ? "Loading..." : "Run"}
        </button>
      </div>
    </div>
  );
}
