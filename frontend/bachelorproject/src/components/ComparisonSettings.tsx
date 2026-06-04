import type { ExperimentType } from "./ControlPanel";

interface ComparisonSettingsProps {
  experimentType: Exclude<ExperimentType, "single">;
  problem: string;
  seeds: number;
  bitLength: number;
  maxIterations: number;
  maxSize: number;
  steps: number;
  onSeedsChange: (n: number) => void;
  onBitLengthChange: (n: number) => void;
  onMaxIterationsChange: (n: number) => void;
  onMaxSizeChange: (n: number) => void;
  onStepsChange: (n: number) => void;
}

export default function ComparisonSettings({
  experimentType,
  problem,
  seeds,
  bitLength,
  maxIterations,
  maxSize,
  steps,
  onSeedsChange,
  onBitLengthChange,
  onMaxIterationsChange,
  onMaxSizeChange,
  onStepsChange,
}: ComparisonSettingsProps) {
  const isTsp = problem === "tsp";

  return (
    <div className="card controlPanel">
      <div className="paramGrid">
        <span className="paramTitle">Comparison settings</span>

        <div className="paramField">
          <label className="label" htmlFor="cmp-seeds">
            Seeds (runs to average)
          </label>
          <input
            id="cmp-seeds"
            className="paramInput"
            type="number"
            min={1}
            max={100}
            value={seeds}
            onChange={(e) => onSeedsChange(Number(e.target.value))}
          />
        </div>

        {experimentType === "convergence" && !isTsp && (
          <div className="paramField">
            <label className="label" htmlFor="cmp-bitlen">
              Bit length (n)
            </label>
            <input
              id="cmp-bitlen"
              className="paramInput"
              type="number"
              min={2}
              max={500}
              value={bitLength}
              onChange={(e) => onBitLengthChange(Number(e.target.value))}
            />
          </div>
        )}

        {experimentType === "convergence" && isTsp && (
          <div className="paramField">
            <label className="label" htmlFor="cmp-maxiter">
              Max iterations
            </label>
            <input
              id="cmp-maxiter"
              className="paramInput"
              type="number"
              min={100}
              max={500000}
              step={100}
              value={maxIterations}
              onChange={(e) => onMaxIterationsChange(Number(e.target.value))}
            />
          </div>
        )}

        {experimentType === "scaling" && (
          <>
            <div className="paramField">
              <label className="label" htmlFor="cmp-maxsize">
                Max size (n)
              </label>
              <input
                id="cmp-maxsize"
                className="paramInput"
                type="number"
                min={10}
                max={500}
                value={maxSize}
                onChange={(e) => onMaxSizeChange(Number(e.target.value))}
              />
            </div>
            <div className="paramField">
              <label className="label" htmlFor="cmp-steps">
                Step size
              </label>
              <input
                id="cmp-steps"
                className="paramInput"
                type="number"
                min={1}
                max={100}
                value={steps}
                onChange={(e) => onStepsChange(Number(e.target.value))}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
