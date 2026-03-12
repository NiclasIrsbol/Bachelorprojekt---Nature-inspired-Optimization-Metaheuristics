import { useEffect, useMemo } from "react";

interface ParamDef {
  key: string;
  label: string;
  default: number;
  min: number;
  max: number;
  step: number;
}

const ALGORITHM_PARAMS: Record<string, ParamDef[]> = {
  "(1+1) EA": [
    { key: "bit_length", label: "Bit length", default: 20, min: 2, max: 1000, step: 1 },
    { key: "prob", label: "Mutation prob", default: 0.05, min: 0.001, max: 1, step: 0.001 },
  ],

  "(μ+λ) EA": [
    { key: "bit_length", label: "Bit length", default: 20, min: 2, max: 1000, step: 1 },
    { key: "mu_size", label: "μ (parents)", default: 20, min: 2, max: 200, step: 1 },
    { key: "lambda_size", label: "λ (offspring)", default: 40, min: 2, max: 400, step: 1 },
    { key: "tournament_k", label: "Tournament k", default: 3, min: 2, max: 20, step: 1 },
    { key: "mutation_prob", label: "Mutation prob", default: 0.05, min: 0.001, max: 1, step: 0.001 },
  ],

  "Simulated Annealing": [
    { key: "bit_length", label: "Bit length", default: 20, min: 2, max: 1000, step: 1 },
    { key: "cooling", label: "Cooling rate", default: 0.99, min: 0.01, max: 0.999, step: 0.001 },
    { key: "T0", label: "Initial temp (T₀)", default: 100, min: 1, max: 10000, step: 1 },
  ],
  
  "ACO": [
    { key: "bit_length", label: "Bit length", default: 100, min: 2, max: 1000, step: 1 },
    { key: "rho", label: "Evaporation (ρ)", default: 0.1, min: 0.01, max: 1, step: 0.01 },
    { key: "max_iterations", label: "Max iterations", default: 10000, min: 100, max: 100000, step: 100 },
  ],
};

export function getDefaultParams(algorithm: string): Record<string, number> {
  const defs = ALGORITHM_PARAMS[algorithm] ?? [];
  const out: Record<string, number> = {};
  for (const d of defs) out[d.key] = d.default;
  return out;
}

interface ParametersPanelProps {
  algorithm: string;
  params: Record<string, number>;
  onChange: (params: Record<string, number>) => void;
}

export default function ParametersPanel({
  algorithm,
  params,
  onChange,
}: ParametersPanelProps) {
  const defs = useMemo(() => ALGORITHM_PARAMS[algorithm] ?? [], [algorithm]);

  // Reset params to defaults when algorithm changes
  useEffect(() => {
    onChange(getDefaultParams(algorithm));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithm]);

  if (defs.length === 0) return null;

  const handleChange = (key: string, value: number) => {
    onChange({ ...params, [key]: value });
  };

  return (
    <div className="card controlPanel">
      <div className="paramGrid">
        <span className="paramTitle">Parameters</span>
        {defs.map((d) => (
          <div className="paramField" key={d.key}>
            <label className="label" htmlFor={`param-${d.key}`}>
              {d.label}
            </label>
            <input
              id={`param-${d.key}`}
              className="paramInput"
              type="number"
              min={d.min}
              max={d.max}
              step={d.step}
              value={params[d.key] ?? d.default}
              onChange={(e) => handleChange(d.key, Number(e.target.value))}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
