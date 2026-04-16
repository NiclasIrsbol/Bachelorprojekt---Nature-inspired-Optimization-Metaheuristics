import { useEffect, useMemo } from "react";

interface ParamDef {
  key: string;
  label: string;
  default: number;
  min: number;
  max: number;
  step: number;
}

const BITSTRING_PARAMS: Record<string, ParamDef[]> = {
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

  "PACO": [
    { key: "bit_length", label: "Bit length", default: 100, min: 2, max: 1000, step: 1 },
    { key: "archive_size", label: "Archive size (K)", default: 10, min: 2, max: 100, step: 1 },
    { key: "num_ants", label: "Number of ants", default: 30, min: 1, max: 200, step: 1 },
    { key: "q0", label: "Greedy prob (q₀)", default: 0.9, min: 0, max: 1, step: 0.05 },
    { key: "max_iterations", label: "Max iterations", default: 10000, min: 100, max: 100000, step: 100 },
  ],
};

const TSP_PARAMS: Record<string, ParamDef[]> = {
  "(1+1) EA": [
    { key: "max_iterations", label: "Max iterations", default: 10000, min: 100, max: 200000, step: 100 },
  ],

  "(μ+λ) EA": [
    { key: "max_iterations", label: "Max iterations", default: 5000, min: 100, max: 50000, step: 100 },
    { key: "mu_size", label: "μ (parents)", default: 20, min: 2, max: 200, step: 1 },
    { key: "lambda_size", label: "λ (offspring)", default: 40, min: 2, max: 400, step: 1 },
    { key: "tournament_k", label: "Tournament k", default: 3, min: 2, max: 20, step: 1 },
  ],

  "Simulated Annealing": [
    { key: "max_iterations", label: "Max iterations", default: 100000, min: 1000, max: 500000, step: 1000 },
    { key: "cooling", label: "Cooling rate", default: 0.9995, min: 0.9, max: 0.99999, step: 0.0001 },
    { key: "T0", label: "Initial temp (T₀)", default: 1000, min: 1, max: 100000, step: 1 },
  ],

  "ACO": [
    { key: "max_iterations", label: "Max iterations", default: 1000, min: 100, max: 50000, step: 100 },
    { key: "rho", label: "Evaporation (ρ)", default: 0.1, min: 0.01, max: 1, step: 0.01 },
    { key: "alpha", label: "α (pheromone)", default: 1, min: 0.1, max: 5, step: 0.1 },
    { key: "beta", label: "β (heuristic)", default: 2, min: 0.1, max: 10, step: 0.1 },
  ],

  "PACO": [
    { key: "max_iterations", label: "Max iterations", default: 1000, min: 100, max: 50000, step: 100 },
    { key: "archive_size", label: "Archive size (K)", default: 10, min: 2, max: 100, step: 1 },
    { key: "num_ants", label: "Number of ants", default: 30, min: 1, max: 200, step: 1 },
    { key: "alpha", label: "α (pheromone)", default: 1, min: 0.1, max: 5, step: 0.1 },
    { key: "beta", label: "β (heuristic)", default: 2, min: 0.1, max: 10, step: 0.1 },
    { key: "q0", label: "Greedy prob (q₀)", default: 0.9, min: 0, max: 1, step: 0.05 },
  ],
};

function getParamDefs(algorithm: string, problem: string): ParamDef[] {
  if (problem === "tsp") {
    return TSP_PARAMS[algorithm] ?? [];
  }
  return BITSTRING_PARAMS[algorithm] ?? [];
}

export function getDefaultParams(algorithm: string, problem = "onemax"): Record<string, number> {
  const defs = getParamDefs(algorithm, problem);
  const out: Record<string, number> = {};
  for (const d of defs) out[d.key] = d.default;
  return out;
}

interface ParametersPanelProps {
  algorithm: string;
  problem: string;
  params: Record<string, number>;
  onChange: (params: Record<string, number>) => void;
}

export default function ParametersPanel({
  algorithm,
  problem,
  params,
  onChange,
}: ParametersPanelProps) {
  const defs = useMemo(() => getParamDefs(algorithm, problem), [algorithm, problem]);

  useEffect(() => {
    onChange(getDefaultParams(algorithm, problem));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithm, problem]);

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
