import { useEffect, useMemo, useState } from "react";

type ParamValue = number | string;

interface ParamDef {
  key: string;
  label: string;
  default: ParamValue;
  // Numeric inputs use min/max/step; enum inputs use options (rendered as a select).
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
}

// TSP local-search mutation operator selector. Only the trajectory/EA solvers
// take a mutation operator; the ant-based methods construct tours from pheromone
// and ignore it, so it is not offered for MMAS-ACO / P-ACO.
const TSP_MUTATION_PARAM: ParamDef = {
  key: "mutation",
  label: "Mutation operator",
  default: "2opt",
  options: [
    { value: "2opt", label: "2-opt" },
    { value: "3opt", label: "3-opt" },
  ],
};

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

  "MMAS-ACO": [
    { key: "bit_length", label: "Bit length", default: 100, min: 2, max: 1000, step: 1 },
    { key: "rho", label: "Evaporation (ρ)", default: 0.1, min: 0.01, max: 1, step: 0.01 },
    { key: "max_iterations", label: "Max iterations", default: 10000, min: 100, max: 100000, step: 100 },
  ],

  "P-ACO": [
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
    TSP_MUTATION_PARAM,
  ],

  "(μ+λ) EA": [
    { key: "max_iterations", label: "Max iterations", default: 5000, min: 100, max: 50000, step: 100 },
    { key: "mu_size", label: "μ (parents)", default: 20, min: 2, max: 200, step: 1 },
    { key: "lambda_size", label: "λ (offspring)", default: 40, min: 2, max: 400, step: 1 },
    { key: "tournament_k", label: "Tournament k", default: 3, min: 2, max: 20, step: 1 },
    TSP_MUTATION_PARAM,
  ],

  "Simulated Annealing": [
    { key: "max_iterations", label: "Max iterations", default: 100000, min: 1000, max: 500000, step: 1000 },
    { key: "cooling", label: "Cooling rate", default: 0.9995, min: 0.9, max: 0.99999, step: 0.0001 },
    { key: "T0", label: "Initial temp (T₀)", default: 1000, min: 1, max: 100000, step: 1 },
    TSP_MUTATION_PARAM,
  ],

  "MMAS-ACO": [
    { key: "max_iterations", label: "Max iterations", default: 1000, min: 100, max: 50000, step: 100 },
    { key: "rho", label: "Evaporation (ρ)", default: 0.1, min: 0.01, max: 1, step: 0.01 },
    { key: "alpha", label: "α (pheromone)", default: 1, min: 0.1, max: 5, step: 0.1 },
    { key: "beta", label: "β (heuristic)", default: 2, min: 0.1, max: 10, step: 0.1 },
  ],

  "P-ACO": [
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

export function getDefaultParams(algorithm: string, problem = "onemax"): Record<string, ParamValue> {
  const defs = getParamDefs(algorithm, problem);
  const out: Record<string, ParamValue> = {};
  for (const d of defs) out[d.key] = d.default;
  return out;
}

interface ParametersPanelProps {
  algorithm: string;
  problem: string;
  params: Record<string, ParamValue>;
  onChange: (params: Record<string, ParamValue>) => void;
}

export default function ParametersPanel({
  algorithm,
  problem,
  params,
  onChange,
}: ParametersPanelProps) {
  const defs = useMemo(() => getParamDefs(algorithm, problem), [algorithm, problem]);

  // Raw text for number fields while they are being edited. This lets a field be
  // momentarily empty (so deleting a value does not snap it to 0) while the
  // parent `params` stays numeric. Reset whenever the field set changes.
  const [draft, setDraft] = useState<Record<string, string>>({});

  useEffect(() => {
    onChange(getDefaultParams(algorithm, problem));
    setDraft({});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithm, problem]);

  if (defs.length === 0) return null;

  const handleChange = (key: string, value: ParamValue) => {
    onChange({ ...params, [key]: value });
  };

  const handleNumberInput = (key: string, raw: string) => {
    setDraft((prev) => ({ ...prev, [key]: raw }));
    // Only push a real number to the parent; an empty/partial entry stays local.
    if (raw !== "" && !Number.isNaN(Number(raw))) {
      handleChange(key, Number(raw));
    }
  };

  const handleNumberBlur = (def: ParamDef) => {
    const raw = draft[def.key];
    // Left empty or invalid on blur -> fall back to the default.
    if (raw === "" || (raw !== undefined && Number.isNaN(Number(raw)))) {
      handleChange(def.key, def.default);
    }
    setDraft((prev) => {
      const next = { ...prev };
      delete next[def.key];
      return next;
    });
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
            {d.options ? (
              <select
                id={`param-${d.key}`}
                className="paramInput"
                value={String(params[d.key] ?? d.default)}
                onChange={(e) => handleChange(d.key, e.target.value)}
              >
                {d.options.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                id={`param-${d.key}`}
                className="paramInput"
                type="number"
                min={d.min}
                max={d.max}
                step={d.step}
                value={d.key in draft ? draft[d.key] : (params[d.key] ?? d.default)}
                onChange={(e) => handleNumberInput(d.key, e.target.value)}
                onBlur={() => handleNumberBlur(d)}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
