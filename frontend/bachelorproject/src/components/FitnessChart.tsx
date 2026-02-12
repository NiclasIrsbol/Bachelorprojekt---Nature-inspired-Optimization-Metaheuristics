import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Individual {
  bit: string;
  fitness: number;
}

interface Population {
  [key: string]: Individual;
}

interface FitnessChartProps {
  population: Population;
}

interface ChartItem {
  index: number;
  bitstring: string;
  fitness: number;
}

export default function FitnessChart({ population }: FitnessChartProps) {
  const entries = Object.entries(population);
  const maxFitness = Math.max(...entries.map(([, v]) => v.fitness));

  const data: ChartItem[] = entries.map(([key, val], i) => ({
    index: i,
    bitstring: val.bit,
    fitness: val.fitness,
    name: key,
  }));

  return (
    <div className="card viz">
      <div className="cardHeader">
        <h3 className="cardTitle">Fitness Distribution</h3>
        <span className="pill mono">Gen 1</span>
      </div>
      <div className="vizBody">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 8, right: 12, bottom: 24, left: 0 }}
          >
            <XAxis
              dataKey="index"
              tick={{ fontSize: 11 }}
              label={{
                value: "Individual",
                position: "insideBottom",
                offset: -16,
                fontSize: 12,
              }}
            />
            <YAxis
              domain={[0, "dataMax + 1"]}
              tick={{ fontSize: 11 }}
              label={{
                value: "Fitness",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                fontSize: 12,
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload as ChartItem;
                return (
                  <div
                    style={{
                      background: "var(--bg-surface)",
                      border: "1px solid var(--card-border)",
                      borderRadius: 8,
                      padding: "8px 12px",
                      fontSize: "0.85rem",
                    }}
                  >
                    <div style={{ fontWeight: 600 }}>
                      Individual {d.index}
                    </div>
                    <div style={{ fontFamily: "monospace", marginTop: 2 }}>
                      {d.bitstring}
                    </div>
                    <div style={{ marginTop: 2 }}>
                      Fitness: <strong>{d.fitness}</strong>
                    </div>
                  </div>
                );
              }}
            />
            <Bar dataKey="fitness" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={
                    entry.fitness === maxFitness
                      ? "var(--bar-fill-best)"
                      : "var(--bar-fill)"
                  }
                  opacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
