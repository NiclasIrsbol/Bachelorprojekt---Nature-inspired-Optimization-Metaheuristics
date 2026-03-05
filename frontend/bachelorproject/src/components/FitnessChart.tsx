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

interface Coord {
  x: number;
  y: number;
}

interface FitnessChartProps {
  population: Population;
  coords?: Coord[];
}

interface ChartItem {
  index: number;
  bitstring: string;
  fitness: number;
}

const CIRCLE_SIZE = 380;
const CIRCLE_CX = CIRCLE_SIZE / 2;
const CIRCLE_CY = CIRCLE_SIZE / 2;
const CIRCLE_R = CIRCLE_SIZE / 2 - 20;

function toSvg(c: Coord) {
  return { sx: CIRCLE_CX + c.x * CIRCLE_R, sy: CIRCLE_CY - c.y * CIRCLE_R };
}

function CirclePlotInner({ coords }: { coords: Coord[] }) {
  const pts = coords.map(toSvg);
  const polyline = pts.map((p) => `${p.sx},${p.sy}`).join(" ");

  return (
    <div
      className="vizBody"
      style={{ display: "flex", alignItems: "center", justifyContent: "center" }}
    >
      <svg
        width={CIRCLE_SIZE}
        height={CIRCLE_SIZE}
        viewBox={`0 0 ${CIRCLE_SIZE} ${CIRCLE_SIZE}`}
        style={{ overflow: "visible" }}
      >
        <circle cx={CIRCLE_CX} cy={CIRCLE_CY} r={CIRCLE_R} fill="none" stroke="var(--card-border)" strokeWidth={1.5} />
        <line x1={CIRCLE_CX - CIRCLE_R} y1={CIRCLE_CY} x2={CIRCLE_CX + CIRCLE_R} y2={CIRCLE_CY} stroke="var(--card-border)" strokeWidth={0.5} strokeDasharray="4 4" />
        <line x1={CIRCLE_CX} y1={CIRCLE_CY - CIRCLE_R} x2={CIRCLE_CX} y2={CIRCLE_CY + CIRCLE_R} stroke="var(--card-border)" strokeWidth={0.5} strokeDasharray="4 4" />
        <polyline points={polyline} fill="none" stroke="var(--accent)" strokeWidth={1.5} strokeLinejoin="round" />
        {pts.map((p, i) => (
          <circle
            key={i}
            cx={p.sx}
            cy={p.sy}
            r={i === 0 ? 5 : i === pts.length - 1 ? 5 : 2.5}
            fill={i === pts.length - 1 ? "#34d399" : "var(--accent)"}
            opacity={i === 0 || i === pts.length - 1 ? 1 : 0.6}
          />
        ))}
        <text x={pts[0].sx} y={pts[0].sy - 8} textAnchor="middle" fontSize={10} fill="var(--text-muted)">start</text>
        <text x={pts[pts.length - 1].sx} y={pts[pts.length - 1].sy + 16} textAnchor="middle" fontSize={10} fill="#34d399">end</text>
      </svg>
    </div>
  );
}

export default function FitnessChart({ population, coords }: FitnessChartProps) {
  if (coords && coords.length > 0) {
    return (
      <div className="card viz">
        <div className="cardHeader">
          <h3 className="cardTitle">Visualizations</h3>
          <select className="dropDown">
          <option value="search-path">Search path</option>
          <option value="fitness-chart">Fitness-chart</option>
        </select>
        </div>
        <CirclePlotInner coords={coords} />
      </div>
    );
  }

  const entries = Object.entries(population);

  if (entries.length === 0) {
    return (
      <div className="card viz">
        <div className="cardHeader">
          <h3 className="cardTitle">Fitness Distribution</h3>
        </div>
        <div className="vizBody" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <p style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
            Single-individual algorithm — no population to display
          </p>
        </div>
      </div>
    );
  }

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
