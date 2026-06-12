// @authors: Andrej Kitanovski and Niclas Søe Irsbøl

import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface CurvePoint {
  evaluations: number;
  value: number;
}

export interface CompareAlgo {
  algorithm: string;
  mean: number;
  std: number;
  best: number;
  worst: number;
  median: number;
  success_rate?: number;
  gap_percent?: number | null;
  curve: CurvePoint[];
}

export interface CompareData {
  problem: string;
  metric: string;
  bit_length?: number;
  tsp_instance?: string;
  optimum?: number | null;
  algorithms: CompareAlgo[];
}

interface ScalingPoint {
  size: number;
  value: number;
}

export interface ScalingSeries {
  algorithm: string;
  points: ScalingPoint[];
}

export interface TheorySeries {
  label: string;
  points: ScalingPoint[];
}

export interface ScalingData {
  problem: string;
  y_metric: string;
  series: ScalingSeries[];
  theory?: TheorySeries[];
}

const ALGO_COLORS: Record<string, string> = {
  "(1+1) EA": "#60a5fa",
  "(μ+λ) EA": "#f59e0b",
  "Simulated Annealing": "#a78bfa",
  "MMAS-ACO": "#34d399",
  ACO: "#34d399",
  "P-ACO": "#f472b6",
};

function colorFor(algo: string): string {
  return ALGO_COLORS[algo] ?? "#94a3b8";
}

const THEORY_COLOR = "#64748b";

// Shared layout so the convergence and scaling charts stay visually consistent.
const CHART_MARGIN = { top: 8, right: 24, bottom: 36, left: 16 } as const;

const TOOLTIP_STYLE = {
  background: "var(--bg-surface)",
  border: "1px solid var(--card-border)",
  borderRadius: 8,
  fontSize: "0.85rem",
} as const;

function formatNumber(n: number): string {
  if (!Number.isFinite(n)) return "-";
  return Math.abs(n) >= 1000
    ? Math.round(n).toLocaleString()
    : (Math.round(n * 100) / 100).toString();
}

interface LegendItem {
  label: string;
  color: string;
  dashed?: boolean;
  selected?: boolean;
}

// Custom HTML legend rendered below the plot. Keeping it out of the SVG avoids
// the axis-label/legend collisions that the inline Recharts legend caused when
// many series (algorithms + theory curves) are shown at once.
function ChartLegend({ items }: { items: LegendItem[] }) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "6px 16px",
        justifyContent: "center",
        padding: "10px 12px 4px",
        fontSize: "0.8rem",
      }}
    >
      {items.map((it) => (
        <span
          key={it.label}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            opacity: it.selected === false ? 0.7 : 1,
            fontWeight: it.selected ? 700 : 400,
            color: "var(--text)",
          }}
        >
          <span
            style={{
              display: "inline-block",
              width: 18,
              height: 0,
              borderTop: `${it.dashed ? 2 : 3}px ${it.dashed ? "dashed" : "solid"} ${it.color}`,
            }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

// Merge per-algorithm convergence curves (each indexed by fitness evaluations)
// onto a shared x-grid using forward-fill (a converged algorithm holds its
// final value; before its first point the value is null so the line starts later).
function buildConvergenceRows(algos: CompareAlgo[]) {
  const xset = new Set<number>();
  algos.forEach((a) => a.curve.forEach((p) => xset.add(p.evaluations)));
  const xs = Array.from(xset).sort((a, b) => a - b);

  const pointers = algos.map(() => 0);
  const last = algos.map(() => null as number | null);
  const started = algos.map(() => false);

  return xs.map((x) => {
    const row: Record<string, number | null> = { x };
    algos.forEach((a, ai) => {
      while (pointers[ai] < a.curve.length && a.curve[pointers[ai]].evaluations <= x) {
        last[ai] = a.curve[pointers[ai]].value;
        started[ai] = true;
        pointers[ai]++;
      }
      row[a.algorithm] = started[ai] ? last[ai] : null;
    });
    return row;
  });
}

function buildScalingRows(series: ScalingSeries[], theory: TheorySeries[] = []) {
  const sizeSet = new Set<number>();
  series.forEach((s) => s.points.forEach((p) => sizeSet.add(p.size)));
  theory.forEach((t) => t.points.forEach((p) => sizeSet.add(p.size)));
  const sizes = Array.from(sizeSet).sort((a, b) => a - b);
  return sizes.map((size) => {
    const row: Record<string, number | null> = { size };
    series.forEach((s) => {
      const pt = s.points.find((p) => p.size === size);
      row[s.algorithm] = pt ? pt.value : null;
    });
    theory.forEach((t) => {
      const pt = t.points.find((p) => p.size === size);
      row[t.label] = pt ? pt.value : null;
    });
    return row;
  });
}

interface ComparisonPanelProps {
  mode: "convergence" | "scaling";
  compareData?: CompareData | null;
  scalingData?: ScalingData | null;
  selectedAlgorithm: string;
}

export default function ComparisonPanel({
  mode,
  compareData,
  scalingData,
  selectedAlgorithm,
}: ComparisonPanelProps) {
  if (mode === "convergence" && compareData) {
    const algos = compareData.algorithms;
    const rows = buildConvergenceRows(algos);
    const isTsp = compareData.problem === "tsp";
    const yLabel = isTsp ? "Best cost" : "Best fitness";
    const metricLabel = isTsp ? "Best cost" : "Evaluations to optimum";
    const titleSuffix = isTsp
      ? `TSP ${compareData.tsp_instance ?? ""}`
      : `${compareData.problem}, n=${compareData.bit_length}`;

    return (
      <div className="card viz">
        <div className="cardHeader">
          <h3 className="cardTitle">Convergence comparison ({titleSuffix})</h3>
        </div>
        <div className="vizBody">
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={rows} margin={CHART_MARGIN}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--card-border)" />
              <XAxis
                dataKey="x"
                type="number"
                domain={["dataMin", "dataMax"]}
                tick={{ fontSize: 11 }}
                tickFormatter={formatNumber}
                height={48}
                tickMargin={8}
                label={{ value: "Fitness evaluations", position: "insideBottom", offset: 0, fontSize: 12 }}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                width={72}
                tickMargin={6}
                tickFormatter={formatNumber}
                label={{ value: yLabel, angle: -90, position: "insideLeft", offset: -4, fontSize: 12, style: { textAnchor: "middle" } }}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(value, name) => [formatNumber(Number(value)), name]}
                labelFormatter={(label) => `${formatNumber(Number(label))} evaluations`}
              />
              {algos.map((a) => (
                <Line
                  key={a.algorithm}
                  type="monotone"
                  dataKey={a.algorithm}
                  stroke={colorFor(a.algorithm)}
                  strokeWidth={a.algorithm === selectedAlgorithm ? 3.5 : 1.5}
                  opacity={a.algorithm === selectedAlgorithm ? 1 : 0.7}
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <ChartLegend
            items={algos.map((a) => ({
              label: a.algorithm,
              color: colorFor(a.algorithm),
              selected: a.algorithm === selectedAlgorithm,
            }))}
          />
        </div>
        <StatsTable algos={algos} selected={selectedAlgorithm} metricLabel={metricLabel} isTsp={isTsp} />
      </div>
    );
  }

  if (mode === "scaling" && scalingData) {
    const theory = scalingData.theory ?? [];
    const rows = buildScalingRows(scalingData.series, theory);
    const yLabel =
      scalingData.y_metric === "iterations" ? "Average iterations" : "Average fitness evaluations";

    return (
      <div className="card viz">
        <div className="cardHeader">
          <h3 className="cardTitle">
            Performance comparison ({scalingData.problem}, {yLabel.toLowerCase()} vs. size)
          </h3>
        </div>
        <div className="vizBody">
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={rows} margin={CHART_MARGIN}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--card-border)" />
              <XAxis
                dataKey="size"
                type="number"
                domain={[0, "dataMax"]}
                tick={{ fontSize: 11 }}
                tickFormatter={formatNumber}
                height={48}
                tickMargin={8}
                label={{ value: "Problem size n", position: "insideBottom", offset: 0, fontSize: 12 }}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                width={72}
                tickMargin={6}
                tickFormatter={formatNumber}
                label={{ value: yLabel, angle: -90, position: "insideLeft", offset: -4, fontSize: 12, style: { textAnchor: "middle" } }}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(value, name) => [formatNumber(Number(value)), name]}
                labelFormatter={(label) => `n = ${formatNumber(Number(label))}`}
              />
              {scalingData.series.map((s) => (
                <Line
                  key={s.algorithm}
                  type="monotone"
                  dataKey={s.algorithm}
                  stroke={colorFor(s.algorithm)}
                  strokeWidth={s.algorithm === selectedAlgorithm ? 3.5 : 1.5}
                  opacity={s.algorithm === selectedAlgorithm ? 1 : 0.7}
                  dot={{ r: 2 }}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
              {theory.map((t) => (
                <Line
                  key={t.label}
                  type="monotone"
                  dataKey={t.label}
                  stroke={THEORY_COLOR}
                  strokeWidth={1.5}
                  strokeDasharray="6 4"
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <ChartLegend
            items={[
              ...scalingData.series.map((s) => ({
                label: s.algorithm,
                color: colorFor(s.algorithm),
                selected: s.algorithm === selectedAlgorithm,
              })),
              ...theory.map((t) => ({
                label: t.label,
                color: THEORY_COLOR,
                dashed: true,
              })),
            ]}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="card viz">
      <div className="vizBody" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
        <p style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
          Click Run to compute the comparison.
        </p>
      </div>
    </div>
  );
}

function StatsTable({
  algos,
  selected,
  metricLabel,
  isTsp,
}: {
  algos: CompareAlgo[];
  selected: string;
  metricLabel: string;
  isTsp: boolean;
}) {
  const fmt = (n: number) =>
    n >= 1000 ? n.toLocaleString(undefined, { maximumFractionDigits: 0 }) : n.toFixed(1);

  return (
    <div style={{ padding: "0 12px 12px", overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
        <thead>
          <tr style={{ color: "var(--text-muted)", textAlign: "right" }}>
            <th style={{ textAlign: "left", padding: "6px 8px" }}>Algorithm ({metricLabel})</th>
            <th style={{ padding: "6px 8px" }}>Mean</th>
            <th style={{ padding: "6px 8px" }}>Std</th>
            <th style={{ padding: "6px 8px" }}>Best</th>
            <th style={{ padding: "6px 8px" }}>Worst</th>
            <th style={{ padding: "6px 8px" }}>{isTsp ? "Gap %" : "Success"}</th>
          </tr>
        </thead>
        <tbody>
          {algos.map((a) => {
            const isSel = a.algorithm === selected;
            return (
              <tr
                key={a.algorithm}
                style={{
                  textAlign: "right",
                  fontWeight: isSel ? 700 : 400,
                  background: isSel ? "var(--bg-surface)" : "transparent",
                }}
              >
                <td style={{ textAlign: "left", padding: "6px 8px" }}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background: colorFor(a.algorithm),
                      marginRight: 8,
                    }}
                  />
                  {a.algorithm}
                </td>
                <td style={{ padding: "6px 8px" }}>{fmt(a.mean)}</td>
                <td style={{ padding: "6px 8px" }}>{fmt(a.std)}</td>
                <td style={{ padding: "6px 8px" }}>{fmt(a.best)}</td>
                <td style={{ padding: "6px 8px" }}>{fmt(a.worst)}</td>
                <td style={{ padding: "6px 8px" }}>
                  {isTsp
                    ? a.gap_percent == null
                      ? "-"
                      : `${a.gap_percent.toFixed(1)}%`
                    : a.success_rate == null
                    ? "-"
                    : `${Math.round(a.success_rate * 100)}%`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
