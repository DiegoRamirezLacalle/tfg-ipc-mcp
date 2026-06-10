import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import type { WhatifPoint } from "@/lib/types";

interface WhatifChartProps {
  history: WhatifPoint[];
  baseline: WhatifPoint[];
  counterfactual: number[];
  unit?: string | null;
}

const TICK_STYLE = {
  fontFamily: "'Geist Mono', monospace",
  fontSize: 11,
  fill: "hsl(var(--foreground-subtle))",
};

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number | null; color: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-card border border-border rounded p-3 font-mono text-data-sm shadow-xl">
      <p className="text-foreground-muted mb-2 micro uppercase">{label}</p>
      {payload.map((p) =>
        p.value !== null && p.value !== undefined ? (
          <div key={p.name} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-sm" style={{ background: p.color }} />
            <span className="text-foreground-muted">{p.name}:</span>
            <span className="text-foreground font-medium">{p.value?.toFixed(3)}</span>
          </div>
        ) : null
      )}
    </div>
  );
}

export function WhatifChart({ history, baseline, counterfactual, unit }: WhatifChartProps) {
  if (!baseline.length) {
    return (
      <div className="h-80 flex items-center justify-center font-mono text-data-sm text-foreground-subtle">
        No forecast available
      </div>
    );
  }

  const histRows = history.map((h) => ({
    date: h.timestamp.slice(0, 7),
    actual: h.value,
    baseline: null as number | null,
    counterfactual: null as number | null,
  }));

  // Connector: anchor the forecast lines at the last observed point
  if (histRows.length) {
    const last = histRows[histRows.length - 1];
    last.baseline = last.actual;
    last.counterfactual = last.actual;
  }

  const futureRows = baseline.map((b, i) => ({
    date: b.timestamp.slice(0, 7),
    actual: null as number | null,
    baseline: b.value,
    counterfactual: counterfactual[i] ?? b.value,
  }));

  const data = [...histRows, ...futureRows];
  const splitDate = baseline[0]?.timestamp.slice(0, 7);

  const tickCount = Math.min(data.length, 9);
  const step = Math.max(1, Math.floor(data.length / tickCount));
  const ticks = data.filter((_, i) => i % step === 0).map((d) => d.date);

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="date"
          ticks={ticks}
          tick={TICK_STYLE}
          axisLine={{ stroke: "#3F3F46" }}
          tickLine={false}
        />
        <YAxis
          tick={TICK_STYLE}
          axisLine={false}
          tickLine={false}
          width={52}
          tickFormatter={(v: number) => v.toFixed(1)}
          label={
            unit
              ? { value: unit, angle: -90, position: "insideLeft", fill: "hsl(var(--foreground-subtle))", fontSize: 10, fontFamily: "Geist Mono" }
              : undefined
          }
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontFamily: "'Geist Mono', monospace", fontSize: 11, color: "hsl(var(--foreground-muted))" }} />
        {splitDate && (
          <ReferenceLine
            x={splitDate}
            stroke="#3F3F46"
            strokeDasharray="4 4"
            label={{ value: "forecast →", fill: "hsl(var(--foreground-subtle))", fontSize: 10, fontFamily: "Geist Mono", position: "insideTopRight" }}
          />
        )}
        <Line
          dataKey="actual"
          stroke="hsl(var(--foreground-subtle))"
          strokeWidth={1.5}
          dot={false}
          name="History"
          connectNulls
          isAnimationActive={false}
        />
        <Line
          dataKey="baseline"
          stroke="#8B5CF6"
          strokeWidth={2}
          strokeDasharray="5 3"
          dot={false}
          activeDot={{ r: 4, fill: "#8B5CF6" }}
          name="Baseline"
          connectNulls
          isAnimationActive={false}
        />
        <Line
          dataKey="counterfactual"
          stroke="#E0B96A"
          strokeWidth={2.5}
          dot={false}
          activeDot={{ r: 4, fill: "#E0B96A" }}
          name="Counterfactual"
          connectNulls
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
