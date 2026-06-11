import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { Prediction } from "@/lib/types";

interface ForecastChartProps {
  predictions: Prediction[];
  splitDate?: string;
}

const TICK_STYLE = {
  fontFamily: "'Geist Mono', monospace",
  fontSize: 11,
  fill: "hsl(var(--foreground-subtle))",
};

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-card border border-border rounded p-3 font-mono text-data-sm shadow-xl">
      <p className="text-foreground-muted mb-2 text-label-caps uppercase tracking-wider">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-foreground-muted">{p.name}:</span>
          <span className="text-foreground font-medium">{p.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

export function ForecastChart({ predictions, splitDate }: ForecastChartProps) {
  if (!predictions.length) {
    return (
      <div className="h-64 flex items-center justify-center text-foreground-subtle font-mono text-data-sm">
        No predictions available
      </div>
    );
  }

  const data = predictions.map((p) => ({
    date: p.timestamp.slice(0, 10),
    forecast: p.value,
    lower: p.lower_ci,
    upper: p.upper_ci,
  }));

  const tickCount = Math.min(data.length, 8);
  const step = Math.floor(data.length / tickCount);
  const ticks = data.filter((_, i) => i % step === 0).map((d) => d.date);

  return (
    <ResponsiveContainer width="100%" height={320}>
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
          width={48}
          tickFormatter={(v) => v.toFixed(2)}
        />
        <Tooltip content={<CustomTooltip />} />
        {splitDate && (
          <ReferenceLine
            x={splitDate}
            stroke="#3F3F46"
            strokeDasharray="4 4"
            label={{ value: "split", fill: "hsl(var(--foreground-subtle))", fontSize: 10, fontFamily: "Geist Mono" }}
          />
        )}
        {/* Confidence interval as thin lines */}
        {predictions[0]?.lower_ci !== null && (
          <>
            <Line
              dataKey="lower"
              stroke="#8B5CF6"
              strokeWidth={1}
              strokeOpacity={0.35}
              dot={false}
              name="Lower CI"
              isAnimationActive
              animationBegin={600}
              animationDuration={1200}
              animationEasing="ease-out"
            />
            <Line
              dataKey="upper"
              stroke="#8B5CF6"
              strokeWidth={1}
              strokeOpacity={0.35}
              dot={false}
              name="Upper CI"
              isAnimationActive
              animationBegin={600}
              animationDuration={1200}
              animationEasing="ease-out"
            />
          </>
        )}
        <Line
          dataKey="forecast"
          stroke="#8B5CF6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#8B5CF6" }}
          name="Forecast"
          isAnimationActive
          animationBegin={350}
          animationDuration={1800}
          animationEasing="ease-out"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
