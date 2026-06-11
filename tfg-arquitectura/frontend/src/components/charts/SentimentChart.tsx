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

interface SignalRow {
  year_month: string;
  sentiment_mean?: number | null;
  sentiment_hawkish?: number | null;
  [key: string]: unknown;
}

interface SentimentChartProps {
  signals: SignalRow[];
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
      {payload.map((p) => (
        p.value !== null && p.value !== undefined ? (
          <div key={p.name} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-sm" style={{ background: p.color }} />
            <span className="text-foreground-muted">{p.name}:</span>
            <span className="text-foreground font-medium">{p.value?.toFixed(3)}</span>
          </div>
        ) : null
      ))}
    </div>
  );
}

export function SentimentChart({ signals }: SentimentChartProps) {
  const data = signals
    .filter((s) => s.sentiment_mean !== undefined && s.sentiment_mean !== null)
    .map((s) => ({
      month: s.year_month,
      sentiment: typeof s.sentiment_mean === "number" ? s.sentiment_mean : null,
      hawkish: typeof s.sentiment_hawkish === "number" ? s.sentiment_hawkish : null,
    }));

  if (!data.length) {
    return (
      <div className="h-40 flex items-center justify-center font-mono text-data-sm text-foreground-subtle">
        No sentiment data in signals
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="month"
          tick={TICK_STYLE}
          axisLine={{ stroke: "#3F3F46" }}
          tickLine={false}
        />
        <YAxis
          yAxisId="sent"
          domain={[-1, 1]}
          tick={TICK_STYLE}
          axisLine={false}
          tickLine={false}
          width={40}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <YAxis
          yAxisId="hawk"
          orientation="right"
          domain={[0, 1]}
          tick={TICK_STYLE}
          axisLine={false}
          tickLine={false}
          width={40}
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontFamily: "'Geist Mono', monospace", fontSize: 11, color: "hsl(var(--foreground-muted))" }}
        />
        <ReferenceLine
          yAxisId="sent"
          y={0}
          stroke="#3F3F46"
          strokeDasharray="4 4"
        />
        <Line
          yAxisId="sent"
          dataKey="sentiment"
          stroke="#8B5CF6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#8B5CF6" }}
          name="Sentiment"
          connectNulls
          isAnimationActive
          animationBegin={200}
          animationDuration={1200}
          animationEasing="ease-out"
        />
        <Line
          yAxisId="hawk"
          dataKey="hawkish"
          stroke="#FB7185"
          strokeWidth={1.5}
          strokeDasharray="4 2"
          dot={false}
          activeDot={{ r: 3, fill: "#FB7185" }}
          name="Hawkish %"
          connectNulls
          isAnimationActive
          animationBegin={400}
          animationDuration={1200}
          animationEasing="ease-out"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
