import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { modelColor } from "@/lib/modelColors";
import type { ComparisonData } from "@/hooks/useComparisonData";

interface HorizonDecayProps {
  data: ComparisonData;
  hoveredModel: string | null;
  onHoverModel: (slug: string | null) => void;
}

const TICK_STYLE = { fontFamily: "'Geist Mono', monospace", fontSize: 11, fill: "hsl(var(--foreground-subtle))" };

function isoMonth(iso: string): string {
  return iso.slice(0, 7);
}

export function HorizonDecay({ data, hoveredModel, onHoverModel }: HorizonDecayProps) {
  const { rows, slugs, hasHorizon } = useMemo(() => {
    const runs = data.runs.filter((r) => r.runId > 0 && data.predictionsByRun[r.runId]?.length);
    const hasH = runs.some((r) => r.horizon > 1);
    if (!hasH) return { rows: [], slugs: [], hasHorizon: false };

    // pre-index actuals by month per series
    const actualsByMonthPerSeries = new Map<number, Map<string, number>>();
    Object.entries(data.actualsBySeries).forEach(([sid, obs]) => {
      const m = new Map<string, number>();
      obs.forEach((o) => m.set(isoMonth(o.timestamp), o.value));
      actualsByMonthPerSeries.set(Number(sid), m);
    });

    const maxSteps = Math.max(...runs.map((r) => data.predictionsByRun[r.runId].length));
    const rows: Array<Record<string, number | string | null>> = [];

    for (let step = 1; step <= maxSteps; step++) {
      const row: Record<string, number | string | null> = { step };
      runs.forEach((r) => {
        const preds = data.predictionsByRun[r.runId];
        const monthMap = actualsByMonthPerSeries.get(r.seriesId);
        if (step - 1 >= preds.length || !monthMap) {
          row[r.modelSlug] = null;
          return;
        }
        const p = preds[step - 1];
        const actual = monthMap.get(isoMonth(p.timestamp));
        row[r.modelSlug] = actual === undefined ? null : Math.abs(actual - p.value);
      });
      rows.push(row);
    }

    return { rows, slugs: runs.map((r) => r.modelSlug), hasHorizon: true };
  }, [data]);

  if (!hasHorizon) {
    return (
      <div className="font-mono text-data-sm text-foreground-subtle py-10 text-center">
        Horizon decay needs runs with horizon &gt; 1.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart
        data={rows}
        margin={{ top: 8, right: 24, bottom: 0, left: 0 }}
        onMouseLeave={() => onHoverModel(null)}
      >
        <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="step"
          tick={TICK_STYLE}
          axisLine={{ stroke: "#3F3F46" }}
          tickLine={false}
          label={{ value: "step ahead", position: "insideBottom", offset: -2, fill: "hsl(var(--foreground-subtle))", fontSize: 10, fontFamily: "Geist Mono" }}
        />
        <YAxis
          tick={TICK_STYLE}
          axisLine={false}
          tickLine={false}
          width={52}
          tickFormatter={(v) => Number(v).toFixed(2)}
        />
        <Tooltip
          contentStyle={{
            background: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: 4,
            fontFamily: "Geist Mono",
            fontSize: 12,
          }}
          labelStyle={{ color: "hsl(var(--foreground-muted))" }}
          cursor={{ stroke: "#3F3F46", strokeDasharray: "3 3" }}
          formatter={(v: number) => v.toFixed(4)}
          labelFormatter={(label) => `step ${label}`}
        />
        {slugs.map((slug, i) => {
          const isHovered = hoveredModel === slug;
          const isDimmed = hoveredModel !== null && !isHovered;
          return (
            <Line
              key={slug}
              type="monotone"
              dataKey={slug}
              stroke={modelColor(slug, i)}
              strokeWidth={isHovered ? 2.5 : 1.5}
              strokeOpacity={isDimmed ? 0.18 : 1}
              dot={false}
              activeDot={{ r: 3, fill: modelColor(slug, i) }}
              isAnimationActive
              animationBegin={200 + i * 100}
              animationDuration={1100}
              connectNulls
            />
          );
        })}
      </LineChart>
    </ResponsiveContainer>
  );
}
