import { useMemo } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  Cell,
} from "recharts";

import { modelColor } from "@/lib/modelColors";
import type { ComparisonRunData } from "@/hooks/useComparisonData";

interface ParetoScatterProps {
  runs: ComparisonRunData[];
  hoveredModel: string | null;
  onHoverModel: (slug: string | null) => void;
}

const TICK_STYLE = { fontFamily: "'Geist Mono', monospace", fontSize: 11, fill: "hsl(var(--foreground-subtle))" };

export function ParetoScatter({ runs, hoveredModel, onHoverModel }: ParetoScatterProps) {
  const points = useMemo(
    () =>
      runs
        .filter((r) => r.durationSec !== null && r.mae !== null)
        .map((r) => ({
          x: r.durationSec as number,
          y: r.mae as number,
          z: r.mape ?? 0,
          slug: r.modelSlug,
        })),
    [runs]
  );

  if (points.length === 0) {
    return (
      <div className="font-mono text-data-sm text-foreground-subtle py-10 text-center">
        Need runs with recorded start &amp; finish times to plot runtime vs accuracy.
      </div>
    );
  }

  const maxMape = Math.max(...points.map((p) => p.z), 1);

  return (
    <div className="w-full relative">
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart
          margin={{ top: 16, right: 28, bottom: 24, left: 4 }}
          onMouseLeave={() => onHoverModel(null)}
        >
          <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name="runtime"
            unit="s"
            tick={TICK_STYLE}
            axisLine={{ stroke: "#3F3F46" }}
            tickLine={false}
            label={{ value: "runtime (s)", position: "insideBottom", offset: -8, fill: "hsl(var(--foreground-subtle))", fontSize: 10, fontFamily: "Geist Mono" }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="MAE"
            tick={TICK_STYLE}
            axisLine={{ stroke: "#3F3F46" }}
            tickLine={false}
            width={52}
            tickFormatter={(v) => Number(v).toFixed(2)}
            label={{ value: "MAE", angle: -90, position: "insideLeft", fill: "hsl(var(--foreground-subtle))", fontSize: 10, fontFamily: "Geist Mono" }}
          />
          <ZAxis type="number" dataKey="z" range={[60, 280]} name="MAPE" />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 4,
              fontFamily: "Geist Mono",
              fontSize: 12,
            }}
            labelStyle={{ color: "hsl(var(--foreground-muted))" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as { slug: string; x: number; y: number; z: number };
              return (
                <div className="bg-card border border-border rounded p-3 font-mono text-data-sm">
                  <p style={{ color: modelColor(p.slug, 0) }} className="font-medium mb-1">{p.slug}</p>
                  <div className="text-foreground-muted">runtime: <span className="text-foreground">{p.x.toFixed(1)}s</span></div>
                  <div className="text-foreground-muted">MAE: <span className="text-foreground">{p.y.toFixed(4)}</span></div>
                  <div className="text-foreground-muted">MAPE: <span className="text-foreground">{p.z.toFixed(2)}%</span></div>
                </div>
              );
            }}
          />
          <Scatter data={points} isAnimationActive animationDuration={700}>
            {points.map((p, i) => {
              const isHovered = hoveredModel === p.slug;
              const isDimmed = hoveredModel !== null && !isHovered;
              return (
                <Cell
                  key={i}
                  fill={modelColor(p.slug, i)}
                  fillOpacity={isDimmed ? 0.2 : 0.8}
                  stroke={modelColor(p.slug, i)}
                  strokeWidth={isHovered ? 2 : 1}
                  onMouseEnter={() => onHoverModel(p.slug)}
                />
              );
            })}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>

      {/* Quadrant labels */}
      <div className="pointer-events-none absolute top-3 right-4 font-mono text-[10px] uppercase tracking-widest text-destructive/60">
        slow &amp; inaccurate
      </div>
      <div className="pointer-events-none absolute bottom-9 left-3 font-mono text-[10px] uppercase tracking-widest text-success/70">
        fast &amp; accurate
      </div>

      <p className="font-mono text-[10px] text-foreground-subtle uppercase tracking-widest text-center pt-1">
        point size ∝ MAPE | larger = worse | max {maxMape.toFixed(1)}%
      </p>
    </div>
  );
}
