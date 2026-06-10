import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ComparisonRunData } from "@/hooks/useComparisonData";

interface SkillScoreChartProps {
  runs: ComparisonRunData[];
  hoveredModel: string | null;
  onHoverModel: (slug: string | null) => void;
}

const TICK_STYLE = { fontFamily: "'Geist Mono', monospace", fontSize: 11, fill: "hsl(var(--foreground-subtle))" };

export function SkillScoreChart({ runs, hoveredModel, onHoverModel }: SkillScoreChartProps) {
  const rows = useMemo(() => {
    return runs
      .filter((r) => r.skill !== null)
      .map((r) => ({ slug: r.modelSlug, skill: (r.skill as number) * 100 }))
      .sort((a, b) => b.skill - a.skill);
  }, [runs]);

  if (rows.length === 0) {
    return (
      <div className="font-mono text-data-sm text-foreground-subtle py-8 text-center">
        Skill score unavailable — need ≥12 months of history to compute a seasonal-naive baseline.
      </div>
    );
  }

  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.skill)), 10);
  const domainEdge = Math.ceil(maxAbs / 10) * 10;

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={Math.max(180, rows.length * 44)}>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 8, right: 32, bottom: 8, left: 8 }}
          onMouseLeave={() => onHoverModel(null)}
        >
          <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={[-domainEdge, domainEdge]}
            tick={TICK_STYLE}
            axisLine={{ stroke: "#3F3F46" }}
            tickLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <YAxis
            type="category"
            dataKey="slug"
            tick={TICK_STYLE}
            axisLine={false}
            tickLine={false}
            width={100}
          />
          <ReferenceLine x={0} stroke="#3F3F46" strokeWidth={1} />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 4,
              fontFamily: "Geist Mono",
              fontSize: 12,
            }}
            labelStyle={{ color: "hsl(var(--foreground-muted))" }}
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            formatter={(value: number) => [`${value.toFixed(1)}% vs seasonal naive`, "Skill"]}
          />
          <Bar
            dataKey="skill"
            radius={[2, 2, 2, 2]}
            isAnimationActive
            animationDuration={800}
            barSize={18}
          >
            {rows.map((row, i) => {
              const dim = hoveredModel !== null && hoveredModel !== row.slug;
              return (
                <Cell
                  key={i}
                  fill={row.skill >= 0 ? "#10B981" : "#EF4444"}
                  fillOpacity={dim ? 0.25 : 0.7}
                  onMouseEnter={() => onHoverModel(row.slug)}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="font-mono text-[10px] text-foreground-subtle uppercase tracking-widest text-center pt-2">
        positive = beats seasonal naïve baseline · negative = worse than baseline
      </p>
    </div>
  );
}
