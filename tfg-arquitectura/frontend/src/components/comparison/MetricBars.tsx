import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { modelColor } from "@/lib/modelColors";
import type { ComparisonRunData } from "@/hooks/useComparisonData";

interface MetricBarsProps {
  runs: ComparisonRunData[];
  hoveredModel: string | null;
  onHoverModel: (slug: string | null) => void;
}

const TICK_STYLE = { fontFamily: "'Geist Mono', monospace", fontSize: 11, fill: "hsl(var(--foreground-subtle))" };

export function MetricBars({ runs, hoveredModel, onHoverModel }: MetricBarsProps) {
  const rows = useMemo(
    () =>
      runs.map((r) => ({
        slug: r.modelSlug,
        mae: r.mae,
        rmse: r.rmse,
        mape: r.mape,
      })),
    [runs]
  );

  if (rows.length === 0) return null;

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={260}>
        <BarChart
          data={rows}
          margin={{ top: 8, right: 36, bottom: 0, left: 0 }}
          barCategoryGap="22%"
          barGap={4}
          onMouseLeave={() => onHoverModel(null)}
        >
          <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="slug" tick={TICK_STYLE} axisLine={{ stroke: "#3F3F46" }} tickLine={false} />
          <YAxis
            yAxisId="abs"
            tick={TICK_STYLE}
            axisLine={false}
            tickLine={false}
            width={52}
            tickFormatter={(v) => Number(v).toFixed(2)}
          />
          <YAxis
            yAxisId="pct"
            orientation="right"
            tick={TICK_STYLE}
            axisLine={false}
            tickLine={false}
            width={44}
            tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 4,
              fontFamily: "Geist Mono",
              fontSize: 12,
            }}
            labelStyle={{ color: "hsl(var(--foreground-muted))", fontFamily: "Geist Mono" }}
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            formatter={(value: number, name: string) =>
              name === "MAPE"
                ? [`${value.toFixed(2)}%`, name]
                : [value.toFixed(4), name]
            }
          />
          <Legend
            wrapperStyle={{ fontFamily: "Geist Mono", fontSize: 11, color: "hsl(var(--foreground-muted))" }}
            iconType="square"
          />
          <Bar
            yAxisId="abs"
            dataKey="mae"
            name="MAE"
            fill="#8B5CF6"
            radius={[3, 3, 0, 0]}
            isAnimationActive
            animationDuration={700}
          >
            {rows.map((row, i) => {
              const dim = hoveredModel !== null && hoveredModel !== row.slug;
              return <Cell key={i} fillOpacity={dim ? 0.25 : 0.95} onMouseEnter={() => onHoverModel(row.slug)} />;
            })}
          </Bar>
          <Bar
            yAxisId="abs"
            dataKey="rmse"
            name="RMSE"
            fill="#06B6D4"
            radius={[3, 3, 0, 0]}
            isAnimationActive
            animationDuration={700}
            animationBegin={120}
          >
            {rows.map((row, i) => {
              const dim = hoveredModel !== null && hoveredModel !== row.slug;
              return <Cell key={i} fillOpacity={dim ? 0.25 : 0.95} onMouseEnter={() => onHoverModel(row.slug)} />;
            })}
          </Bar>
          <Bar
            yAxisId="pct"
            dataKey="mape"
            name="MAPE"
            fill="#E0B96A"
            radius={[3, 3, 0, 0]}
            isAnimationActive
            animationDuration={700}
            animationBegin={240}
          >
            {rows.map((row, i) => {
              const dim = hoveredModel !== null && hoveredModel !== row.slug;
              return (
                <Cell
                  key={i}
                  fill={modelColor(row.slug, i) === "#E0B96A" ? "#F59E0B" : "#E0B96A"}
                  fillOpacity={dim ? 0.25 : 0.8}
                  onMouseEnter={() => onHoverModel(row.slug)}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-between mt-2 px-1">
        <span className="font-mono text-[10px] text-foreground-subtle uppercase tracking-widest">
          left axis · MAE / RMSE
        </span>
        <span className="font-mono text-[10px] text-foreground-subtle uppercase tracking-widest">
          right axis · MAPE %
        </span>
      </div>
    </div>
  );
}
