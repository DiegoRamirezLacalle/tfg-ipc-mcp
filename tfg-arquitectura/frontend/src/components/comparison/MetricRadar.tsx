import { useMemo } from "react";
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import { modelColor } from "@/lib/modelColors";
import type { ComparisonRunData } from "@/hooks/useComparisonData";

interface MetricRadarProps {
  runs: ComparisonRunData[];
  hoveredModel: string | null;
  onHoverModel: (slug: string | null) => void;
}

const AXES = ["MAE", "RMSE", "MAPE", "Skill"] as const;

export function MetricRadar({ runs, hoveredModel, onHoverModel }: MetricRadarProps) {
  const { rows, slugs } = useMemo(() => {
    const usable = runs.filter((r) => r.mae !== null && r.rmse !== null && r.mape !== null);
    if (usable.length < 2) return { rows: [], slugs: [] };

    const maes = usable.map((r) => r.mae as number);
    const rmses = usable.map((r) => r.rmse as number);
    const mapes = usable.map((r) => r.mape as number);
    const skills = usable.filter((r) => r.skill !== null).map((r) => r.skill as number);

    const minMax = (xs: number[]) => [Math.min(...xs), Math.max(...xs)] as [number, number];
    const norm = (v: number, mn: number, mx: number, invert = false) => {
      if (mx === mn) return 0.5;
      const t = (v - mn) / (mx - mn);
      return invert ? 1 - t : t;
    };

    const [maeMin, maeMax] = minMax(maes);
    const [rmseMin, rmseMax] = minMax(rmses);
    const [mapeMin, mapeMax] = minMax(mapes);
    const [skillMin, skillMax] = skills.length > 0 ? minMax(skills) : [0, 0];

    // Recharts radar wants rows: one per axis, with one key per series
    const rows = AXES.map((axis) => {
      const row: Record<string, number | string> = { axis };
      usable.forEach((r) => {
        let v: number;
        if (axis === "MAE")        v = 1 - norm(r.mae as number, maeMin, maeMax);
        else if (axis === "RMSE")  v = 1 - norm(r.rmse as number, rmseMin, rmseMax);
        else if (axis === "MAPE")  v = 1 - norm(r.mape as number, mapeMin, mapeMax);
        else                       v = r.skill === null ? 0 : norm(r.skill, skillMin, skillMax);
        // Render as % so axis labels read 0..100
        row[r.modelSlug] = Math.max(0, Math.min(1, v)) * 100;
      });
      return row;
    });

    return { rows, slugs: usable.map((r) => r.modelSlug) };
  }, [runs]);

  if (rows.length === 0) {
    return (
      <div className="font-mono text-data-sm text-foreground-subtle py-10 text-center">
        Need at least two completed runs with all metrics to draw the radar.
      </div>
    );
  }

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={340}>
        <RadarChart data={rows} outerRadius="75%">
          <PolarGrid stroke="hsl(var(--border))" />
          <PolarAngleAxis
            dataKey="axis"
            tick={{ fontFamily: "Geist Mono", fontSize: 11, fill: "hsl(var(--foreground-muted))" }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fontFamily: "Geist Mono", fontSize: 9, fill: "hsl(var(--foreground-subtle))" }}
            tickFormatter={(v) => `${v}`}
            stroke="hsl(var(--border))"
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
            formatter={(v: number) => [`${v.toFixed(0)} / 100`, ""]}
          />
          {slugs.map((slug, i) => {
            const c = modelColor(slug, i);
            const isHovered = hoveredModel === slug;
            const isDimmed = hoveredModel !== null && !isHovered;
            return (
              <Radar
                key={slug}
                name={slug}
                dataKey={slug}
                stroke={c}
                strokeWidth={isHovered ? 2 : 1.5}
                fill={c}
                fillOpacity={isDimmed ? 0.05 : isHovered ? 0.4 : 0.25}
                isAnimationActive
                animationDuration={700}
              />
            );
          })}
        </RadarChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 pt-2">
        {slugs.map((slug, i) => {
          const c = modelColor(slug, i);
          const isHovered = hoveredModel === slug;
          const isDimmed = hoveredModel !== null && !isHovered;
          return (
            <button
              key={slug}
              onMouseEnter={() => onHoverModel(slug)}
              onMouseLeave={() => onHoverModel(null)}
              className={`flex items-center gap-1.5 font-mono text-data-sm transition-opacity ${isDimmed ? "opacity-30" : ""}`}
            >
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: c }} />
              <span style={{ color: isHovered ? c : "hsl(var(--foreground-muted))" }}>{slug}</span>
            </button>
          );
        })}
      </div>
      <p className="font-mono text-[10px] text-foreground-subtle uppercase tracking-widest text-center pt-2">
        normalized 0–100 · larger polygon = better overall
      </p>
    </div>
  );
}
