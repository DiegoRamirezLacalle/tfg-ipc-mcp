import { useMemo } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { modelColor } from "@/lib/modelColors";
import type { ComparisonData } from "@/hooks/useComparisonData";

interface ForecastOverlayProps {
  data: ComparisonData;
  hoveredModel: string | null;
  onHoverModel: (slug: string | null) => void;
  historyMonths?: number;
}

const TICK_STYLE = { fontFamily: "'Geist Mono', monospace", fontSize: 11, fill: "#71717A" };

interface ChartRow {
  date: string;
  actual?: number | null;
  [modelSlug: string]: number | string | null | undefined;
}

function isoMonth(iso: string): string {
  return iso.slice(0, 7);
}

export function ForecastOverlay({ data, hoveredModel, onHoverModel, historyMonths = 24 }: ForecastOverlayProps) {
  const { rows, splitDate, modelSlugs, yDomain } = useMemo(() => {
    const validRuns = data.runs.filter((r) => r.runId > 0 && data.predictionsByRun[r.runId]?.length);
    if (validRuns.length === 0) {
      return { rows: [] as ChartRow[], splitDate: null as string | null, modelSlugs: [] as string[], yDomain: undefined as [number, number] | undefined };
    }

    // collect all prediction timestamps from all runs (union)
    const predDates = new Set<string>();
    validRuns.forEach((r) => {
      data.predictionsByRun[r.runId].forEach((p) => predDates.add(isoMonth(p.timestamp)));
    });
    const firstPredDate = [...predDates].sort()[0];

    // pick the first run's series for history window (all selected runs should share series for this view)
    const seriesId = validRuns[0].seriesId;
    const actuals = data.actualsBySeries[seriesId] ?? [];

    // build full date axis: historyMonths of actuals before firstPredDate + union of prediction months
    const actualByMonth = new Map<string, number>();
    actuals.forEach((a) => actualByMonth.set(isoMonth(a.timestamp), a.value));

    const sortedActualMonths = [...actualByMonth.keys()].sort();
    const cutoffIdx = sortedActualMonths.findIndex((d) => d >= firstPredDate);
    const startIdx = Math.max(0, cutoffIdx - historyMonths);
    const historyDates = sortedActualMonths.slice(startIdx, cutoffIdx === -1 ? sortedActualMonths.length : cutoffIdx);

    const allDates = [...new Set([...historyDates, ...predDates])].sort();

    // build prediction lookup per model
    const predByModel = new Map<string, Map<string, number>>();
    validRuns.forEach((r) => {
      const map = new Map<string, number>();
      data.predictionsByRun[r.runId].forEach((p) => map.set(isoMonth(p.timestamp), p.value));
      predByModel.set(r.modelSlug, map);
    });

    const rows: ChartRow[] = allDates.map((d) => {
      const row: ChartRow = { date: d };
      const a = actualByMonth.get(d);
      row.actual = a ?? null;
      predByModel.forEach((map, slug) => {
        row[slug] = map.get(d) ?? null;
      });
      return row;
    });

    // tight y-domain based on visible values
    const allValues: number[] = [];
    rows.forEach((r) => {
      if (typeof r.actual === "number") allValues.push(r.actual);
      Object.keys(r).forEach((k) => {
        if (k !== "date" && k !== "actual" && typeof r[k] === "number") allValues.push(r[k] as number);
      });
    });
    const minV = Math.min(...allValues);
    const maxV = Math.max(...allValues);
    const pad = (maxV - minV) * 0.1 || 1;

    return {
      rows,
      splitDate: firstPredDate,
      modelSlugs: validRuns.map((r) => r.modelSlug),
      yDomain: [minV - pad, maxV + pad] as [number, number],
    };
  }, [data, historyMonths]);

  if (rows.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center font-mono text-data-sm text-foreground-subtle">
        No predictions yet for these runs.
      </div>
    );
  }

  // sparse x ticks
  const tickCount = Math.min(rows.length, 10);
  const step = Math.max(1, Math.floor(rows.length / tickCount));
  const xTicks = rows.filter((_, i) => i % step === 0).map((r) => r.date);

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart
          data={rows}
          margin={{ top: 10, right: 28, bottom: 4, left: 0 }}
          onMouseLeave={() => onHoverModel(null)}
        >
          <CartesianGrid stroke="#27272A" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            ticks={xTicks}
            tick={TICK_STYLE}
            axisLine={{ stroke: "#3F3F46" }}
            tickLine={false}
          />
          <YAxis
            domain={yDomain}
            tick={TICK_STYLE}
            axisLine={false}
            tickLine={false}
            width={52}
            tickFormatter={(v) => Number(v).toFixed(2)}
          />
          <Tooltip
            content={<OverlayTooltip />}
            cursor={{ stroke: "#3F3F46", strokeDasharray: "3 3" }}
          />
          {splitDate && (
            <ReferenceLine
              x={splitDate}
              stroke="#8B5CF6"
              strokeOpacity={0.45}
              strokeDasharray="5 5"
              label={{
                value: "forecast →",
                position: "insideTopRight",
                fill: "#8B5CF6",
                fontSize: 10,
                fontFamily: "Geist Mono",
                offset: 8,
              }}
            />
          )}
          {/* actuals first so they sit behind */}
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#FAFAFA"
            strokeWidth={hoveredModel ? 1.5 : 2}
            strokeOpacity={hoveredModel ? 0.35 : 0.95}
            dot={false}
            isAnimationActive={false}
            name="Actual"
            connectNulls
          />
          {modelSlugs.map((slug, i) => {
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
                animationBegin={250 + i * 120}
                animationDuration={1400}
                animationEasing="ease-out"
                name={slug}
                connectNulls
              />
            );
          })}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend below */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 pt-4 px-1">
        <button
          onMouseEnter={() => onHoverModel(null)}
          className="flex items-center gap-1.5 font-mono text-data-sm text-foreground-muted"
        >
          <span className="inline-block w-3 h-px bg-foreground" />
          actual
        </button>
        {modelSlugs.map((slug, i) => {
          const isHovered = hoveredModel === slug;
          const isDimmed = hoveredModel !== null && !isHovered;
          return (
            <button
              key={slug}
              onMouseEnter={() => onHoverModel(slug)}
              onMouseLeave={() => onHoverModel(null)}
              className={`flex items-center gap-1.5 font-mono text-data-sm transition-opacity ${isDimmed ? "opacity-30" : "opacity-100"}`}
            >
              <span className="inline-block w-3 h-px" style={{ background: modelColor(slug, i) }} />
              <span style={{ color: isHovered ? modelColor(slug, i) : "#A1A1AA" }}>{slug}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function OverlayTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number | null; color: string; dataKey: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const rows = payload.filter((p) => typeof p.value === "number");
  if (rows.length === 0) return null;
  // sort: actual first, then by value
  rows.sort((a, b) => (a.dataKey === "actual" ? -1 : b.dataKey === "actual" ? 1 : (b.value ?? 0) - (a.value ?? 0)));
  return (
    <div className="bg-card border border-border rounded p-3 font-mono text-data-sm shadow-xl min-w-[180px]">
      <p className="text-foreground-muted mb-2 text-label-caps uppercase tracking-wider">{label}</p>
      <div className="flex flex-col gap-1">
        {rows.map((r) => (
          <div key={r.dataKey} className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full" style={{ background: r.color }} />
              <span className="text-foreground-muted">{r.name}</span>
            </div>
            <span className="text-foreground font-medium tabular-nums">{(r.value as number).toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
