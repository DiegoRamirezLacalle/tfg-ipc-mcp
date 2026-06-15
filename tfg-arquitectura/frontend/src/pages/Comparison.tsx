import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { Activity, ChevronDown } from "lucide-react";

import { useExperiments, useAllSeries } from "@/lib/queries";
import { useComparisonData } from "@/hooks/useComparisonData";
import { SectionCard } from "@/components/comparison/SectionCard";
import { ForecastOverlay } from "@/components/comparison/ForecastOverlay";
import { MetricBars } from "@/components/comparison/MetricBars";
import { SkillScoreChart } from "@/components/comparison/SkillScoreChart";
import { MetricRadar } from "@/components/comparison/MetricRadar";
import { HorizonDecay } from "@/components/comparison/HorizonDecay";
import { ParetoScatter } from "@/components/comparison/ParetoScatter";
import { Leaderboard } from "@/components/comparison/Leaderboard";
import { DMMatrix } from "@/components/comparison/DMMatrix";

export default function Comparison() {
  const experiments = useExperiments();
  const seriesInfo = useAllSeries();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [hoveredModel, setHoveredModel] = useState<string | null>(null);
  const [seriesId, setSeriesId] = useState<number | null>(null);
  const [showDeepDive, setShowDeepDive] = useState(false);

  // Series that actually have experiments, with their experiment counts.
  const seriesOptions = useMemo(() => {
    const counts = new Map<number, number>();
    (experiments.data ?? []).forEach((e) =>
      counts.set(e.series_id, (counts.get(e.series_id) ?? 0) + 1)
    );
    return Array.from(counts.entries())
      .map(([sid, count]) => {
        const info = seriesInfo.map.get(sid);
        return {
          id: sid,
          count,
          label: info ? `${info.datasetName} | ${info.name}` : `series #${sid}`,
        };
      })
      .sort((a, b) => b.count - a.count);
  }, [experiments.data, seriesInfo.map]);

  // Default to the series with the most experiments.
  useEffect(() => {
    if (seriesId === null && seriesOptions.length > 0) {
      setSeriesId(seriesOptions[0].id);
    }
  }, [seriesOptions, seriesId]);

  const visibleExperiments = useMemo(
    () => (experiments.data ?? []).filter((e) => e.series_id === seriesId),
    [experiments.data, seriesId]
  );

  const selectedIds = useMemo(() => Array.from(selected), [selected]);
  const { data, isLoading } = useComparisonData(selectedIds);

  const toggle = (id: number) => {
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else if (next.size < 10) next.add(id);
      return next;
    });
  };

  // Switching the target series clears the selection (no cross-series mixing).
  const changeSeries = (sid: number) => {
    setSeriesId(sid);
    setSelected(new Set());
  };

  const runs = data?.runs ?? [];
  const hasEnough = runs.length >= 2;
  const activeSeriesLabel =
    seriesOptions.find((o) => o.id === seriesId)?.label ?? "-";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr] min-h-[calc(100vh-56px-64px)]">
      {/* Sidebar */}
      <aside className="border-r border-border bg-card flex flex-col shrink-0">
        <div className="p-5 border-b border-border flex flex-col gap-3">
          <div>
            <h2 className="font-sans text-headline-md text-foreground">Experiments</h2>
            <p className="font-mono text-data-sm text-foreground-muted mt-1">
              Same-series only - overlays are comparable
            </p>
          </div>
          <label className="flex flex-col gap-1">
            <span className="micro uppercase">Target series</span>
            <select
              value={seriesId ?? ""}
              onChange={(e) => changeSeries(Number(e.target.value))}
              className="bg-muted border border-border rounded px-2.5 py-1.5 font-mono text-data-sm text-foreground focus:border-mcp outline-none"
            >
              {seriesOptions.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label} ({o.count})
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="p-4 flex flex-col gap-1 overflow-y-auto flex-1">
          {experiments.data?.length === 0 && (
            <p className="font-mono text-data-sm text-foreground-subtle p-2">No experiments yet.</p>
          )}
          {experiments.data && experiments.data.length > 0 && visibleExperiments.length === 0 && (
            <p className="font-mono text-data-sm text-foreground-subtle p-2">
              No experiments on this series.
            </p>
          )}
          {visibleExperiments.map((e) => (
            <label
              key={e.id}
              className="flex items-center gap-3 p-2 rounded cursor-pointer hover:bg-muted transition-colors border border-transparent hover:border-border"
            >
              <input
                type="checkbox"
                checked={selected.has(e.id)}
                onChange={() => toggle(e.id)}
                className="w-3.5 h-3.5 accent-mcp"
              />
              <div className="flex flex-col min-w-0">
                <span className="font-mono text-data-sm text-foreground truncate">{e.name}</span>
                <span className="font-mono text-[10px] text-foreground-subtle uppercase tracking-wider">
                  #{e.id} | h={e.horizon}
                </span>
              </div>
            </label>
          ))}
        </div>
        <div className="p-4 border-t border-border">
          <p className="font-mono text-[10px] text-foreground-subtle uppercase tracking-widest">
            {selected.size} / 10 selected
          </p>
        </div>
      </aside>

      {/* Main */}
      <section className="flex flex-col">
        {/* Page header */}
        <div className="px-6 py-6 border-b border-border">
          <div className="flex items-baseline justify-between">
            <div>
              <h1 className="font-sans text-display-lg tracking-tight text-foreground">Model Comparison</h1>
              <p className="font-mono text-data-sm text-foreground-muted mt-1">
                Same-series models on <span className="text-gold">{activeSeriesLabel}</span> | overlay,
                horizon decay, skill score, and Diebold-Mariano significance.
              </p>
            </div>
            {isLoading && selected.size > 0 && (
              <span className="flex items-center gap-2 font-mono text-data-sm text-foreground-subtle">
                <Activity size={13} className="animate-pulse-soft" />
                Loading data...
              </span>
            )}
          </div>
        </div>

        {selected.size === 0 && <EmptyState reason="select" />}
        {selected.size === 1 && data && <EmptyState reason="needMore" />}
        {selected.size > 0 && isLoading && !data && <LoadingSkeleton />}

        {hasEnough && data && (
          <div className="flex flex-col">
            {/* -- Core thesis evidence ------------------------------- */}
            <SectionCard number="01" title="Forecast Overlay" subtitle="Actuals vs every selected forecast - the visual story">
              <ForecastOverlay
                data={data}
                hoveredModel={hoveredModel}
                onHoverModel={setHoveredModel}
              />
            </SectionCard>

            <SectionCard number="02" title="Horizon-wise Error Decay" subtitle="Where foundation models win (long h) and lose (short h)" delay={0.05}>
              <HorizonDecay data={data} hoveredModel={hoveredModel} onHoverModel={setHoveredModel} />
            </SectionCard>

            <SectionCard number="03" title="Skill Score vs Seasonal Naïve" subtitle="Beats the lag-12 benchmark when skill > 0" delay={0.1}>
              <SkillScoreChart runs={runs} hoveredModel={hoveredModel} onHoverModel={setHoveredModel} />
            </SectionCard>

            <SectionCard number="04" title="Leaderboard" subtitle="Ranked by MAE | marks best | click headers to sort" delay={0.15}>
              <Leaderboard runs={runs} hoveredModel={hoveredModel} onHoverModel={setHoveredModel} />
            </SectionCard>

            <SectionCard number="05" title="Statistical Significance" subtitle="Is the MAE gap real? Diebold-Mariano test, HLN-corrected" delay={0.2}>
              <DMMatrix runs={runs} hoveredModel={hoveredModel} onHoverModel={setHoveredModel} />
            </SectionCard>

            {/* -- Deep dive (secondary views) ------------------------ */}
            <button
              onClick={() => setShowDeepDive((v) => !v)}
              className="flex items-center gap-2 px-4 py-3 border-t border-border bg-card/40 hover:bg-muted/50 transition-colors font-mono text-data-sm text-foreground-muted"
            >
              <ChevronDown
                size={14}
                className={"transition-transform " + (showDeepDive ? "rotate-180" : "")}
              />
              {showDeepDive ? "Hide" : "Show"} deep-dive views | metric bars, radar, accuracy×runtime
            </button>

            {showDeepDive && (
              <>
                <SectionCard number="06" title="Metric Comparison" subtitle="MAE / RMSE on left axis | MAPE % on right" delay={0.02}>
                  <MetricBars runs={runs} hoveredModel={hoveredModel} onHoverModel={setHoveredModel} />
                </SectionCard>

                <SectionCard number="07" title="Multi-Metric Radar" subtitle="normalized - bigger is better" delay={0.04}>
                  <MetricRadar runs={runs} hoveredModel={hoveredModel} onHoverModel={setHoveredModel} />
                </SectionCard>

                <SectionCard number="08" title="Accuracy × Runtime" subtitle="Pareto frontier | point size ∝ MAPE" delay={0.06}>
                  <ParetoScatter runs={runs} hoveredModel={hoveredModel} onHoverModel={setHoveredModel} />
                </SectionCard>
              </>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function EmptyState({ reason }: { reason: "select" | "needMore" }) {
  const message =
    reason === "select"
      ? "Select at least two completed runs from the sidebar to begin comparison."
      : "Select at least one more experiment to enable the comparison dashboard.";
  return (
    <motion.div
      className="p-12 m-6 border border-dashed border-border rounded flex flex-col items-center text-center gap-3"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <p className="font-display italic text-2xl text-foreground-subtle leading-snug max-w-md">
        {message}
      </p>
      <p className="font-mono text-data-sm text-foreground-subtle/70 mt-2">
        the dashboard reveals forecast overlay | metric bars | radar | skill score | horizon decay | pareto frontier | leaderboard
      </p>
    </motion.div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-px">
      {[1, 2, 3].map((i) => (
        <div key={i} className="bg-card/40 border-t border-border">
          <div className="px-4 py-3 border-b border-border">
            <div className="h-3 w-32 bg-muted rounded animate-pulse" />
          </div>
          <div className="p-4">
            <div className="h-40 bg-muted/40 rounded animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}
