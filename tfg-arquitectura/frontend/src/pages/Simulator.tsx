import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { SlidersHorizontal, RotateCcw, Cpu, TrendingUp, TrendingDown, Minus } from "lucide-react";

import { useDatasets, useSeries, useWhatifSetup } from "@/lib/queries";
import type { WhatifSignal } from "@/lib/types";
import { WhatifChart } from "@/components/charts/WhatifChart";
import { SimulatorChat, type SimulatorChatContext } from "@/components/chat/SimulatorChat";

const HORIZONS = [3, 6, 12];

export default function Simulator() {
  const datasets = useDatasets();
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [seriesId, setSeriesId] = useState<number | undefined>(undefined);
  const [horizon, setHorizon] = useState(6);
  const [overrides, setOverrides] = useState<Record<string, number>>({});

  // Auto-select a CPI dataset once loaded (prefer the headline IPC index)
  useEffect(() => {
    if (datasetId === null && datasets.data && datasets.data.length > 0) {
      const preferred =
        datasets.data.find((d) => d.slug === "ipc-spain-ine") ??
        datasets.data.find((d) => /ipc|cpi|hicp/i.test(d.slug)) ??
        datasets.data[0];
      setDatasetId(preferred.id);
    }
  }, [datasets.data, datasetId]);

  const series = useSeries(datasetId);

  // Auto-select a sensible default series (prefer the headline index)
  useEffect(() => {
    if (series.data && series.data.length > 0) {
      const preferred =
        series.data.find((s) => s.slug === "indice_general") ?? series.data[0];
      if (!series.data.some((s) => s.id === seriesId)) {
        setSeriesId(preferred.id);
      }
    }
  }, [series.data, seriesId]);

  const setup = useWhatifSetup(seriesId, horizon);

  // Reset overrides to baseline whenever the setup changes
  useEffect(() => {
    if (setup.data) {
      const init: Record<string, number> = {};
      setup.data.signals.forEach((s) => (init[s.key] = s.baseline_value));
      setOverrides(init);
    }
  }, [setup.data]);

  const counterfactual = useMemo(() => {
    if (!setup.data?.baseline) return [];
    return setup.data.baseline.map((b, d) => {
      let v = b.value;
      for (const sig of setup.data!.signals) {
        const ov = overrides[sig.key] ?? sig.baseline_value;
        v += (ov - sig.baseline_value) * (sig.effect_per_step[d] ?? 0);
      }
      return v;
    });
  }, [setup.data, overrides]);

  const summary = useMemo(() => {
    if (!setup.data?.baseline.length || !counterfactual.length) return null;
    const base = setup.data.baseline;
    const lastIdx = base.length - 1;
    const dH1 = counterfactual[0] - base[0].value;
    const dHn = counterfactual[lastIdx] - base[lastIdx].value;

    // Most influential signal at the final horizon
    let topKey: string | null = null;
    let topContribAbs = 0;
    let topContrib = 0;
    for (const sig of setup.data.signals) {
      const ov = overrides[sig.key] ?? sig.baseline_value;
      const contrib = (ov - sig.baseline_value) * (sig.effect_per_step[lastIdx] ?? 0);
      if (Math.abs(contrib) > topContribAbs) {
        topContribAbs = Math.abs(contrib);
        topContrib = contrib;
        topKey = sig.label;
      }
    }
    return { dH1, dHn, topKey, topContrib, lastIdx };
  }, [setup.data, counterfactual, overrides]);

  const chatContext = useMemo<SimulatorChatContext | null>(() => {
    if (!setup.data) return null;
    const lastIdx = Math.max(0, setup.data.baseline.length - 1);
    return {
      series_name: setup.data.series_name,
      series_unit: setup.data.unit,
      horizon: setup.data.horizon,
      signals: setup.data.signals.map((s) => ({
        key: s.key,
        label: s.label,
        baseline_value: s.baseline_value,
        current_value: overrides[s.key] ?? s.baseline_value,
        final_effect: s.effect_per_step[lastIdx] ?? null,
      })),
      baseline: setup.data.baseline.map((p) => p.value),
      counterfactual,
      top_driver_label: summary?.topKey ?? null,
      top_driver_contribution: summary?.topContrib ?? null,
    };
  }, [setup.data, overrides, counterfactual, summary]);

  const anyChange = useMemo(() => {
    if (!setup.data) return false;
    return setup.data.signals.some(
      (s) => Math.abs((overrides[s.key] ?? s.baseline_value) - s.baseline_value) > 1e-9
    );
  }, [setup.data, overrides]);

  const resetAll = () => {
    if (!setup.data) return;
    const init: Record<string, number> = {};
    setup.data.signals.forEach((s) => (init[s.key] = s.baseline_value));
    setOverrides(init);
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <motion.div
        className="flex flex-col gap-2 border-b border-border pb-5"
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex items-center gap-3">
          <SlidersHorizontal size={20} className="text-gold" />
          <h1 className="font-sans text-display-lg tracking-tight text-foreground">What-if Simulator</h1>
          <span className="pill pill-mcp text-[10px] ml-1">◈ MCP signals</span>
        </div>
        <p className="font-mono text-data-sm text-foreground-muted max-w-2xl">
          Perturb the macro / monetary-policy signals and watch the forecast respond. The counterfactual
          isolates the <span className="text-gold">marginal effect</span> of each signal on future
          inflation <span className="text-foreground-muted">changes</span> — the same exogenous-signal logic
          behind the C1 conditions in the thesis.
        </p>
      </motion.div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1">
          <span className="micro uppercase">Dataset</span>
          <select
            value={datasetId ?? ""}
            onChange={(e) => {
              setDatasetId(Number(e.target.value));
              setSeriesId(undefined);
            }}
            className="bg-muted border border-border rounded px-3 py-1.5 font-mono text-data-sm text-foreground focus:border-mcp outline-none"
          >
            {datasets.data?.map((d) => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="micro uppercase">Series</span>
          <select
            value={seriesId ?? ""}
            onChange={(e) => setSeriesId(Number(e.target.value))}
            className="bg-muted border border-border rounded px-3 py-1.5 font-mono text-data-sm text-foreground focus:border-mcp outline-none max-w-[260px]"
          >
            {series.data?.map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="micro uppercase">Horizon</span>
          <div className="flex gap-1">
            {HORIZONS.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={
                  "px-3 py-1.5 rounded font-mono text-data-sm border transition-colors " +
                  (horizon === h
                    ? "bg-mcp/15 text-mcp border-mcp/40"
                    : "bg-muted text-foreground-muted border-border hover:text-foreground")
                }
              >
                h={h}
              </button>
            ))}
          </div>
        </label>
      </div>

      {setup.isLoading && (
        <p className="font-mono text-data-sm text-foreground-subtle py-12">Computing base forecast…</p>
      )}
      {setup.error && (
        <p className="font-mono text-data-sm text-destructive">{(setup.error as Error).message}</p>
      )}

      {setup.data && chatContext && (
        <div className="flex flex-col gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
          {/* Sliders */}
          <div className="card-tech flex flex-col">
            <div className="border-b border-border px-4 py-3 flex items-center gap-2 bg-muted/50">
              <Cpu size={13} className="text-mcp" />
              <span className="micro uppercase">Signal Levers</span>
              <button
                onClick={resetAll}
                disabled={!anyChange}
                className="ml-auto flex items-center gap-1.5 font-mono text-data-sm text-foreground-muted hover:text-foreground transition-colors disabled:opacity-40"
              >
                <RotateCcw size={11} /> Reset
              </button>
            </div>
            <div className="p-4 flex flex-col gap-5">
              {!setup.data.signals_available && (
                <p className="font-mono text-data-sm text-warning">
                  MCP signal parquet not mounted — only the baseline forecast is shown.
                </p>
              )}
              {setup.data.signals.map((sig) => (
                <SignalSlider
                  key={sig.key}
                  sig={sig}
                  value={overrides[sig.key] ?? sig.baseline_value}
                  finalEffect={sig.effect_per_step[summary?.lastIdx ?? 0] ?? 0}
                  onChange={(v) => setOverrides((o) => ({ ...o, [sig.key]: v }))}
                  onReset={() => setOverrides((o) => ({ ...o, [sig.key]: sig.baseline_value }))}
                />
              ))}
            </div>
          </div>

          {/* Chart + summary */}
          <div className="flex flex-col gap-4">
            {summary && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <DeltaCard label={`Δ at h=1`} value={summary.dH1} />
                <DeltaCard label={`Δ at h=${horizon}`} value={summary.dHn} />
                <div className="card-tech p-4 flex flex-col gap-1">
                  <span className="micro uppercase">Top Driver</span>
                  <span className="font-mono text-data-base text-foreground truncate">
                    {summary.topKey && Math.abs(summary.topContrib) > 1e-6 ? summary.topKey : "—"}
                  </span>
                  {summary.topKey && Math.abs(summary.topContrib) > 1e-6 && (
                    <span className={"font-mono text-data-sm " + (summary.topContrib > 0 ? "text-destructive" : "text-success")}>
                      {summary.topContrib > 0 ? "+" : ""}{summary.topContrib.toFixed(3)} {setup.data.unit ?? ""}
                    </span>
                  )}
                </div>
              </div>
            )}

            <div className="card-tech flex flex-col overflow-hidden">
              <div className="border-b border-border px-4 py-3 flex items-center justify-between bg-muted/50">
                <span className="micro uppercase">{setup.data.series_name} · forecast</span>
                <div className="flex items-center gap-4 font-mono text-data-sm">
                  <span className="flex items-center gap-1.5 text-foreground-muted">
                    <span className="w-4 border-t-2 border-dashed border-mcp" /> Baseline
                  </span>
                  <span className="flex items-center gap-1.5 text-foreground-muted">
                    <span className="w-4 h-0.5 bg-gold rounded" /> Counterfactual
                  </span>
                </div>
              </div>
              <div className="p-4">
                <WhatifChart
                  history={setup.data.history}
                  baseline={setup.data.baseline}
                  counterfactual={counterfactual}
                  unit={setup.data.unit}
                />
              </div>
            </div>

            <p className="font-mono text-[11px] text-foreground-subtle px-1">
              Counterfactual = baseline + Σ (slider − baseline) × per-step marginal effect. Effects come from a
              direct multi-step Ridge fit on the <span className="text-foreground-muted">h-step change</span> (not
              the level — which avoids the spurious level correlation), standardised and expressed in{" "}
              {setup.data.unit ?? "target"} units.
            </p>
          </div>
        </div>

        {/* Chat tutor — knows the live simulator state and the thesis concepts */}
        <SimulatorChat context={chatContext} />
        </div>
      )}
    </div>
  );
}

function DeltaCard({ label, value }: { label: string; value: number }) {
  const sign = value > 1e-6 ? "up" : value < -1e-6 ? "down" : "flat";
  const color = sign === "up" ? "text-destructive" : sign === "down" ? "text-success" : "text-foreground-subtle";
  const Icon = sign === "up" ? TrendingUp : sign === "down" ? TrendingDown : Minus;
  return (
    <div className="card-tech p-4 flex flex-col gap-1">
      <span className="micro uppercase">{label}</span>
      <div className="flex items-center gap-2">
        <Icon size={16} className={color} />
        <span className={"font-mono text-data-2xl tabular " + color}>
          {value > 0 ? "+" : ""}{value.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

function SignalSlider({
  sig,
  value,
  finalEffect,
  onChange,
  onReset,
}: {
  sig: WhatifSignal;
  value: number;
  finalEffect: number;
  onChange: (v: number) => void;
  onReset: () => void;
}) {
  const changed = Math.abs(value - sig.baseline_value) > 1e-9;
  const contribution = (value - sig.baseline_value) * finalEffect;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="font-sans text-sm text-foreground">{sig.label}</span>
        <div className="flex items-center gap-2">
          <span className="font-mono text-data-sm text-foreground tabular">{value.toFixed(2)}</span>
          {changed && (
            <button onClick={onReset} className="text-foreground-subtle hover:text-foreground transition-colors">
              <RotateCcw size={11} />
            </button>
          )}
        </div>
      </div>
      <input
        type="range"
        min={sig.min}
        max={sig.max}
        step={sig.step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-gold cursor-pointer"
      />
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] text-foreground-subtle">{sig.hint}</span>
        {changed && Math.abs(contribution) > 1e-6 && (
          <span className={"font-mono text-[10px] " + (contribution > 0 ? "text-destructive" : "text-success")}>
            {contribution > 0 ? "+" : ""}{contribution.toFixed(3)}
          </span>
        )}
      </div>
    </div>
  );
}
