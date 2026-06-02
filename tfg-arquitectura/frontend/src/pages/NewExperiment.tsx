import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronDown } from "lucide-react";
import { motion } from "motion/react";
import { useCreateExperiment, useDatasets, useModels, useSeries } from "@/lib/queries";
import type { ExperimentCreate } from "@/lib/types";

const INPUT_CLS = "w-full h-10 px-3 bg-background border border-border rounded font-mono text-data-base text-foreground focus:border-mcp focus:ring-1 focus:ring-mcp/20 outline-none transition-colors placeholder:text-foreground-subtle disabled:opacity-40";
const SELECT_CLS = "w-full h-10 px-3 pr-9 bg-background border border-border rounded font-mono text-data-base text-foreground focus:border-mcp focus:ring-1 focus:ring-mcp/20 outline-none transition-colors appearance-none disabled:opacity-40 cursor-pointer";
const LABEL_CLS = "flex flex-col gap-1.5";

const MODEL_GROUPS: { label: string; slugs: string[] }[] = [
  {
    label: "Statistical baselines",
    slugs: ["naive-seasonal", "arima", "auto-arima", "sarima", "sarimax", "ridge-exog"],
  },
  {
    label: "Foundation models",
    slugs: ["timesfm", "chronos-2", "timegpt"],
  },
];

function SelectWrap({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative">
      {children}
      <ChevronDown size={13} className="absolute right-3 top-1/2 -translate-y-1/2 text-foreground-subtle pointer-events-none" />
    </div>
  );
}

export default function NewExperiment() {
  const navigate = useNavigate();
  const datasets = useDatasets();
  const models = useModels();

  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [seriesId, setSeriesId] = useState<number | null>(null);
  const [modelId, setModelId] = useState<number | null>(null);
  const [horizon, setHorizon] = useState(12);
  const [useMcp, setUseMcp] = useState(false);

  const series = useSeries(datasetId);
  const create = useCreateExperiment();

  const payload: Partial<ExperimentCreate> = {
    name: name || undefined,
    series_id: seriesId ?? undefined,
    model_id: modelId ?? undefined,
    horizon,
    use_mcp: useMcp,
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!seriesId || !modelId || !name) return;
    create.mutate(
      { name, series_id: seriesId, model_id: modelId, horizon, use_mcp: useMcp },
      { onSuccess: (exp) => navigate(`/experiments/${exp.id}`) }
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-8 max-w-4xl">
      {/* Left — form */}
      <motion.form
        onSubmit={onSubmit}
        className="flex flex-col gap-6"
        initial={{ opacity: 0, x: -8 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="border-b border-border pb-4">
          <h1 className="font-sans text-display-lg tracking-tight text-foreground">New Experiment</h1>
          <p className="font-mono text-data-sm text-foreground-muted mt-1">
            Configure a forecasting experiment. Trigger runs from the detail page.
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <label className={LABEL_CLS}>
            <span className="micro">Experiment Name</span>
            <input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className={INPUT_CLS}
              placeholder="e.g. IPC_Baseline_Q1_2025"
            />
          </label>

          <label className={LABEL_CLS}>
            <span className="micro">Dataset</span>
            <SelectWrap>
              <select
                value={datasetId ?? ""}
                onChange={(e) => { setDatasetId(Number(e.target.value) || null); setSeriesId(null); }}
                required
                className={SELECT_CLS}
              >
                <option value="" className="bg-card text-foreground-subtle">— select dataset —</option>
                {datasets.data?.map((d) => (
                  <option key={d.id} value={d.id} className="bg-card text-foreground">{d.name} ({d.frequency})</option>
                ))}
              </select>
            </SelectWrap>
          </label>

          <label className={LABEL_CLS}>
            <span className="micro">Series</span>
            <SelectWrap>
              <select
                value={seriesId ?? ""}
                onChange={(e) => setSeriesId(Number(e.target.value) || null)}
                required
                disabled={!datasetId}
                className={SELECT_CLS}
              >
                <option value="" className="bg-card text-foreground-subtle">— select series —</option>
                {series.data?.map((s) => (
                  <option key={s.id} value={s.id} className="bg-card text-foreground">{s.name}{s.unit ? ` (${s.unit})` : ""}</option>
                ))}
              </select>
            </SelectWrap>
          </label>

          <label className={LABEL_CLS}>
            <span className="micro">Model</span>
            <SelectWrap>
              <select
                value={modelId ?? ""}
                onChange={(e) => setModelId(Number(e.target.value) || null)}
                required
                className={SELECT_CLS}
              >
                <option value="" className="bg-card text-foreground-subtle">— select model —</option>
                {MODEL_GROUPS.map((group) => {
                  const groupModels = models.data?.filter(
                    (m) => m.is_active && group.slugs.includes(m.slug)
                  ).sort((a, b) => group.slugs.indexOf(a.slug) - group.slugs.indexOf(b.slug));
                  if (!groupModels?.length) return null;
                  return (
                    <optgroup key={group.label} label={group.label} className="bg-card text-foreground-subtle font-mono">
                      {groupModels.map((m) => (
                        <option key={m.id} value={m.id} className="bg-card text-foreground">
                          {m.name} · {m.slug}{m.supports_mcp ? " ◈" : ""}
                        </option>
                      ))}
                    </optgroup>
                  );
                })}
              </select>
            </SelectWrap>
          </label>

          <label className={LABEL_CLS}>
            <span className="micro">Forecast Horizon (months)</span>
            <div className="flex items-center gap-3">
              <input
                type="number"
                min={1}
                max={60}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-28 h-10 px-3 bg-background border border-border rounded font-mono text-data-base text-foreground focus:border-mcp focus:ring-1 focus:ring-mcp/20 outline-none transition-colors"
              />
              <span className="font-mono text-data-sm text-foreground-muted">months ahead</span>
            </div>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <div className="relative">
              <input
                type="checkbox"
                checked={useMcp}
                onChange={(e) => setUseMcp(e.target.checked)}
                className="sr-only"
              />
              <div className={`w-10 h-5 rounded-full transition-colors ${useMcp ? "bg-mcp" : "bg-border"}`}>
                <div className={`w-4 h-4 rounded-full bg-white shadow-sm absolute top-0.5 transition-transform ${useMcp ? "translate-x-5" : "translate-x-0.5"}`} />
              </div>
            </div>
            <div>
              <span className="font-mono text-data-base text-foreground">Use MCP semantic context</span>
              <p className="font-mono text-data-sm text-foreground-muted">
                Enriches forecast with macro signals via MCP server ◈
              </p>
            </div>
          </label>
        </div>

        {create.error && (
          <p className="font-mono text-data-sm text-destructive">{(create.error as Error).message}</p>
        )}

        <div className="flex gap-3 pt-2">
          <button
            type="submit"
            disabled={create.isPending || !name || !seriesId || !modelId}
            className="h-10 px-6 bg-mcp hover:bg-mcp/90 text-white font-mono text-label-caps uppercase tracking-widest rounded transition-colors disabled:opacity-40"
          >
            {create.isPending ? "Creating…" : "Create →"}
          </button>
          <button
            type="button"
            onClick={() => navigate("/experiments")}
            className="h-10 px-4 border border-border text-foreground-muted hover:text-foreground font-mono text-data-sm rounded transition-colors"
          >
            Cancel
          </button>
        </div>
      </motion.form>

      {/* Right — JSON preview */}
      <motion.aside
        className="card-tech p-5 flex flex-col gap-3"
        initial={{ opacity: 0, x: 8 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.1, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="flex items-center justify-between">
          <span className="micro">Payload Preview</span>
          <span className="font-mono text-[10px] text-foreground-subtle uppercase tracking-wider">application/json</span>
        </div>
        <div className="gold-rule" />
        <pre className="font-mono text-data-sm text-foreground-muted leading-relaxed overflow-auto whitespace-pre-wrap break-all">
          {JSON.stringify(payload, null, 2)}
        </pre>
        <div className="mt-auto pt-4 border-t border-border">
          <p className="font-mono text-[10px] text-foreground-subtle/60 uppercase tracking-widest">
            POST /api/v1/experiments
          </p>
        </div>
      </motion.aside>
    </div>
  );
}
