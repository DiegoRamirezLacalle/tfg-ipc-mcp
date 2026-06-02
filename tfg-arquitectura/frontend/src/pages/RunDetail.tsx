import { useParams, Link } from "react-router-dom";
import { BarChart2, TrendingDown, Percent, ArrowLeft, Cpu, Sparkles, Loader2 } from "lucide-react";
import { motion } from "motion/react";
import { useMcpContext, useNarration, usePredictions, useRun, useRunMetrics } from "@/lib/queries";
import { NumberTicker } from "@/components/ui/NumberTicker";
import { StatusPill } from "@/components/ui/StatusPill";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { SentimentChart } from "@/components/charts/SentimentChart";

function fmt(iso: string | null) {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 19) + " UTC";
}

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: number | null;
  suffix?: string;
  decimals?: number;
  delay?: number;
}

function MetricCard({ icon, label, value, suffix = "", decimals = 4, delay = 0 }: MetricCardProps) {
  return (
    <motion.div
      className="card-tech p-5 flex flex-col gap-3"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="flex items-center justify-between">
        <span className="micro uppercase">{label}</span>
        <span className="text-foreground-subtle">{icon}</span>
      </div>
      <div className="flex items-baseline gap-2">
        {value !== null ? (
          <NumberTicker
            value={value}
            decimals={decimals}
            suffix={suffix}
            className="font-mono text-data-2xl text-foreground"
          />
        ) : (
          <span className="font-mono text-data-2xl text-foreground-subtle">—</span>
        )}
      </div>
    </motion.div>
  );
}

export default function RunDetail() {
  const { id } = useParams<{ id: string }>();
  const runId = id ? Number(id) : undefined;

  const run = useRun(runId);
  const preds = usePredictions(runId);
  const metrics = useRunMetrics(runId);
  const mcp = useMcpContext(runId);
  const narration = useNarration(runId);

  if (run.isLoading) {
    return (
      <div className="flex items-center gap-2 font-mono text-data-sm text-foreground-subtle py-12">
        Loading run…
      </div>
    );
  }
  if (run.error) return <p className="font-mono text-data-sm text-destructive">{(run.error as Error).message}</p>;
  if (!run.data) return null;

  const r = run.data;
  const mae  = metrics.data?.find((m) => m.name === "mae")?.value ?? null;
  const rmse = metrics.data?.find((m) => m.name === "rmse")?.value ?? null;
  const mape = metrics.data?.find((m) => m.name === "mape")?.value ?? null;

  return (
    <div className="flex flex-col gap-6">
      {/* Breadcrumb */}
      <Link
        to={`/experiments/${r.experiment_id}`}
        className="flex items-center gap-1.5 font-mono text-data-sm text-foreground-muted hover:text-foreground transition-colors w-fit"
      >
        <ArrowLeft size={13} />
        Experiment #{r.experiment_id}
      </Link>

      {/* Header */}
      <motion.div
        className="flex flex-col gap-2 border-b border-border pb-5"
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex items-center gap-3">
          <h1 className="font-sans text-display-lg tracking-tight text-foreground">RUN-{r.id}</h1>
          <StatusPill status={r.status} />
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-data-sm text-foreground-muted">
          <span><span className="text-foreground-subtle">Started:</span> {fmt(r.started_at)}</span>
          <span><span className="text-foreground-subtle">Finished:</span> {fmt(r.finished_at)}</span>
          {(r.status === "running" || r.status === "pending") && (
            <span className="text-info flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-info animate-pulse" />
              Live — auto-refreshing every 2s
            </span>
          )}
        </div>
        {r.error_message && (
          <p className="font-mono text-data-sm text-destructive mt-1">{r.error_message}</p>
        )}
      </motion.div>

      {/* Metric cards */}
      {metrics.data && metrics.data.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricCard icon={<BarChart2 size={14} />} label="Mean Absolute Error (MAE)" value={mae} delay={0} />
          <MetricCard icon={<TrendingDown size={14} />} label="Root Mean Sq Error (RMSE)" value={rmse} delay={0.12} />
          <MetricCard icon={<Percent size={14} />} label="Mean Abs % Error (MAPE)" value={mape} decimals={2} suffix="%" delay={0.24} />
        </div>
      )}

      {/* Forecast chart */}
      {preds.data && preds.data.length > 0 && (
        <motion.section
          className="card-tech flex flex-col overflow-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.35, duration: 0.5 }}
        >
          <div className="border-b border-border px-4 py-3 flex items-center justify-between bg-muted/50">
            <span className="micro uppercase">Forecast Trajectory (YoY %)</span>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 font-mono text-data-sm text-foreground-muted">
                <div className="w-4 h-0.5 bg-mcp rounded" />
                Forecast
              </div>
            </div>
          </div>
          <div className="p-4">
            <ForecastChart predictions={preds.data} />
          </div>
        </motion.section>
      )}

      {/* MCP context */}
      {mcp.data && (
        <motion.section
          className="card-tech flex flex-col"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.4 }}
        >
          <div className="border-b border-border px-4 py-3 flex items-center gap-2 bg-muted/50">
            <Cpu size={13} className="text-mcp" />
            <span className="micro uppercase">MCP Macro Context</span>
            <span className="pill pill-mcp text-[10px] ml-auto">◈ MCP</span>
          </div>
          <div className="p-4 flex flex-col gap-4">
            <p className="font-mono text-data-sm text-foreground-muted">
              Fetched: {fmt(mcp.data.fetched_at)} · {mcp.data.signals?.length ?? 0} signal rows
            </p>

            {/* Sentiment timeline — only rendered if at least one row has sentiment data */}
            {(mcp.data.signals ?? []).some((s) => s.sentiment_mean !== undefined && s.sentiment_mean !== null) && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="micro uppercase text-foreground-muted">FinBERT Sentiment Timeline</span>
                  <span className="pill pill-mcp text-[10px]">◈ FinBERT</span>
                </div>
                <SentimentChart signals={mcp.data.signals ?? []} />
                <p className="font-mono text-[11px] text-foreground-subtle">
                  Sentiment: FinBERT pos−neg score (violet) · Hawkish %: fraction of articles with tightening keywords (rose)
                </p>
              </div>
            )}

            <pre className="font-mono text-data-sm text-foreground-muted bg-background border border-border rounded p-4 overflow-auto max-h-48 text-xs">
              {JSON.stringify(mcp.data.signals?.slice(0, 3), null, 2)}
              {(mcp.data.signals?.length ?? 0) > 3 && "\n…"}
            </pre>
          </div>
        </motion.section>
      )}

      {/* LLM narration */}
      {r.status === "done" && (
        <motion.section
          className="card-tech flex flex-col"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.4 }}
        >
          <div className="border-b border-border px-4 py-3 flex items-center gap-2 bg-muted/50">
            <Sparkles size={13} className="text-accent" />
            <span className="micro uppercase">LLM Analysis</span>
            <span className="font-mono text-[10px] text-foreground-subtle ml-1">
              via Ollama
            </span>
            <button
              onClick={() => narration.mutate()}
              disabled={narration.isPending}
              className="ml-auto flex items-center gap-1.5 px-3 py-1 rounded border border-border font-mono text-data-sm text-foreground-muted hover:text-foreground hover:border-foreground-subtle transition-colors disabled:opacity-50"
            >
              {narration.isPending ? (
                <>
                  <Loader2 size={11} className="animate-spin" />
                  Generating…
                </>
              ) : (
                <>
                  <Sparkles size={11} />
                  {narration.data ? "Regenerate" : "Generate"}
                </>
              )}
            </button>
          </div>
          <div className="p-4">
            {narration.error && (
              <p className="font-mono text-data-sm text-destructive">
                {(narration.error as Error).message}
              </p>
            )}
            {narration.data ? (
              <p className="font-sans text-sm text-foreground leading-relaxed">
                {narration.data.narrative}
              </p>
            ) : (
              !narration.isPending && (
                <p className="font-mono text-data-sm text-foreground-subtle">
                  Click Generate to produce an LLM-written analysis of this run using your local Ollama instance.
                </p>
              )
            )}
          </div>
        </motion.section>
      )}
    </div>
  );
}
