import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { PlayCircle, ArrowLeft, Activity, AlertTriangle } from "lucide-react";
import { motion } from "motion/react";
import { useExperiment, useExperimentRuns, useModels, useTriggerRun, useDrift } from "@/lib/queries";
import { StatusPill } from "@/components/ui/StatusPill";

function fmt(iso: string | null) {
  if (!iso) return "-";
  return iso.replace("T", " ").slice(0, 19) + " UTC";
}

function elapsed(start: string | null, end: string | null) {
  if (!start || !end) return "-";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms / 60000)}m`;
}

export default function ExperimentDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const expId = id ? Number(id) : undefined;

  const exp = useExperiment(expId);
  const runs = useExperimentRuns(expId);
  const trigger = useTriggerRun();
  const drift = useDrift(expId);
  const { data: models = [] } = useModels();
  const modelMap = useMemo(
    () => Object.fromEntries(models.map((m) => [m.id, m.slug])),
    [models]
  );

  if (exp.isLoading) {
    return (
      <div className="flex items-center gap-2 font-mono text-data-sm text-foreground-subtle py-12">
        <Activity size={14} className="animate-pulse-soft" />
        Loading experiment...
      </div>
    );
  }
  if (exp.error) return <p className="font-mono text-data-sm text-destructive">{(exp.error as Error).message}</p>;
  if (!exp.data) return null;

  const e = exp.data;
  const hasRunning = runs.data?.some((r) => r.status === "running" || r.status === "pending");

  return (
    <div className="flex flex-col gap-6">
      {/* Breadcrumb */}
      <button
        onClick={() => navigate("/experiments")}
        className="flex items-center gap-1.5 font-mono text-data-sm text-foreground-muted hover:text-foreground transition-colors w-fit"
      >
        <ArrowLeft size={13} />
        Experiments
      </button>

      {/* Header */}
      <motion.div
        className="flex items-start justify-between border-b border-border pb-5"
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex flex-col gap-2">
          <h1 className="font-sans text-display-lg tracking-tight text-foreground">{e.name}</h1>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-1.5 font-mono text-data-sm text-foreground-muted">
            <span><span className="text-foreground-subtle">ID:</span> {e.id}</span>
            <span><span className="text-foreground-subtle">Series:</span> {e.series_id}</span>
            <span><span className="text-foreground-subtle">Model:</span> {modelMap[e.model_id] ?? `#${e.model_id}`}</span>
            <span><span className="text-foreground-subtle">Horizon:</span> {e.horizon}m</span>
            <span><span className="text-foreground-subtle">Created:</span> {fmt(e.created_at)}</span>
            {e.use_mcp && <span className="pill pill-mcp text-[10px]">MCP </span>}
          </div>
        </div>
        <button
          onClick={() => expId && trigger.mutate(expId)}
          disabled={trigger.isPending || !!hasRunning}
          className="flex items-center gap-2 bg-mcp hover:bg-mcp/90 text-white font-mono text-data-sm px-4 py-2 rounded transition-colors disabled:opacity-40 shrink-0"
        >
          <PlayCircle size={14} />
          {trigger.isPending ? "Triggering..." : "New Run"}
        </button>
      </motion.div>

      {/* Status meta-row */}
      <div className="card-tech p-4 flex flex-wrap gap-x-8 gap-y-2 font-mono text-data-sm">
        <div className="flex items-center gap-2 text-foreground-muted">
          Status: <StatusPill status={e.status} />
        </div>
        <div className="text-foreground-muted">
          Runs: <span className="text-foreground">{runs.data?.length ?? 0}</span>
        </div>
        {hasRunning && (
          <div className="flex items-center gap-1.5 text-info">
            <span className="w-1.5 h-1.5 rounded-full bg-info animate-pulse" />
            Live - auto-refreshing
          </div>
        )}
      </div>

      {/* Drift alert banner */}
      {drift.data?.drifted && (
        <motion.div
          className="flex items-start gap-3 border border-warning/40 bg-warning/5 rounded-lg px-4 py-3"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <AlertTriangle size={14} className="text-warning mt-0.5 shrink-0" />
          <div className="flex flex-col gap-0.5">
            <span className="micro uppercase text-warning">Residual Drift Detected</span>
            <span className="font-mono text-data-sm text-foreground-muted">
              {drift.data.message}
            </span>
            <span className="font-mono text-[11px] text-foreground-subtle">
              KS test on run #{drift.data.run_id} residuals - early window n={drift.data.n_early}, recent window n={drift.data.n_recent}
            </span>
          </div>
        </motion.div>
      )}

      {/* Runs table */}
      <section className="flex flex-col gap-3">
        <h2 className="micro uppercase">Runs</h2>

        {runs.data?.length === 0 && (
          <div className="card-tech px-4 py-8 text-center font-mono text-data-sm text-foreground-subtle">
            No runs yet - trigger one above.
          </div>
        )}

        {(runs.data?.length ?? 0) > 0 && (
          <div className="card-tech overflow-hidden">
            <div className="grid grid-cols-12 gap-3 px-4 py-2.5 bg-muted border-b border-border">
              <div className="col-span-2 micro">Run</div>
              <div className="col-span-3 micro">Status</div>
              <div className="col-span-3 micro">Started</div>
              <div className="col-span-3 micro">Finished</div>
              <div className="col-span-1 micro text-right">Duration</div>
            </div>
            <div className="divide-y divide-border">
              {runs.data?.map((r, i) => (
                <motion.div
                  key={r.id}
                  className="grid grid-cols-12 gap-3 px-4 py-3 items-center hover:bg-muted/60 transition-colors"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.04 }}
                >
                  <div className="col-span-2">
                    <Link
                      to={`/runs/${r.id}`}
                      className="font-mono text-data-sm text-mcp hover:underline"
                    >
                      RUN-{r.id}
                    </Link>
                  </div>
                  <div className="col-span-3">
                    <StatusPill status={r.status} />
                  </div>
                  <div className="col-span-3 font-mono text-data-sm text-foreground-muted">{fmt(r.started_at)}</div>
                  <div className="col-span-3 font-mono text-data-sm text-foreground-muted">{fmt(r.finished_at)}</div>
                  <div className="col-span-1 font-mono text-data-sm text-foreground-muted text-right">
                    {elapsed(r.started_at, r.finished_at)}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Error if last run failed */}
      {runs.data?.some((r) => r.status === "failed" && r.error_message) && (
        <div className="card-tech border-destructive/40 p-4">
          <p className="micro text-destructive mb-1">Last Run Error</p>
          <pre className="font-mono text-data-sm text-foreground-muted whitespace-pre-wrap">
            {runs.data?.find((r) => r.status === "failed")?.error_message}
          </pre>
        </div>
      )}
    </div>
  );
}
