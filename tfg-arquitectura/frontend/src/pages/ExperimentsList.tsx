import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, Search } from "lucide-react";
import { motion } from "motion/react";
import { useExperiments, useModels, useAllSeries } from "@/lib/queries";
import { StatusPill } from "@/components/ui/StatusPill";
import type { Experiment, ExperimentStatus } from "@/lib/types";

const FILTERS: Array<{ key: ExperimentStatus | "all"; label: string }> = [
  { key: "all",     label: "All" },
  { key: "running", label: "Running" },
  { key: "done",    label: "Completed" },
  { key: "failed",  label: "Failed" },
  { key: "created", label: "Created" },
];

function fmt(iso: string) {
  return iso.replace("T", " ").slice(0, 19) + " UTC";
}

export default function ExperimentsList() {
  const navigate = useNavigate();
  const { data: experiments = [], isLoading } = useExperiments();
  const { data: models = [] } = useModels();
  const seriesInfo = useAllSeries();
  const [filter, setFilter] = useState<ExperimentStatus | "all">("all");
  const [search, setSearch] = useState("");

  const modelMap = useMemo(
    () => Object.fromEntries(models.map((m) => [m.id, m.slug])),
    [models]
  );

  const counts: Partial<Record<ExperimentStatus, number>> = {
    running: experiments.filter((e) => e.status === "running").length,
    done:    experiments.filter((e) => e.status === "done").length,
    failed:  experiments.filter((e) => e.status === "failed").length,
  };

  const visible = experiments.filter((e: Experiment) => {
    if (filter !== "all" && e.status !== filter) return false;
    if (search && !e.name.toLowerCase().includes(search.toLowerCase()) &&
        !String(e.id).includes(search)) return false;
    return true;
  });

  // Divide experiments by the series they forecast - only same-series experiments
  // are comparable (same target, same scale).
  const grouped = useMemo(() => {
    const groups = new Map<
      number,
      { seriesId: number; seriesName: string; datasetName: string; items: Experiment[] }
    >();
    visible.forEach((e) => {
      const info = seriesInfo.map.get(e.series_id);
      const sid = e.series_id;
      if (!groups.has(sid)) {
        groups.set(sid, {
          seriesId: sid,
          seriesName: info?.name ?? `series #${sid}`,
          datasetName: info?.datasetName ?? "Unassigned dataset",
          items: [],
        });
      }
      groups.get(sid)!.items.push(e);
    });
    return Array.from(groups.values()).sort(
      (a, b) =>
        a.datasetName.localeCompare(b.datasetName) ||
        a.seriesName.localeCompare(b.seriesName)
    );
  }, [visible, seriesInfo.map]);

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-end justify-between border-b border-border pb-4">
        <div className="flex items-baseline gap-3">
          <h1 className="font-sans text-display-lg tracking-tight text-foreground">Experiments</h1>
          <span className="font-mono text-data-sm text-foreground-muted">{experiments.length} total</span>
        </div>
        <button
          onClick={() => navigate("/experiments/new")}
          className="flex items-center gap-1.5 bg-mcp hover:bg-mcp/90 text-white font-mono text-data-sm px-4 py-2 rounded transition-colors"
        >
          <Plus size={14} />
          New experiment
        </button>
      </div>

      {/* Filters + search */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex gap-2">
          {FILTERS.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className={`px-3 py-1 rounded font-mono text-data-sm border transition-colors ${
                filter === key
                  ? "border-border-strong bg-muted text-foreground"
                  : "border-border bg-transparent text-foreground-muted hover:bg-muted hover:text-foreground"
              }`}
            >
              {label}
              {key !== "all" && counts[key] !== undefined && (
                <span className="ml-1.5 opacity-50">{counts[key]}</span>
              )}
            </button>
          ))}
        </div>
        <div className="relative">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-foreground-subtle" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search ID or name"
            className="bg-background border border-border rounded pl-8 pr-3 py-1.5 font-mono text-data-sm text-foreground focus:border-mcp focus:ring-1 focus:ring-mcp/20 outline-none w-52 placeholder:text-foreground-subtle transition-colors"
          />
        </div>
      </div>

      {/* Table */}
      <div className="card-tech overflow-hidden">
        <div className="grid grid-cols-12 gap-3 px-4 py-2.5 bg-muted border-b border-border">
          <div className="col-span-1 micro">ID</div>
          <div className="col-span-4 micro">Name</div>
          <div className="col-span-2 micro">Model</div>
          <div className="col-span-1 micro text-center">MCP</div>
          <div className="col-span-2 micro">Created</div>
          <div className="col-span-2 micro text-right">Status</div>
        </div>

        <div>
          {isLoading && (
            <div className="px-4 py-8 text-center font-mono text-data-sm text-foreground-subtle">
              Loading experiments...
            </div>
          )}
          {!isLoading && visible.length === 0 && (
            <div className="px-4 py-8 text-center font-mono text-data-sm text-foreground-subtle">
              No experiments found
            </div>
          )}
          {grouped.map((g) => (
            <section key={g.seriesId}>
              {/* Series divider */}
              <div className="px-4 py-2 bg-muted/40 border-b border-border flex items-center gap-2 sticky top-0 z-10">
                <span className="w-1.5 h-1.5 rounded-full bg-gold" />
                <span className="font-mono text-data-sm text-foreground">{g.seriesName}</span>
                <span className="font-mono text-[10px] text-foreground-subtle">- {g.datasetName}</span>
                <span className="font-mono text-[10px] text-foreground-subtle uppercase tracking-wider ml-auto">
                  {g.items.length} experiment{g.items.length === 1 ? "" : "s"}
                </span>
              </div>
              <div className="divide-y divide-border">
                {g.items.map((exp, i) => (
                  <motion.div
                    key={exp.id}
                    className="grid grid-cols-12 gap-3 px-4 py-3 items-center cursor-pointer hover:bg-muted/60 transition-colors group"
                    onClick={() => navigate(`/experiments/${exp.id}`)}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.02, duration: 0.3 }}
                  >
                    <div className="col-span-1 font-mono text-data-sm text-foreground-muted group-hover:text-mcp transition-colors">
                      #{exp.id}
                    </div>
                    <div className="col-span-4 font-sans text-body-sm text-foreground truncate pr-2">
                      {exp.name}
                    </div>
                    <div className="col-span-2 font-mono text-data-sm text-foreground-muted">
                      {modelMap[exp.model_id] ?? `#${exp.model_id}`}
                    </div>
                    <div className="col-span-1 text-center">
                      {exp.use_mcp && (
                        <span className="pill pill-mcp text-[10px]">MCP</span>
                      )}
                    </div>
                    <div className="col-span-2 font-mono text-data-sm text-foreground-muted truncate">
                      {fmt(exp.created_at)}
                    </div>
                    <div className="col-span-2 flex justify-end">
                      <StatusPill status={exp.status} />
                    </div>
                  </motion.div>
                ))}
              </div>
            </section>
          ))}
        </div>

        <div className="border-t border-border bg-muted/50 px-4 py-2 flex justify-between items-center font-mono text-data-sm text-foreground-muted">
          <span>Showing {visible.length} of {experiments.length}</span>
        </div>
      </div>
    </div>
  );
}
