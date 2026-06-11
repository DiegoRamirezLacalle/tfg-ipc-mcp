import type { ExperimentStatus, RunStatus } from "@/lib/types";

type Status = ExperimentStatus | RunStatus;

const CONFIG: Record<Status, { label: string; cls: string; pulse?: boolean }> = {
  created: { label: "Created",   cls: "pill pill-muted" },
  pending: { label: "Pending",   cls: "pill pill-muted" },
  running: { label: "Running",   cls: "pill pill-info",  pulse: true },
  done:    { label: "Completed", cls: "pill pill-success" },
  failed:  { label: "Failed",    cls: "pill pill-destructive" },
};

export function StatusPill({ status }: { status: Status }) {
  const cfg = CONFIG[status] ?? CONFIG.pending;
  return (
    <span className={cfg.cls}>
      {cfg.pulse && <span className="w-1.5 h-1.5 rounded-full bg-info animate-pulse" />}
      {cfg.label}
    </span>
  );
}
