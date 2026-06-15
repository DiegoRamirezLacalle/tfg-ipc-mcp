import { useMemo } from "react";
import { Activity } from "lucide-react";

import { modelColor } from "@/lib/modelColors";
import { useDmMatrix } from "@/lib/queries";
import type { DmPair } from "@/lib/queries";
import type { ComparisonRunData } from "@/hooks/useComparisonData";

interface DMMatrixProps {
  runs: ComparisonRunData[];
  hoveredModel: string | null;
  onHoverModel: (slug: string | null) => void;
}

export function DMMatrix({ runs, hoveredModel, onHoverModel }: DMMatrixProps) {
  const runIds = useMemo(
    () => runs.map((r) => r.runId).filter((id) => id > 0),
    [runs]
  );
  const dm = useDmMatrix(runIds);

  const pairMap = useMemo(() => {
    const m = new Map<string, DmPair>();
    (dm.data?.pairs ?? []).forEach((p) => {
      m.set(`${p.a_run_id}:${p.b_run_id}`, p);
      m.set(`${p.b_run_id}:${p.a_run_id}`, p);
    });
    return m;
  }, [dm.data]);

  if (dm.isLoading) {
    return (
      <div className="h-40 flex items-center justify-center gap-2 font-mono text-data-sm text-foreground-subtle">
        <Activity size={13} className="animate-pulse-soft" /> Computing Diebold-Mariano tests...
      </div>
    );
  }
  if (dm.error) {
    return (
      <p className="font-mono text-data-sm text-destructive p-4">
        {(dm.error as Error).message}
      </p>
    );
  }
  if (!dm.data || runs.length < 2) {
    return (
      <p className="font-mono text-data-sm text-foreground-subtle p-4">
        Select at least two completed runs on the same series.
      </p>
    );
  }

  const anyComparable = (dm.data.pairs ?? []).some((p) => p.comparable);

  return (
    <div className="flex flex-col gap-4">
      <p className="font-mono text-data-sm text-foreground-muted">
        Row beats column? HLN-corrected DM test on squared-error loss, t(n−1). Green = row model
        significantly better (p &lt; {dm.data.alpha}); rose = significantly worse; muted = no significant
        difference.
      </p>

      {!anyComparable && (
        <p className="font-mono text-data-sm text-warning">
          No comparable pairs - selected runs use different series or have too few aligned forecast points.
        </p>
      )}

      <div className="overflow-x-auto">
        <table className="border-collapse">
          <thead>
            <tr>
              <th className="p-2" />
              {runs.map((c, j) => (
                <th
                  key={c.runId}
                  className="p-2 align-bottom"
                  onMouseEnter={() => onHoverModel(c.modelSlug)}
                  onMouseLeave={() => onHoverModel(null)}
                >
                  <div className="flex flex-col items-center gap-1">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ background: modelColor(c.modelSlug, j) }}
                    />
                    <span
                      className="font-mono text-[10px] tabular"
                      style={{ color: modelColor(c.modelSlug, j) }}
                      title={c.experimentName}
                    >
                      {c.modelSlug}
                    </span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((row, i) => {
              const rowDimmed = hoveredModel !== null && hoveredModel !== row.modelSlug;
              return (
                <tr key={row.runId} className={rowDimmed ? "opacity-40 transition-opacity" : "transition-opacity"}>
                  <th
                    className="p-2 text-right whitespace-nowrap"
                    onMouseEnter={() => onHoverModel(row.modelSlug)}
                    onMouseLeave={() => onHoverModel(null)}
                  >
                    <span
                      className="font-mono text-[11px]"
                      style={{ color: modelColor(row.modelSlug, i) }}
                      title={row.experimentName}
                    >
                      {row.modelSlug}
                    </span>
                  </th>
                  {runs.map((col, j) => {
                    if (i === j) {
                      return (
                        <td key={col.runId} className="p-1">
                          <div className="w-14 h-10 flex items-center justify-center text-foreground-subtle font-mono text-[11px] bg-muted/40 rounded">
                            -
                          </div>
                        </td>
                      );
                    }
                    const pair = pairMap.get(`${row.runId}:${col.runId}`);
                    return (
                      <td key={col.runId} className="p-1">
                        <Cell pair={pair} rowRunId={row.runId} />
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Cell({ pair, rowRunId }: { pair: DmPair | undefined; rowRunId: number }) {
  if (!pair) {
    return <div className="w-14 h-10 flex items-center justify-center text-foreground-subtle/50 font-mono text-[11px]">-</div>;
  }
  if (!pair.comparable) {
    return (
      <div
        className="w-14 h-10 flex items-center justify-center text-foreground-subtle/50 font-mono text-[10px]"
        title={pair.reason}
      >
        n/a
      </div>
    );
  }

  const p = pair.p_value ?? 1;
  const winnerRunId =
    pair.better === "model1" ? pair.a_run_id : pair.better === "model2" ? pair.b_run_id : null;

  let tone = "bg-muted/30 text-foreground-subtle border-transparent";
  if (pair.significant && winnerRunId === rowRunId) {
    tone = "bg-success/15 text-success border-success/40";
  } else if (pair.significant && winnerRunId !== null) {
    tone = "bg-destructive/15 text-destructive border-destructive/40";
  }

  const title = `DM=${pair.dm_stat}, p=${pair.p_value}, n=${pair.n}`;

  return (
    <div
      className={`w-14 h-10 flex flex-col items-center justify-center rounded border font-mono tabular ${tone}`}
      title={title}
    >
      <span className="text-[11px] leading-none">{p < 0.001 ? "<.001" : p.toFixed(3)}</span>
      <span className="text-[8px] opacity-70 mt-0.5">{pair.significant ? "sig" : "ns"}</span>
    </div>
  );
}
