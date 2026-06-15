import { useMemo, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ArrowDown, ArrowUp } from "lucide-react";

import { modelColor } from "@/lib/modelColors";
import type { ComparisonRunData } from "@/hooks/useComparisonData";

type SortKey = "mae" | "rmse" | "mape" | "skill" | "duration";

interface LeaderboardProps {
  runs: ComparisonRunData[];
  hoveredModel: string | null;
  onHoverModel: (slug: string | null) => void;
}

function fmtNum(v: number | null, decimals = 4): string {
  return v === null ? "-" : v.toFixed(decimals);
}

function fmtDuration(sec: number | null): string {
  if (sec === null) return "-";
  if (sec < 60) return `${sec.toFixed(1)}s`;
  return `${(sec / 60).toFixed(1)}m`;
}

function fmtPct(v: number | null): string {
  return v === null ? "-" : `${(v * 100).toFixed(1)}%`;
}

export function Leaderboard({ runs, hoveredModel, onHoverModel }: LeaderboardProps) {
  const [sortKey, setSortKey] = useState<SortKey>("mae");
  const [asc, setAsc] = useState(true);

  const bests = useMemo(() => {
    const best: Record<SortKey, number | null> = { mae: null, rmse: null, mape: null, skill: null, duration: null };
    runs.forEach((r) => {
      if (r.mae !== null && (best.mae === null || r.mae < best.mae)) best.mae = r.mae;
      if (r.rmse !== null && (best.rmse === null || r.rmse < best.rmse)) best.rmse = r.rmse;
      if (r.mape !== null && (best.mape === null || r.mape < best.mape)) best.mape = r.mape;
      if (r.skill !== null && (best.skill === null || r.skill > best.skill)) best.skill = r.skill;
      if (r.durationSec !== null && (best.duration === null || r.durationSec < best.duration)) best.duration = r.durationSec;
    });
    return best;
  }, [runs]);

  const sorted = useMemo(() => {
    const copy = [...runs];
    copy.sort((a, b) => {
      let av: number | null, bv: number | null;
      switch (sortKey) {
        case "mae":      av = a.mae;        bv = b.mae;        break;
        case "rmse":     av = a.rmse;       bv = b.rmse;       break;
        case "mape":     av = a.mape;       bv = b.mape;       break;
        case "skill":    av = a.skill;      bv = b.skill;      break;
        case "duration": av = a.durationSec; bv = b.durationSec; break;
      }
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return asc ? av - bv : bv - av;
    });
    return copy;
  }, [runs, sortKey, asc]);

  const toggle = (k: SortKey) => {
    if (sortKey === k) setAsc((v) => !v);
    else {
      setSortKey(k);
      setAsc(k === "skill" ? false : true);
    }
  };

  if (runs.length === 0) return null;

  return (
    <div className="overflow-hidden rounded border border-border">
      <div className="grid grid-cols-[36px_2fr_1.4fr_56px_1fr_1fr_1fr_1fr_1fr] gap-2 px-3 py-2.5 bg-muted border-b border-border items-center">
        <div className="micro">#</div>
        <div className="micro">Experiment</div>
        <div className="micro">Model</div>
        <div className="micro text-center">MCP</div>
        <SortHeader label="MAE" active={sortKey === "mae"} asc={asc} onClick={() => toggle("mae")} />
        <SortHeader label="RMSE" active={sortKey === "rmse"} asc={asc} onClick={() => toggle("rmse")} />
        <SortHeader label="MAPE" active={sortKey === "mape"} asc={asc} onClick={() => toggle("mape")} />
        <SortHeader label="Skill" active={sortKey === "skill"} asc={asc} onClick={() => toggle("skill")} />
        <SortHeader label="Runtime" active={sortKey === "duration"} asc={asc} onClick={() => toggle("duration")} />
      </div>
      <div>
        <AnimatePresence initial={false}>
          {sorted.map((row, i) => {
            const isHovered = hoveredModel === row.modelSlug;
            const isDimmed = hoveredModel !== null && !isHovered;
            return (
              <motion.div
                layout
                key={row.runId}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: isDimmed ? 0.45 : 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                onMouseEnter={() => onHoverModel(row.modelSlug)}
                onMouseLeave={() => onHoverModel(null)}
                className={`grid grid-cols-[36px_2fr_1.4fr_56px_1fr_1fr_1fr_1fr_1fr] gap-2 px-3 py-2.5 items-center border-b border-border last:border-b-0 transition-colors ${
                  isHovered ? "bg-muted/80" : "hover:bg-muted/50"
                }`}
              >
                <div className="font-mono text-data-sm text-foreground-muted">
                  {i === 0 ? <span className="text-gold">1st</span> : `${i + 1}`}
                </div>
                <div className="font-sans text-body-sm text-foreground truncate">{row.experimentName}</div>
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: modelColor(row.modelSlug, i) }} />
                  <span className="font-mono text-data-sm truncate" style={{ color: modelColor(row.modelSlug, i) }}>
                    {row.modelSlug}
                  </span>
                </div>
                <div className="text-center">
                  {row.useMcp && <span className="pill pill-mcp text-[10px]"></span>}
                </div>
                <ValueCell value={row.mae} best={bests.mae} compare="lt" />
                <ValueCell value={row.rmse} best={bests.rmse} compare="lt" />
                <ValueCell value={row.mape} best={bests.mape} compare="lt" formatter={(v) => `${v.toFixed(2)}%`} />
                <SkillCell value={row.skill} best={bests.skill} />
                <DurationCell value={row.durationSec} best={bests.duration} />
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}

function SortHeader({ label, active, asc, onClick }: { label: string; active: boolean; asc: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`micro text-right flex items-center justify-end gap-1 cursor-pointer hover:text-foreground transition-colors ${
        active ? "text-foreground" : "text-foreground-muted"
      }`}
    >
      {label}
      {active && (asc ? <ArrowUp size={9} /> : <ArrowDown size={9} />)}
    </button>
  );
}

interface ValueCellProps {
  value: number | null;
  best: number | null;
  compare: "lt" | "gt";
  formatter?: (v: number) => string;
}

function ValueCell({ value, best, compare, formatter }: ValueCellProps) {
  if (value === null) return <span className="font-mono text-data-sm text-foreground-subtle text-right">-</span>;
  const isBest =
    best !== null && (compare === "lt" ? value === best : value === best);
  const text = formatter ? formatter(value) : value.toFixed(4);
  return (
    <div className="text-right">
      {isBest ? (
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-gold/15 border border-gold/40 font-mono text-data-sm text-gold tabular-nums">
          {text}
        </span>
      ) : (
        <span className="font-mono text-data-sm text-foreground tabular-nums">{text}</span>
      )}
    </div>
  );
}

function SkillCell({ value, best }: { value: number | null; best: number | null }) {
  if (value === null) return <span className="font-mono text-data-sm text-foreground-subtle text-right block">-</span>;
  const isBest = best !== null && value === best;
  const tone = value > 0 ? "text-success" : value < 0 ? "text-destructive" : "text-foreground-muted";
  const sign = value > 0 ? "+" : "";
  return (
    <div className="text-right">
      {isBest ? (
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-gold/15 border border-gold/40 font-mono text-data-sm text-gold tabular-nums">
          {sign}{(value * 100).toFixed(1)}%
        </span>
      ) : (
        <span className={`font-mono text-data-sm tabular-nums ${tone}`}>{sign}{(value * 100).toFixed(1)}%</span>
      )}
    </div>
  );
}

function DurationCell({ value, best }: { value: number | null; best: number | null }) {
  if (value === null) return <span className="font-mono text-data-sm text-foreground-subtle text-right block">-</span>;
  const isBest = best !== null && value === best;
  const text = value < 60 ? `${value.toFixed(1)}s` : `${(value / 60).toFixed(1)}m`;
  return (
    <div className="text-right">
      {isBest ? (
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-gold/15 border border-gold/40 font-mono text-data-sm text-gold tabular-nums">
          {text}
        </span>
      ) : (
        <span className="font-mono text-data-sm text-foreground tabular-nums">{text}</span>
      )}
    </div>
  );
}

export const _formatters = { fmtNum, fmtPct, fmtDuration };
