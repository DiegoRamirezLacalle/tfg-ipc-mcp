import { Newspaper, RefreshCw, ExternalLink, Cpu, Loader2 } from "lucide-react";

import { useNewsToday, useNewsSentiment, useRefreshNews } from "@/lib/queries";

interface NewsPulseProps {
  limit?: number;
  showRefresh?: boolean;
}

function fmtRefresh(iso: string | null): string {
  if (!iso) return "never";
  const d = new Date(iso);
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  return `${Math.round(mins / 60)}h ago`;
}

export function NewsPulse({ limit = 10, showRefresh = true }: NewsPulseProps) {
  const today = useNewsToday(limit);
  const month = today.data?.latest_month ?? null;
  const sentiment = useNewsSentiment(month);
  const refresh = useRefreshNews();

  const articles = today.data?.articles ?? [];
  const empty = !today.isLoading && articles.length === 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-px bg-border rounded overflow-hidden">
      {/* -- Sentiment panel ------------------------------------------- */}
      <div className="bg-card p-5 flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <Cpu size={13} className="text-mcp" />
          <span className="micro uppercase">FinBERT Pulse</span>
          <span className="pill pill-mcp text-[10px] ml-auto">◈ MCP</span>
        </div>

        {sentiment.isLoading && month && (
          <div className="flex items-center gap-2 font-mono text-data-sm text-foreground-subtle py-4">
            <Loader2 size={12} className="animate-spin" /> Scoring with FinBERT…
          </div>
        )}

        {sentiment.data?.available ? (
          <div className="flex flex-col gap-4">
            <Gauge
              label="News sentiment"
              hint="negative ← → positive"
              value={sentiment.data.sentiment_mean ?? 0}
              min={-1}
              max={1}
              fmt={(v) => v.toFixed(2)}
            />
            <Gauge
              label="Hawkish share"
              hint="dovish ← → hawkish"
              value={sentiment.data.hawkish_score ?? 0}
              min={0}
              max={1}
              fmt={(v) => `${(v * 100).toFixed(0)}%`}
              accent
            />
            <div className="flex items-center justify-between font-mono text-[11px] text-foreground-subtle pt-1 border-t border-border">
              <span>{sentiment.data.n_articles ?? 0} articles scored</span>
              <span>{month}</span>
            </div>
          </div>
        ) : (
          !sentiment.isLoading && (
            <p className="font-mono text-data-sm text-foreground-subtle">
              {empty ? "No news cached yet." : "Sentiment not available."}
            </p>
          )
        )}

        {showRefresh && (
          <button
            onClick={() => refresh.mutate("3d")}
            disabled={refresh.isPending}
            className="mt-auto flex items-center justify-center gap-1.5 px-3 py-2 rounded border border-border font-mono text-data-sm text-foreground-muted hover:text-foreground hover:border-foreground-subtle transition-colors disabled:opacity-50"
          >
            {refresh.isPending ? (
              <><Loader2 size={11} className="animate-spin" /> Fetching GDELT…</>
            ) : (
              <><RefreshCw size={11} /> Refresh from GDELT</>
            )}
          </button>
        )}
        {refresh.data && "rate_limited" in (refresh.data as object) && (
          <p className="font-mono text-[10px] text-warning">GDELT rate-limited — try again in a few seconds.</p>
        )}
        <p className="font-mono text-[10px] text-foreground-subtle">
          updated {fmtRefresh(today.data?.last_refresh ?? null)} · {today.data?.total ?? 0} cached
        </p>
      </div>

      {/* -- Article feed ---------------------------------------------- */}
      <div className="bg-card flex flex-col">
        <div className="border-b border-border px-4 py-3 flex items-center gap-2 bg-muted/40">
          <Newspaper size={13} className="text-gold" />
          <span className="micro uppercase">Live inflation headlines</span>
          <span className="font-mono text-[10px] text-foreground-subtle ml-auto">via GDELT</span>
        </div>

        {today.isLoading && (
          <div className="p-6 font-mono text-data-sm text-foreground-subtle flex items-center gap-2">
            <Loader2 size={12} className="animate-spin" /> Loading headlines…
          </div>
        )}

        {empty && (
          <div className="p-8 flex flex-col items-center text-center gap-3">
            <p className="font-mono text-data-sm text-foreground-subtle">
              No headlines cached yet. Pull the latest inflation news from GDELT.
            </p>
            {showRefresh && (
              <button
                onClick={() => refresh.mutate("7d")}
                disabled={refresh.isPending}
                className="flex items-center gap-1.5 px-4 py-2 rounded bg-mcp/15 border border-mcp/40 font-mono text-data-sm text-mcp hover:bg-mcp/25 transition-colors disabled:opacity-50"
              >
                {refresh.isPending ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                Fetch latest news
              </button>
            )}
          </div>
        )}

        <div className="divide-y divide-border max-h-[460px] overflow-y-auto">
          {articles.map((a) => (
            <a
              key={a.url}
              href={a.url}
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-3 flex items-start gap-3 hover:bg-muted/50 transition-colors group"
            >
              <span className="font-mono text-[10px] text-foreground-subtle w-16 shrink-0 pt-0.5 tabular">
                {a.date}
              </span>
              <span className="flex-1 font-sans text-body-sm text-foreground leading-snug group-hover:text-gold transition-colors">
                {a.title}
              </span>
              <span className="font-mono text-[10px] text-foreground-subtle shrink-0 hidden sm:flex items-center gap-1 pt-0.5">
                {a.source}
                <ExternalLink size={9} className="opacity-0 group-hover:opacity-100 transition-opacity" />
              </span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}

function Gauge({
  label,
  hint,
  value,
  min,
  max,
  fmt,
  accent,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  fmt: (v: number) => string;
  accent?: boolean;
}) {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min))) * 100;
  const tone =
    accent
      ? "bg-mcp"
      : value > 0.05
      ? "bg-success"
      : value < -0.05
      ? "bg-destructive"
      : "bg-foreground-subtle";
  const valTone =
    accent
      ? "text-mcp"
      : value > 0.05
      ? "text-success"
      : value < -0.05
      ? "text-destructive"
      : "text-foreground-muted";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="font-mono text-data-sm text-foreground-muted">{label}</span>
        <span className={"font-mono text-data-base tabular " + valTone}>{fmt(value)}</span>
      </div>
      <div className="relative h-1.5 rounded-full bg-muted overflow-hidden">
        {/* zero marker for symmetric gauges */}
        {min < 0 && (
          <span className="absolute top-0 bottom-0 left-1/2 w-px bg-border-strong" />
        )}
        <span
          className={"absolute top-0 bottom-0 left-0 rounded-full transition-all duration-500 " + tone}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-foreground-subtle">{hint}</span>
    </div>
  );
}
