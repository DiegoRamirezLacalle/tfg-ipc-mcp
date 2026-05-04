"""
tabla_maestra_modelos.py
------------------------
Generates the master table of all TFG models in HTML format.

Structure:
  Country (Global / Spain)
    Tier (C0 no signals | C1 with signals)
      Signal type (— / energy / energy_only / macro / institutional / energy+macro)
        Model

Metrics: MAE h=1,3,6,12 + delta% vs best C0 baseline per country.
Green delta = improvement, red delta = worse.

Usage:
    python tabla_maestra_modelos.py             # generate HTML and open in browser
    python tabla_maestra_modelos.py --no-open  # generate file only
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS = ROOT / "08_results"
OUT_HTML = RESULTS / "tabla_maestra.html"
OUT_CSV = RESULTS / "tabla_maestra.csv"

HORIZONS = ["h1", "h3", "h6", "h12"]

# ---------------------------------------------------------------------------
# Models: (country, tier, signal_type, display_name, file, internal_key)
# ---------------------------------------------------------------------------

MODELS = [
    # ===== GLOBAL =====
    ("Global", "C0", "—",             "Naive (lag-12)",  "rolling_metrics_global.json",          "naive"),
    ("Global", "C0", "—",             "ARIMA(3,1,0)",    "rolling_metrics_global.json",          "arima"),
    ("Global", "C0", "—",             "ARIMA(1,1,1)",    "rolling_metrics_global.json",          "arima111"),
    ("Global", "C0", "—",             "ARIMAX C0",       "rolling_metrics_global.json",          "arimax"),
    ("Global", "C0", "—",             "LSTM",            "deep_rolling_metrics_global.json",     "lstm"),
    ("Global", "C0", "—",             "N-BEATS",         "deep_rolling_metrics_global.json",     "nbeats"),
    ("Global", "C0", "—",             "N-HiTS",          "deep_rolling_metrics_global.json",     "nhits"),
    ("Global", "C0", "—",             "Chronos-2",       "chronos2_C0_metrics.json",             "chronos2_C0"),
    ("Global", "C0", "—",             "TimesFM",         "timesfm_C0_metrics.json",              "timesfm_C0"),
    ("Global", "C0", "—",             "TimeGPT",         "timegpt_C0_metrics.json",              "timegpt_C0"),
    ("Global", "C1", "institutional", "ARIMAX C1_inst",  "rolling_metrics_C1_inst_global.json",  "arimax_C1_inst"),
    ("Global", "C1", "institutional", "Chronos-2",       "chronos2_C1_inst_global_metrics.json", "chronos2_C1_inst_global"),
    ("Global", "C1", "institutional", "TimesFM",         "timesfm_C1_inst_global_metrics.json",  "timesfm_C1_inst_global"),
    ("Global", "C1", "institutional", "TimeGPT",         "timegpt_C1_inst_global_metrics.json",  "timegpt_C1_inst_global"),

    # ===== SPAIN =====
    ("Spain", "C0", "—",             "Naive (lag-12)",  "metrics_summary_final.json",           "naive"),
    ("Spain", "C0", "—",             "ARIMA",           "metrics_summary_final.json",           "arima"),
    ("Spain", "C0", "—",             "SARIMA",          "metrics_summary_final.json",           "sarima"),
    ("Spain", "C0", "—",             "SARIMAX C0",      "metrics_summary_final.json",           "sarimax"),
    ("Spain", "C0", "—",             "LSTM",            "metrics_summary_final.json",           "lstm"),
    ("Spain", "C0", "—",             "N-BEATS",         "metrics_summary_final.json",           "nbeats"),
    ("Spain", "C0", "—",             "N-HiTS",          "metrics_summary_final.json",           "nhits"),
    ("Spain", "C0", "—",             "Chronos-2",       "metrics_summary_final.json",           "chronos2_C0"),
    ("Spain", "C0", "—",             "TimesFM",         "metrics_summary_final.json",           "timesfm_C0"),
    ("Spain", "C0", "—",             "TimeGPT",         "metrics_summary_final.json",           "timegpt_C0"),
    ("Spain", "C1", "energy",        "Chronos-2",       "metrics_summary_final.json",           "chronos2_C1_energy"),
    ("Spain", "C1", "energy",        "TimeGPT",         "metrics_summary_final.json",           "timegpt_C1_energy"),
    ("Spain", "C1", "energy only",   "Chronos-2",       "metrics_summary_final.json",           "chronos2_C1_energy_only"),
    ("Spain", "C1", "energy only",   "TimeGPT",         "metrics_summary_final.json",           "timegpt_C1_energy_only"),
    ("Spain", "C1", "macro",         "Chronos-2",       "metrics_summary_final.json",           "chronos2_C1_macro"),
    ("Spain", "C1", "macro",         "TimesFM",         "metrics_summary_final.json",           "timesfm_C1_macro"),
    ("Spain", "C1", "macro",         "TimeGPT",         "metrics_summary_final.json",           "timegpt_C1_macro"),
    ("Spain", "C1", "institutional", "Chronos-2",       "metrics_summary_final.json",           "chronos2_C1_inst"),
    ("Spain", "C1", "institutional", "TimesFM",         "metrics_summary_final.json",           "timesfm_C1_inst"),
    ("Spain", "C1", "institutional", "TimeGPT",         "metrics_summary_final.json",           "timegpt_C1_inst"),
    ("Spain", "C1", "energy+macro",  "Chronos-2",       "metrics_summary_final.json",           "chronos2_C1"),
    ("Spain", "C1", "energy+macro",  "TimesFM",         "metrics_summary_final.json",           "timesfm_C1"),
    ("Spain", "C1", "energy+macro",  "TimeGPT",         "metrics_summary_final.json",           "timegpt_C1"),
]

# ---------------------------------------------------------------------------
# Metric loading
# ---------------------------------------------------------------------------

_CACHE: dict[str, dict] = {}


def _load(fname: str) -> dict:
    if fname not in _CACHE:
        p = RESULTS / fname
        _CACHE[fname] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _CACHE[fname]


def _mae(fname: str, key: str, h: str) -> float | None:
    d = _load(fname)
    hd = d.get(key, {}).get(h, {})
    return hd.get("MAE") if isinstance(hd, dict) else None


def _best_c0(country: str) -> dict[str, float]:
    best: dict[str, float] = {}
    for m_country, tier, _, _, fname, key in MODELS:
        if m_country != country or tier != "C0":
            continue
        for h in HORIZONS:
            v = _mae(fname, key, h)
            if v is not None and (h not in best or v < best[h]):
                best[h] = v
    return best


def build_rows() -> list[dict]:
    rows = []
    for country, tier, signal_type, model_name, fname, key in MODELS:
        row: dict = {
            "Country": country,
            "Tier": tier,
            "Signals": signal_type,
            "Model": model_name,
        }
        ok = False
        for h in HORIZONS:
            v = _mae(fname, key, h)
            row[f"MAE_{h}"] = v
            if v is not None:
                ok = True
        if ok:
            rows.append(row)
    return rows


def add_deltas(rows: list[dict]) -> list[dict]:
    refs = {country: _best_c0(country) for country in ("Global", "Spain")}
    for row in rows:
        ref = refs[row["Country"]]
        for h in HORIZONS:
            mae = row[f"MAE_{h}"]
            ref_mae = ref.get(h)
            if mae is not None and ref_mae and ref_mae > 0:
                row[f"d%_{h}"] = round((mae - ref_mae) / ref_mae * 100, 1)
            else:
                row[f"d%_{h}"] = None
    return rows


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f2f5;
    padding: 24px;
    color: #1a1a2e;
}
h1 {
    font-size: 1.5rem;
    margin-bottom: 6px;
    color: #1a1a2e;
}
.subtitle {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 20px;
}
.legend {
    display: flex;
    gap: 18px;
    margin-bottom: 16px;
    font-size: 0.78rem;
    align-items: center;
}
.legend-dot {
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 3px;
    margin-right: 4px;
    vertical-align: middle;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    padding: 10px 14px;
    margin-top: 28px;
    margin-bottom: 0;
    border-radius: 6px 6px 0 0;
    letter-spacing: 0.04em;
}
.global-header { background: #1a1a2e; color: #fff; }
.espana-header { background: #c60b1e; color: #fff; }

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
    background: #fff;
    border-radius: 0 0 6px 6px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 8px;
}
thead tr {
    background: #2d2d44;
    color: #fff;
}
thead th {
    padding: 9px 10px;
    text-align: center;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    border-right: 1px solid #444;
}
thead th:last-child { border-right: none; }
thead th.left { text-align: left; }

tbody tr:hover { background: #f5f7ff; }
td {
    padding: 7px 10px;
    border-bottom: 1px solid #eee;
    border-right: 1px solid #eee;
    text-align: center;
    vertical-align: middle;
}
td:last-child { border-right: none; }
td.left { text-align: left; }

/* Tier cell */
td.tier-c0 { background: #e8f4fd; font-weight: 700; color: #1565c0; }
td.tier-c1 { background: #fef3e2; font-weight: 700; color: #e65100; }

/* Signal cell */
td.signal {
    font-size: 0.75rem;
    font-style: italic;
    color: #555;
    background: #fafafa;
}

/* Model cell */
td.model { font-weight: 500; text-align: left; padding-left: 14px; }
td.best-row td { font-weight: 700; }

/* MAE cell */
td.mae { font-family: 'Consolas', monospace; font-size: 0.82rem; }

/* Delta cells */
td.delta-neg { color: #1b7f3a; font-weight: 700; background: #eafaf1; }
td.delta-pos-lo { color: #666; }
td.delta-pos-mid { color: #c97a00; font-weight: 600; background: #fffbeb; }
td.delta-pos-hi { color: #c0392b; font-weight: 700; background: #fdf0ef; }
td.delta-zero { color: #888; }

/* Best model highlight */
tr.best-model td { background: #e8fdf0 !important; }
tr.best-model td.tier-c0 { background: #c8f0dc !important; }
tr.best-model td.tier-c1 { background: #c8f0dc !important; }

/* Separator rows */
tr.tier-separator td {
    background: #f7f7f7;
    font-size: 0.75rem;
    font-weight: 600;
    color: #888;
    padding: 4px 10px;
    border-bottom: 1px solid #ddd;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
"""

TIER_LABELS = {
    "C0": "C0 — No exogenous signals",
    "C1": "C1 — With exogenous signals",
}

SIGNAL_LABELS = {
    "—": "—",
    "energy": "Energy (Brent, gas)",
    "energy only": "Energy only",
    "macro": "Macro (PPI, rates, USD)",
    "institutional": "Institutional (Fed, ECB, commodities)",
    "energy+macro": "Energy + Macro (full)",
}


def _delta_class(v: float | None) -> str:
    if v is None:
        return ""
    if v <= 0:
        return "delta-neg"
    if v <= 10:
        return "delta-pos-lo"
    if v <= 50:
        return "delta-pos-mid"
    return "delta-pos-hi"


def _fmt_mae(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def _find_best_mae_rows(rows: list[dict], country: str, h: str) -> set[int]:
    best_val = None
    best_idxs = set()
    for i, row in enumerate(rows):
        if row["Country"] != country:
            continue
        v = row.get(f"MAE_{h}")
        if v is not None:
            if best_val is None or v < best_val:
                best_val = v
                best_idxs = {i}
            elif v == best_val:
                best_idxs.add(i)
    return best_idxs


def build_html(rows: list[dict]) -> str:
    # Find overall best MAE per country
    best_h12 = {}
    for country in ("Global", "Spain"):
        country_rows = [r for r in rows if r["Country"] == country]
        best_val = min(
            (r["MAE_h12"] for r in country_rows if r["MAE_h12"] is not None),
            default=None,
        )
        best_h12[country] = best_val

    sections_html = ""

    for country in ("Global", "Spain"):
        hdr_class = "global-header" if country == "Global" else "espana-header"
        country_rows = [r for r in rows if r["Country"] == country]

        sections_html += f'<div class="section-header {hdr_class}">{country}</div>\n'
        sections_html += "<table>\n"
        sections_html += """<thead><tr>
  <th class="left" style="width:50px">Tier</th>
  <th class="left" style="width:140px">Signals</th>
  <th class="left" style="width:150px">Model</th>
  <th>h=1 MAE</th><th>Δ%</th>
  <th>h=3 MAE</th><th>Δ%</th>
  <th>h=6 MAE</th><th>Δ%</th>
  <th>h=12 MAE</th><th>Δ%</th>
</tr></thead>\n<tbody>\n"""

        prev_tier = ""
        prev_tipo = ""

        for row in country_rows:
            tier = row["Tier"]
            tipo = row["Signals"]
            mod = row["Model"]

            is_best = (row["MAE_h12"] is not None and row["MAE_h12"] == best_h12[country])
            tr_class = ' class="best-model"' if is_best else ""

            # Separator when tier/signal group changes
            if tier != prev_tier or tipo != prev_tipo:
                label = TIER_LABELS.get(tier, tier)
                if tipo != "—":
                    sig_label = SIGNAL_LABELS.get(tipo, tipo)
                    label = f"{label} &nbsp;›&nbsp; {sig_label}"
                sections_html += (
                    f'<tr class="tier-separator"><td colspan="11">{label}</td></tr>\n'
                )
                prev_tier = tier
                prev_tipo = tipo

            tier_class = "tier-c0" if tier == "C0" else "tier-c1"

            sections_html += f"<tr{tr_class}>\n"
            sections_html += f'  <td class="{tier_class}">{tier}</td>\n'
            sections_html += f'  <td class="signal">{tipo}</td>\n'
            sections_html += f'  <td class="model">{mod}{"  ★" if is_best else ""}</td>\n'

            for h in HORIZONS:
                mae = row.get(f"MAE_{h}")
                delta = row.get(f"d%_{h}")
                dc = _delta_class(delta)
                sections_html += f'  <td class="mae">{_fmt_mae(mae)}</td>\n'
                sections_html += f'  <td class="{dc}">{_fmt_delta(delta)}</td>\n'

            sections_html += "</tr>\n"

        sections_html += "</tbody></table>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Master Model Table — TFG IPC/MCP</title>
<style>{CSS}</style>
</head>
<body>
<h1>Master Model Table — TFG IPC/MCP</h1>
<p class="subtitle">
  Backtesting rolling expanding-window (2021-01 → 2024-12) &nbsp;|&nbsp;
  Metric: MAE &nbsp;|&nbsp;
  Δ% = relative difference vs. best C0 baseline for the same country &nbsp;|&nbsp;
  ★ = best overall model per country (h=12)
</p>
<div class="legend">
  <span><span class="legend-dot" style="background:#eafaf1;border:1px solid #1b7f3a"></span> Δ% negative = improvement</span>
  <span><span class="legend-dot" style="background:#fffbeb;border:1px solid #c97a00"></span> Δ% 10–50% worse</span>
  <span><span class="legend-dot" style="background:#fdf0ef;border:1px solid #c0392b"></span> Δ% &gt;50% worse</span>
  <span><span class="legend-dot" style="background:#e8fdf0;border:1px solid #1b7f3a"></span> ★ Best overall h=12</span>
</div>
{sections_html}
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict]) -> None:
    import csv
    cols = ["Country", "Tier", "Signals", "Model"]
    for h in HORIZONS:
        cols += [f"MAE_{h}", f"d%_{h}"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true",
                        help="Do not open in browser")
    args = parser.parse_args()

    rows = build_rows()
    rows = add_deltas(rows)

    html = build_html(rows)
    OUT_HTML.write_text(html, encoding="utf-8")
    save_csv(rows)

    logger.info(f"HTML saved: {OUT_HTML}")
    logger.info(f"CSV  saved: {OUT_CSV}")

    if not args.no_open:
        webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
