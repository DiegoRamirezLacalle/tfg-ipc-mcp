"""
12_forward_path_comparison.py
-----------------------------
Consolidated flat-hold vs forward-path covariate comparison for the Chronos-2
C1 models across the three CPI series (Spain, Global, Europe).

For each series it lays out, per horizon {1,3,6,12}:
  * C0 (univariate)              MAE / RMSE / MASE
  * C1_inst flat-hold covariates MAE / RMSE / MASE
  * C1_fwd  forward-path cov.    MAE / RMSE / MASE
  * dMAE% (fwd vs flat-hold) and dMAE% (fwd vs C0)
  * strict HLN-adjusted DM p-value (path-level, abs loss) for both contrasts,
    read from thesis_critical_dm_recompute.csv (script 07) so there is a single
    source of statistical truth. Marked [VERIFY] if that row is unavailable.

Reads only stored metrics JSON and the DM CSV. Writes:
  08_results/forward_path_comparison.csv
  08_results/forward_path_comparison.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
VERIFY = "[VERIFY]"

# series -> (C0 key/file, C1 flat key/file, C1 fwd key/file, DM comparison labels)
SERIES = {
    "Spain": {
        "c0":   ("chronos2_C0",            "chronos2_C0_metrics.json"),
        "flat": ("chronos2_C1_inst",       "chronos2_C1_inst_metrics.json"),
        "fwd":  ("chronos2_C1_fwd_spain",  "chronos2_C1_fwd_spain_metrics.json"),
    },
    "Global": {
        "c0":   ("chronos2_C0_global",           "chronos2_C0_global_metrics.json"),
        "flat": ("chronos2_C1_inst_global",      "chronos2_C1_inst_global_metrics.json"),
        "fwd":  ("chronos2_C1_fwd_global",       "chronos2_C1_fwd_global_metrics.json"),
    },
    "Europe": {
        "c0":   ("chronos2_C0_europe",           "chronos2_C0_europe_metrics.json"),
        "flat": ("chronos2_C1_inst_europe",      "chronos2_C1_inst_europe_metrics.json"),
        "fwd":  ("chronos2_C1_fwd_europe",       "chronos2_C1_fwd_europe_metrics.json"),
    },
}

# DM comparison labels in thesis_critical_dm_recompute.csv (path level, abs loss).
DM_LABELS = {
    "fwd_vs_flat": "Chronos-2 forward vs flat-hold",
    "fwd_vs_c0":   "Chronos-2 forward-path context",
}


def _load_metrics(fname: str, key: str) -> dict | None:
    p = RESULTS / fname
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(key)


def _load_dm() -> dict:
    """Map (series, comparison_label, horizon) -> (p_abs, delta_pct) from script 07."""
    p = RESULTS / "thesis_critical_dm_recompute.csv"
    out: dict[tuple, tuple] = {}
    if not p.exists():
        logger.warning("[!] %s missing - DM p-values will be %s", p.name, VERIFY)
        return out
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["level"] != "path":
                continue
            key = (row["series"], row["comparison"], int(row["horizon"]))
            try:
                p_abs = f"{float(row['p_abs']):.4f}"
            except (ValueError, TypeError):
                p_abs = row["p_abs"]
            out[key] = (p_abs, row["sig_abs"])
    return out


def _dpct(a: float | None, b: float | None) -> float | None:
    """percent change of b vs a."""
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a * 100.0


def main() -> None:
    dm = _load_dm()
    rows: list[dict] = []

    for series, spec in SERIES.items():
        c0 = _load_metrics(*reversed(spec["c0"]))     # (file, key) order for _load_metrics
        flat = _load_metrics(*reversed(spec["flat"]))
        fwd = _load_metrics(*reversed(spec["fwd"]))
        if fwd is None:
            logger.warning("[!] %s forward-path metrics missing - skipping series", series)
            continue
        for h in HORIZONS:
            hk = f"h{h}"
            c0h = (c0 or {}).get(hk, {})
            fh = (flat or {}).get(hk, {})
            wh = fwd.get(hk, {})
            p_flat, sig_flat = dm.get((series, DM_LABELS["fwd_vs_flat"], h), (VERIFY, VERIFY))
            p_c0, sig_c0 = dm.get((series, DM_LABELS["fwd_vs_c0"], h), (VERIFY, VERIFY))
            rows.append({
                "series": series, "horizon": h,
                "C0_MAE": c0h.get("MAE"), "C0_RMSE": c0h.get("RMSE"), "C0_MASE": c0h.get("MASE"),
                "flat_MAE": fh.get("MAE"), "flat_RMSE": fh.get("RMSE"), "flat_MASE": fh.get("MASE"),
                "fwd_MAE": wh.get("MAE"), "fwd_RMSE": wh.get("RMSE"), "fwd_MASE": wh.get("MASE"),
                "dMAEpct_fwd_vs_flat": _round(_dpct(fh.get("MAE"), wh.get("MAE"))),
                "dMAEpct_fwd_vs_C0": _round(_dpct(c0h.get("MAE"), wh.get("MAE"))),
                "DMp_fwd_vs_flat": p_flat, "DMsig_fwd_vs_flat": sig_flat,
                "DMp_fwd_vs_C0": p_c0, "DMsig_fwd_vs_C0": sig_c0,
            })

    _write_csv(rows)
    _write_md(rows)
    _write_tex(rows)
    logger.info("Done. %d rows.", len(rows))


def _round(v):
    return None if v is None else round(v, 1)


def _write_csv(rows: list[dict]) -> None:
    out = RESULTS / "forward_path_comparison.csv"
    cols = ["series", "horizon",
            "C0_MAE", "C0_RMSE", "C0_MASE",
            "flat_MAE", "flat_RMSE", "flat_MASE",
            "fwd_MAE", "fwd_RMSE", "fwd_MASE",
            "dMAEpct_fwd_vs_flat", "dMAEpct_fwd_vs_C0",
            "DMp_fwd_vs_flat", "DMsig_fwd_vs_flat", "DMp_fwd_vs_C0", "DMsig_fwd_vs_C0"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    logger.info("CSV saved: %s", out.name)


def _fmt(v, d=4):
    return "-" if v is None else f"{v:.{d}f}"


def _write_md(rows: list[dict]) -> None:
    out = RESULTS / "forward_path_comparison.md"
    lines = [
        "# Forward-path vs flat-hold covariates - Chronos-2 C1",
        "",
        "> Same rolling-origin protocol (origins 2021-01->2024-12, expanding window, "
        "MASE scale on 2002-2020). Only the future covariate path differs: flat-hold "
        "(last value repeated) vs forward-path (damped RW-with-drift, "
        "`shared.exog_policies.FORWARD_PATH`, data <= origin). dMAE% negative = "
        "improvement. DM p = strict HLN-adjusted, path-level, abs loss "
        "(thesis_critical_dm_recompute.csv). ** p<0.05, * p<0.10.",
        "",
    ]
    for series in SERIES:
        srows = [r for r in rows if r["series"] == series]
        if not srows:
            continue
        lines += [
            f"## {series}",
            "",
            "| h | C0 MAE | flat MAE | **fwd MAE** | fwd MASE | dMAE% fwd-vs-flat (DM p) | dMAE% fwd-vs-C0 (DM p) |",
            "|--:|------:|--------:|-----------:|--------:|-----------------------|---------------------|",
        ]
        for r in srows:
            df = r["dMAEpct_fwd_vs_flat"]
            dc = r["dMAEpct_fwd_vs_C0"]
            pf = r["DMp_fwd_vs_flat"]; sf = r["DMsig_fwd_vs_flat"]
            pc = r["DMp_fwd_vs_C0"]; sc = r["DMsig_fwd_vs_C0"]
            cell_f = f"{'-' if df is None else f'{df:+.1f}%'} ({pf}{'' if sf in ('ns', VERIFY) else sf})"
            cell_c = f"{'-' if dc is None else f'{dc:+.1f}%'} ({pc}{'' if sc in ('ns', VERIFY) else sc})"
            lines.append(
                f"| {r['horizon']} | {_fmt(r['C0_MAE'])} | {_fmt(r['flat_MAE'])} | "
                f"**{_fmt(r['fwd_MAE'])}** | {_fmt(r['fwd_MASE'])} | {cell_f} | {cell_c} |"
            )
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("MD  saved: %s", out.name)


def _tex_pct(v, p, sig):
    """LaTeX cell: '-5.7\\%$^{\\ast}$' with DM significance superscript."""
    if v is None:
        return "--"
    if sig in ("ns", VERIFY, "NA"):
        star = ""
    else:
        ast = sig.replace("*", "\\ast ").strip()
        star = "$^{" + ast + "}$"
    return f"{v:+.1f}\\%{star}"


def _write_tex(rows: list[dict]) -> None:
    """Thesis-ready booktabs tabular. Requires \\usepackage{booktabs} (and siunitx
    optional). MAE rows + the two delta columns with HLN-DM significance stars."""
    out = RESULTS / "forward_path_comparison.tex"
    lines = [
        "% Forward-path vs flat-hold covariates (Chronos-2 C1). Requires \\usepackage{booktabs}.",
        "% dMAE: negative = forward-path better. DM significance (strict HLN, path-level,",
        "% abs loss): $\\ast\\ast$ p<0.05, $\\ast$ p<0.10. Generated by 12_forward_path_comparison.py.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrr rr}",
        "\\toprule",
        "Series & $h$ & C0 & flat-hold & fwd & fwd MASE & "
        "$\\Delta$MAE fwd/flat & $\\Delta$MAE fwd/C0 \\\\",
        "\\midrule",
    ]
    for si, series in enumerate(SERIES):
        srows = [r for r in rows if r["series"] == series]
        if not srows:
            continue
        for j, r in enumerate(srows):
            lab = series if j == 0 else ""
            cell_f = _tex_pct(r["dMAEpct_fwd_vs_flat"], r["DMp_fwd_vs_flat"], r["DMsig_fwd_vs_flat"])
            cell_c = _tex_pct(r["dMAEpct_fwd_vs_C0"], r["DMp_fwd_vs_C0"], r["DMsig_fwd_vs_C0"])
            lines.append(
                f"{lab} & {r['horizon']} & {_fmt(r['C0_MAE'])} & {_fmt(r['flat_MAE'])} & "
                f"{_fmt(r['fwd_MAE'])} & {_fmt(r['fwd_MASE'])} & {cell_f} & {cell_c} \\\\"
            )
        if si < len(SERIES) - 1:
            lines.append("\\midrule")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Forward-path vs.\\ flat carry-forward covariates for the Chronos-2 "
        "context model. Rolling origins 2021-01--2024-12, MASE scale on 2002--2020. "
        "Only the future covariate path differs (flat last-value vs.\\ damped "
        "random-walk-with-drift, data $\\le$ origin). $\\Delta$MAE negative favours the "
        "forward path. Significance from the strict HLN-adjusted Diebold--Mariano test "
        "(path-level, absolute-error loss).}",
        "\\label{tab:forward_path}",
        "\\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("TeX saved: %s", out.name)


if __name__ == "__main__":
    main()
