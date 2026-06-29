"""
16_panel_robustness_dm.py - Phase B robustness / placebo panel DM
-----------------------------------------------------------------
Combines the canonical panel forecasts (C0, C1_inst flat-hold, C1_fwd at
phi=0.85/window=12; script 36) with the robustness variants (fwd_phi100,
fwd_w24, placebo_randsign; script 37) and runs the SAME cluster-robust pooled
test as 15_panel_dm_country.py for each forward-path variant.

Two questions:
  (1) Not tuned? -> every *informed* forward-path setting (canonical, undamped,
      window=24) should still beat flat-hold significantly.
  (2) Informed, not just non-flat? -> the random-sign PLACEBO (same magnitude,
      randomized direction) should NOT beat flat-hold, and the informed forward
      path should beat the placebo.

Reads stored forecasts only. Output:
  08_results/panel_robustness_dm.csv
  08_results/panel_robustness_dm.md
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
CANON = RESULTS / "chronos2_panel_europe_predictions.parquet"
ROB = RESULTS / "chronos2_panel_robustness_predictions.parquet"
CANON_METRICS = RESULTS / "chronos2_panel_europe_metrics.json"

FWD_CANON = "C1_fwd"  # phi=0.85, window=12 (from script 36)

# (label, A, B): d=|e_A|-|e_B|>0 => B better.
CONTRASTS = [
    ("fwd canonical vs flat-hold",      "C1_inst", "C1_fwd"),
    ("fwd canonical vs C0",             "C0",      "C1_fwd"),
    ("fwd phi=1.0 (undamped) vs flat",  "C1_inst", "fwd_phi100"),
    ("fwd window=24 vs flat",           "C1_inst", "fwd_w24"),
    ("fwd phi=1.0 vs C0",               "C0",      "fwd_phi100"),
    ("fwd window=24 vs C0",             "C0",      "fwd_w24"),
    ("PLACEBO rand-sign vs flat",       "C1_inst", "placebo_randsign"),
    ("PLACEBO rand-sign vs C0",         "C0",      "placebo_randsign"),
    ("fwd canonical vs PLACEBO",        "placebo_randsign", "C1_fwd"),
]


def _cluster_robust(d, clusters):
    n = len(d)
    d_bar = float(np.mean(d))
    resid = d - d_bar
    uniq = np.unique(clusters)
    G = len(uniq)
    meat = sum(float(resid[clusters == g].sum()) ** 2 for g in uniq)
    if G < 2 or meat <= 0:
        return math.nan, G
    var = (G / (G - 1)) * meat / (n * n)
    t = d_bar / math.sqrt(var)
    return 2.0 * stats.t.sf(abs(t), df=G - 1), G


def _sig(p):
    if not math.isfinite(p):
        return "NA"
    return "**" if p < 0.05 else ("*" if p < 0.10 else "ns")


def _pair_all_h(df, a, b):
    parts = []
    for h in HORIZONS:
        sa = df[(df["model"] == a) & (df["horizon"] == h)][["country", "origin", "fc_date", "abs_error"]]
        sb = df[(df["model"] == b) & (df["horizon"] == h)][["country", "origin", "fc_date", "abs_error"]]
        parts.append(sa.merge(sb, on=["country", "origin", "fc_date"], suffixes=("_a", "_b")))
    return pd.concat(parts, ignore_index=True)


def _row(df, scales, label, a, b):
    m = _pair_all_h(df, a, b)
    if not len(m):
        return None
    m["scale"] = m["country"].map(scales)
    m["d"] = (m["abs_error_a"] - m["abs_error_b"]) / m["scale"]
    m["origin"] = pd.to_datetime(m["origin"])
    d = m["d"].to_numpy()
    p_o, _ = _cluster_robust(d, m["origin"].values.astype("datetime64[ns]").astype("int64"))
    p_c, _ = _cluster_robust(d, m["country"].astype("category").cat.codes.to_numpy())
    p_honest = max(p_o, p_c)
    sums = m.groupby("country")[["abs_error_a", "abs_error_b"]].sum()
    per = sums["abs_error_b"] - sums["abs_error_a"]
    d_bar = float(np.mean(d))
    return {
        "contrast": label, "n": int(len(m)),
        "pooled_scaled_dbar": round(d_bar, 4),
        "direction": "B_better" if d_bar > 0 else "A_better",
        "p_time": None if not math.isfinite(p_o) else round(p_o, 4),
        "p_country": None if not math.isfinite(p_c) else round(p_c, 4),
        "p_honest": None if not math.isfinite(p_honest) else round(p_honest, 4),
        "sig_honest": _sig(p_honest),
        "n_countries_B_better": f"{int((per < 0).sum())}/{len(per)}",
    }


def main():
    for p in (CANON, ROB):
        if not p.exists():
            logger.error("[!] missing %s - run scripts 36 and 37 first.", p.name)
            raise SystemExit(1)
    df = pd.concat([pd.read_parquet(CANON), pd.read_parquet(ROB)], ignore_index=True)
    metrics = json.loads(CANON_METRICS.read_text(encoding="utf-8"))
    scales = {c: metrics[c]["mase_scale"] for c in metrics}

    logger.info("=" * 64)
    logger.info("PHASE B ROBUSTNESS / PLACEBO PANEL DM (%d countries)", df["country"].nunique())
    logger.info("=" * 64)

    rows = [r for r in (_row(df, scales, *c) for c in CONTRASTS) if r]
    _write_csv(rows)
    _write_md(rows, df["country"].nunique())

    logger.info("\nResults (all-horizon pooled, honest p):")
    for r in rows:
        logger.info("  %-32s dbar=%+.4f  p=%s %-3s  %s better  (%s)",
                    r["contrast"], r["pooled_scaled_dbar"], r["p_honest"], r["sig_honest"],
                    "B" if r["direction"] == "B_better" else "A", r["n_countries_B_better"])
    logger.info("\nDone.")


def _write_csv(rows):
    out = RESULTS / "panel_robustness_dm.csv"
    cols = ["contrast", "n", "pooled_scaled_dbar", "direction",
            "p_time", "p_country", "p_honest", "sig_honest", "n_countries_B_better"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    logger.info("CSV saved: %s", out.name)


def _f(v, d=4):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def _write_md(rows, n_countries):
    out = RESULTS / "panel_robustness_dm.md"
    lines = [
        f"# Phase B robustness + placebo - euro-area panel ({n_countries} countries, Chronos-2)",
        "",
        "> All-horizon pooled, cluster-robust (honest p = max of time- and "
        "country-clustering). `B>A` => the variant has lower error. Informed "
        "forward-path variants (canonical phi=0.85/w=12, undamped phi=1.0, w=24) "
        "should beat flat-hold; the random-sign PLACEBO (same magnitude, random "
        "direction) should NOT. ** p<0.05, * p<0.10.",
        "",
        "| Contrast | n | pooled dbar | dir | p (time) | p (country) | p honest | sig | countries B better |",
        "|---|--:|--:|---|--:|--:|--:|---|--:|",
    ]
    for r in rows:
        dirn = "B>A" if r["direction"] == "B_better" else "A>B"
        lines.append(
            f"| {r['contrast']} | {r['n']} | {_f(r['pooled_scaled_dbar'])} | {dirn} | "
            f"{_f(r['p_time'],4)} | {_f(r['p_country'],4)} | {_f(r['p_honest'],4)} | "
            f"{r['sig_honest']} | {r['n_countries_B_better']} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- **Not tuned:** if every *informed* forward-path setting beats flat-hold "
        "(B>A, significant), the result is not an artifact of phi=0.85/window=12.",
        "- **Informed, not just non-flat:** if PLACEBO vs flat-hold is ns (or A>B) "
        "while *fwd canonical vs PLACEBO* is B>A, the benefit is the informed "
        "direction of recent momentum, not merely adding a non-flat path.",
        "- Placebo is one fixed-seed random-sign realization (seed 20260629).",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("MD  saved: %s", out.name)


if __name__ == "__main__":
    main()
