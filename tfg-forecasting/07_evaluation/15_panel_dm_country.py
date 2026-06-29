"""
15_panel_dm_country.py - Phase B: euro-area country panel DM
------------------------------------------------------------
Pools the ~19 euro-area HICP countries (chronos2_panel_europe_*) into one
cluster-robust predictive-accuracy test, the powered version of Phase A
(13_panel_dm.py, which had only 3 series).

Same method as Phase A: per-country loss differential d = |e_A| - |e_B|
(>0 => B better), scaled by each country's own MASE denominator so units are
commensurable, then a cluster-robust mean test. Two clusterings are reported:
  * by ORIGIN (time): absorbs the cross-country correlation of a shared
    monthly shock; ~47 clusters each now holding 19 countries (the power gain).
  * by COUNTRY: robustness check; 19 clusters.
The honest p-value is the smaller-cluster (more conservative) one.

Also reports how many of the N countries individually improve (a model-free
sign count), which is robust to the scaling choice.

Reads only the stored panel forecasts (no retraining).

Output:
  08_results/panel_dm_country.csv
  08_results/panel_dm_country.md
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
PRED = RESULTS / "chronos2_panel_europe_predictions.parquet"
METRICS = RESULTS / "chronos2_panel_europe_metrics.json"

# (label, A model, B model). d = |e_A|-|e_B| > 0 => B better.
CONTRASTS = [
    ("C1 context vs C0",        "C0",      "C1_inst"),
    ("C1 forward-path vs C0",   "C0",      "C1_fwd"),
    ("C1 forward vs flat-hold", "C1_inst", "C1_fwd"),
]


def _cluster_robust(d: np.ndarray, clusters: np.ndarray) -> tuple[float, float, int]:
    """One-sample cluster-robust mean test (intercept-only, CR1). -> (t, p, G)."""
    n = len(d)
    d_bar = float(np.mean(d))
    resid = d - d_bar
    uniq = np.unique(clusters)
    G = len(uniq)
    meat = sum(float(resid[clusters == g].sum()) ** 2 for g in uniq)
    if G < 2 or meat <= 0:
        return math.nan, math.nan, G
    var = (G / (G - 1)) * meat / (n * n)
    t = d_bar / math.sqrt(var)
    return t, 2.0 * stats.t.sf(abs(t), df=G - 1), G


def _sig(p: float) -> str:
    if not math.isfinite(p):
        return "NA"
    return "**" if p < 0.05 else ("*" if p < 0.10 else "ns")


def _pair(df: pd.DataFrame, a: str, b: str, h: int) -> pd.DataFrame:
    """Aligned |errors| of models a,b at horizon h, per (country,origin,fc_date)."""
    sa = df[(df["model"] == a) & (df["horizon"] == h)][["country", "origin", "fc_date", "abs_error"]]
    sb = df[(df["model"] == b) & (df["horizon"] == h)][["country", "origin", "fc_date", "abs_error"]]
    return sa.merge(sb, on=["country", "origin", "fc_date"], suffixes=("_a", "_b"))


def _row(df: pd.DataFrame, scales: dict, label: str, a: str, b: str, horizons: list[int]) -> dict | None:
    frames = [_pair(df, a, b, h) for h in horizons]
    frames = [f for f in frames if len(f)]
    if not frames:
        return None
    m = pd.concat(frames, ignore_index=True)
    m["scale"] = m["country"].map(scales)
    m["d"] = (m["abs_error_a"] - m["abs_error_b"]) / m["scale"]
    m["origin"] = pd.to_datetime(m["origin"])

    d = m["d"].to_numpy()
    t_o, p_o, g_o = _cluster_robust(d, m["origin"].values.astype("datetime64[ns]").astype("int64"))
    t_c, p_c, g_c = _cluster_robust(d, m["country"].astype("category").cat.codes.to_numpy())
    p_honest = max(p_o, p_c)  # the more conservative of the two clusterings

    # per-country sign count (scale-free): total |error| B minus A; <0 => B better
    sums = m.groupby("country")[["abs_error_a", "abs_error_b"]].sum()
    per = sums["abs_error_b"] - sums["abs_error_a"]
    n_country = len(per)
    n_b_better = int((per < 0).sum())

    d_bar = float(np.mean(d))
    return {
        "contrast": label,
        "horizons": "+".join(map(str, horizons)) if len(horizons) > 1 else str(horizons[0]),
        "n": int(len(m)), "countries": n_country,
        "pooled_scaled_dbar": round(d_bar, 4),
        "direction": "B_better" if d_bar > 0 else "A_better",
        "p_origin_clustered": None if not math.isfinite(p_o) else round(p_o, 4),
        "p_country_clustered": None if not math.isfinite(p_c) else round(p_c, 4),
        "p_honest": None if not math.isfinite(p_honest) else round(p_honest, 4),
        "sig_honest": _sig(p_honest),
        "n_countries_B_better": f"{n_b_better}/{n_country}",
    }


def main() -> None:
    if not PRED.exists():
        logger.error("[!] %s missing - run 36_chronos2_panel_europe.py first.", PRED.name)
        raise SystemExit(1)
    df = pd.read_parquet(PRED)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    scales = {c: metrics[c]["mase_scale"] for c in metrics}

    logger.info("=" * 64)
    logger.info("PHASE B - COUNTRY PANEL DM (%d countries)", df["country"].nunique())
    logger.info("=" * 64)

    rows = []
    for label, a, b in CONTRASTS:
        for h in HORIZONS:
            r = _row(df, scales, label, a, b, [h])
            if r:
                rows.append(r)
        r_all = _row(df, scales, label, a, b, HORIZONS)
        if r_all:
            rows.append(r_all)

    _write_csv(rows)
    _write_md(rows, df["country"].nunique())

    logger.info("\nHeadline (all-horizon pooled):")
    for r in rows:
        if "+" in r["horizons"]:
            logger.info("  %-26s dbar=%+.4f  p_honest=%s %s  (%s better; %s countries)",
                        r["contrast"], r["pooled_scaled_dbar"], r["p_honest"],
                        r["sig_honest"], "B" if r["direction"] == "B_better" else "A",
                        r["n_countries_B_better"])
    logger.info("\nDone. %d rows.", len(rows))


def _write_csv(rows: list[dict]) -> None:
    out = RESULTS / "panel_dm_country.csv"
    cols = ["contrast", "horizons", "n", "countries", "pooled_scaled_dbar", "direction",
            "p_origin_clustered", "p_country_clustered", "p_honest", "sig_honest",
            "n_countries_B_better"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    logger.info("CSV saved: %s", out.name)


def _f(v, d=4):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def _write_md(rows: list[dict], n_countries: int) -> None:
    out = RESULTS / "panel_dm_country.md"
    lines = [
        f"# Phase B - euro-area country panel DM ({n_countries} countries, Chronos-2)",
        "",
        "> Powered version of Phase A: pools ~19 euro-area HICP countries (shared "
        "euro covariates) instead of 3 series. Loss differential `d=|e_A|-|e_B|` "
        "(>0 => B better) scaled per country by its MASE denominator, then a "
        "cluster-robust mean test. `p_honest` = the more conservative of "
        "origin-clustered (time, ~47 clusters x19 countries) and country-clustered "
        "(19 clusters). `B>A` means context/forward-path has lower error. "
        "** p<0.05, * p<0.10.",
        "",
        "| Contrast | h | n | pooled dbar | dir | p (time) | p (country) | p honest | sig | countries B better |",
        "|---|--:|--:|--:|---|--:|--:|--:|---|--:|",
    ]
    for r in rows:
        dirn = "B>A" if r["direction"] == "B_better" else "A>B"
        lines.append(
            f"| {r['contrast']} | {r['horizons']} | {r['n']} | {_f(r['pooled_scaled_dbar'])} | "
            f"{dirn} | {_f(r['p_origin_clustered'],4)} | {_f(r['p_country_clustered'],4)} | "
            f"{_f(r['p_honest'],4)} | {r['sig_honest']} | {r['n_countries_B_better']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- B = with-context / forward-path; A = reference (C0 or flat-hold).",
        "- `countries B better` is a scale-free sign count (how many of the N "
        "countries have lower total |error| under B) - robust to the MASE scaling.",
        "- Covariates are the shared euro-area set (EPU Europe, Brent, ECB rate, "
        "ESI, EUR/USD), common to all members; per-country covariates would be a "
        "further extension.",
        "- No retraining: pooled from stored Chronos-2 panel forecasts only.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("MD  saved: %s", out.name)


if __name__ == "__main__":
    main()
