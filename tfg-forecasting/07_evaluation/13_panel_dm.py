"""
13_panel_dm.py — Phase A: pooled (panel) predictive-accuracy test
-----------------------------------------------------------------
The per-series Diebold–Mariano tests are underpowered (~36–47 origins each), so
real effects show "right sign, not significant". This pools the three CPI series
(Spain, Global, Europe) into ONE test to gain power, for the Chronos-2 family.

Method
------
For a contrast (model A vs model B) at horizon h, per series s:
  d_{s,i} = |e^A_{s,i}| - |e^B_{s,i}|        (loss differential; >0 => B better)
Series live on different scales (Spain index, Global rate, Europe HICP), so each
series' d is made commensurable by dividing by that series' own MASE denominator
(mean |y_t - y_{t-12}| on the 2002–2020 training window, recovered exactly from
the stored metrics as MAE/MASE).

The pooled scaled differentials are tested with a **cluster-robust mean test
clustered by origin** (the panel-DM analog): clustering by origin absorbs both
the serial dependence of the multi-step path AND the cross-series correlation at
a shared forecast date in one estimator. Number of clusters G = number of
origins (~36–47), enough for a usable t-reference (df = G-1). This is the honest
inference here — see the caveat printed at the end about cross-sectional clusters.

No retraining: reads only stored prediction Parquets and metric JSONs.

Output:
  08_results/panel_dm.csv
  08_results/panel_dm.md
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

# Per series: prediction files + model labels for C0, flat-hold C1, forward-path
# C1, and the metrics file used to recover the MASE scale (= MAE/MASE at h1).
SERIES = {
    "Spain": {
        "c0":   ("chronos2_C0_predictions.parquet",            "chronos2_C0"),
        "flat": ("chronos2_C1_inst_predictions.parquet",       "chronos2_C1_inst"),
        "fwd":  ("chronos2_C1_fwd_spain_predictions.parquet",  "chronos2_C1_fwd_spain"),
        "scale": ("chronos2_C0_metrics.json",                  "chronos2_C0"),
    },
    "Global": {
        "c0":   ("chronos2_C0_global_predictions.parquet",        "chronos2_C0_global"),
        "flat": ("chronos2_C1_inst_global_predictions.parquet",   "chronos2_C1_inst_global"),
        "fwd":  ("chronos2_C1_fwd_global_predictions.parquet",    "chronos2_C1_fwd_global"),
        "scale": ("chronos2_C0_global_metrics.json",             "chronos2_C0_global"),
    },
    "Europe": {
        "c0":   ("chronos2_C0_europe_predictions.parquet",        "chronos2_C0_europe"),
        "flat": ("chronos2_C1_inst_europe_predictions.parquet",   "chronos2_C1_inst_europe"),
        "fwd":  ("chronos2_C1_fwd_europe_predictions.parquet",    "chronos2_C1_fwd_europe"),
        "scale": ("chronos2_C0_europe_metrics.json",             "chronos2_C0_europe"),
    },
}

# Pooled contrasts: (label, A-role, B-role). d>0 => B better. A is the reference.
CONTRASTS = [
    ("C1 context vs C0",          "c0",   "flat"),
    ("C1 forward-path vs C0",     "c0",   "fwd"),
    ("C1 forward vs flat-hold",   "flat", "fwd"),
]


def _load_preds(fname: str, model: str) -> pd.DataFrame | None:
    p = RESULTS / fname
    if not p.exists():
        logger.warning("[!] missing predictions: %s", fname)
        return None
    df = pd.read_parquet(p)
    df = df[df["model"] == model].copy()
    if df.empty:
        logger.warning("[!] model %s not in %s", model, fname)
        return None
    df["origin"] = pd.to_datetime(df["origin"])
    df["fc_date"] = pd.to_datetime(df["fc_date"])
    df["abserr"] = df["error"].abs()
    return df


def _mase_scale(fname: str, key: str) -> float | None:
    p = RESULTS / fname
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8")).get(key, {})
    h1 = d.get("h1", {})
    mae, mase = h1.get("MAE"), h1.get("MASE")
    if mae and mase:
        return mae / mase
    return None


def _scaled_diffs(series_cfg: dict, role_a: str, role_b: str, h: int) -> pd.DataFrame | None:
    """Aligned, per-series-scaled loss differentials for one series/contrast/horizon.

    Returns rows with columns: origin, d (scaled), and the raw |e_a|,|e_b| sums
    are returned via attributes for per-series dMAE%.
    """
    a = _load_preds(*series_cfg[role_a])
    b = _load_preds(*series_cfg[role_b])
    scale = _mase_scale(*series_cfg["scale"])
    if a is None or b is None or not scale:
        return None
    a = a[a["horizon"] == h][["origin", "fc_date", "abserr"]]
    b = b[b["horizon"] == h][["origin", "fc_date", "abserr"]]
    m = a.merge(b, on=["origin", "fc_date"], suffixes=("_a", "_b"))
    if len(m) < 4:
        return None
    out = pd.DataFrame({
        "origin": m["origin"].values,
        "d": (m["abserr_a"].values - m["abserr_b"].values) / scale,
        "abs_a": m["abserr_a"].values,
        "abs_b": m["abserr_b"].values,
    })
    return out


def _cluster_robust_mean_test(d: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float, int]:
    """One-sample cluster-robust mean test (intercept-only, CR1 by cluster).

    Returns (d_bar, t_stat, p_value, n_clusters). df = G-1.
    """
    n = len(d)
    d_bar = float(np.mean(d))
    resid = d - d_bar
    uniq = np.unique(clusters)
    G = len(uniq)
    meat = 0.0
    for g in uniq:
        s = float(resid[clusters == g].sum())
        meat += s * s
    if G < 2 or meat <= 0:
        return d_bar, math.nan, math.nan, G
    # CR1 small-sample adjustment for an intercept-only model: G/(G-1).
    var = (G / (G - 1)) * meat / (n * n)
    t = d_bar / math.sqrt(var)
    p = 2.0 * stats.t.sf(abs(t), df=G - 1)
    return d_bar, float(t), float(p), G


def _sig(p: float) -> str:
    if not math.isfinite(p):
        return "NA"
    return "**" if p < 0.05 else ("*" if p < 0.10 else "ns")


def _panel_row(label: str, role_a: str, role_b: str, horizons: list[int]) -> dict | None:
    parts, per_series = [], {}
    for sname, cfg in SERIES.items():
        frames = [_scaled_diffs(cfg, role_a, role_b, h) for h in horizons]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        f = pd.concat(frames, ignore_index=True)
        f["origin"] = pd.to_datetime(f["origin"])
        f["series"] = sname
        parts.append(f)
        sa, sb = f["abs_a"].sum(), f["abs_b"].sum()
        per_series[sname] = round((sb - sa) / sa * 100, 1) if sa else None
    if not parts:
        return None
    pooled = pd.concat(parts, ignore_index=True)
    # Cluster by calendar origin: all three series' rows sharing a forecast origin
    # fall in one cluster, so the estimator absorbs BOTH the cross-series
    # correlation at a shared date and the within-path serial correlation across
    # the pooled horizons/steps. Clusters G = number of distinct origins.
    clusters = pooled["origin"].values.astype("datetime64[ns]").astype("int64")
    d_bar, t, p, G = _cluster_robust_mean_test(pooled["d"].values, clusters)
    return {
        "contrast": label,
        "horizons": "+".join(str(h) for h in horizons) if len(horizons) > 1 else str(horizons[0]),
        "n": int(len(pooled)),
        "clusters": G,
        "pooled_scaled_dbar": round(d_bar, 4),
        "direction": "B_better" if d_bar > 0 else "A_better",
        "t": None if not math.isfinite(t) else round(t, 3),
        "p_value": None if not math.isfinite(p) else round(p, 4),
        "sig": _sig(p),
        "Spain_dMAEpct": per_series.get("Spain"),
        "Global_dMAEpct": per_series.get("Global"),
        "Europe_dMAEpct": per_series.get("Europe"),
    }


def main() -> None:
    logger.info("=" * 64)
    logger.info("PHASE A — PANEL DM (Chronos-2, pooled Spain+Global+Europe)")
    logger.info("=" * 64)

    rows: list[dict] = []
    for label, ra, rb in CONTRASTS:
        for h in HORIZONS:
            r = _panel_row(label, ra, rb, [h])
            if r:
                rows.append(r)
        r_all = _panel_row(label, ra, rb, HORIZONS)  # all-horizon pooled headline
        if r_all:
            rows.append(r_all)

    _write_csv(rows)
    _write_md(rows)

    logger.info("\nHeadline (all-horizon pooled):")
    for r in rows:
        if "+" in r["horizons"]:
            better = "context/fwd better" if r["direction"] == "B_better" else "reference better"
            logger.info("  %-26s dbar=%+.4f  p=%s %s  (%s; G=%d, n=%d)",
                        r["contrast"], r["pooled_scaled_dbar"],
                        r["p_value"], r["sig"], better, r["clusters"], r["n"])
    logger.info("\nDone. %d rows.", len(rows))


def _write_csv(rows: list[dict]) -> None:
    out = RESULTS / "panel_dm.csv"
    cols = ["contrast", "horizons", "n", "clusters", "pooled_scaled_dbar",
            "direction", "t", "p_value", "sig",
            "Spain_dMAEpct", "Global_dMAEpct", "Europe_dMAEpct"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    logger.info("CSV saved: %s", out.name)


def _f(v, d=4):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def _write_md(rows: list[dict]) -> None:
    out = RESULTS / "panel_dm.md"
    lines = [
        "# Panel Diebold–Mariano — pooled across Spain + Global + Europe (Chronos-2)",
        "",
        "> Phase A. Pools the three CPI series into one test to gain power. Loss "
        "differential `d = |e_A| - |e_B|` (>0 => B better) is scaled per series by "
        "its MASE denominator so the units are commensurable, then tested with a "
        "**cluster-robust mean test clustered by origin** (panel-DM analog; df = "
        "clusters-1). `dbar` = pooled scaled mean (in MASE units). ** p<0.05, * "
        "p<0.10. Per-series columns are raw dMAE% (B vs A; negative = B better).",
        "",
        "| Contrast | h | n | clusters | pooled dbar | dir | t | p | sig | Spain | Global | Europe |",
        "|---|--:|--:|--:|--:|---|--:|--:|---|--:|--:|--:|",
    ]
    for r in rows:
        dirn = "B>A" if r["direction"] == "B_better" else "A>B"
        lines.append(
            f"| {r['contrast']} | {r['horizons']} | {r['n']} | {r['clusters']} | "
            f"{_f(r['pooled_scaled_dbar'])} | {dirn} | {_f(r['t'],3)} | {_f(r['p_value'],4)} | "
            f"{r['sig']} | {_f(r['Spain_dMAEpct'],1)} | {_f(r['Global_dMAEpct'],1)} | "
            f"{_f(r['Europe_dMAEpct'],1)} |"
        )
    lines += [
        "",
        "## Caveat (read before citing)",
        "",
        "- This pools only **three** series. Clustering by origin gives ~36–47 "
        "*time* clusters (adequate for the t-reference), but the **cross-sectional** "
        "dimension is still 3 — a genuinely robust panel needs ~20–40 countries "
        "(Phase B). Treat a clean pooled point estimate here as the green light for "
        "Phase B, not as definitive inference.",
        "- B = the with-context / forward-path model, A = the reference (C0 or "
        "flat-hold). `dir = B>A` means context/forward-path has the lower error.",
        "- No retraining: pooled from stored Chronos-2 forecasts only.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("MD  saved: %s", out.name)


if __name__ == "__main__":
    main()
