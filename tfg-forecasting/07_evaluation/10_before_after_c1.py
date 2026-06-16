"""
10_before_after_c1.py
---------------------
Deliverable #4: the before/after comparison. For each (series, family,
horizon) it tabulates C1 (with-signals) vs C0 (univariate) MAE and the DM
p-value across every C1 integration variant, so the thesis can state plainly
where signals now help and where they do not.

Variants (all evaluated against the same foundation C0 of that series/family):
  inst/full   - the ORIGINAL with-signals model (flat-hold future, naive
                additive nudge or native covariates).
  validated   - pre-2021 Ridge context overlay (script 08).
  regime      - change-correlated + regime-gated overlay (script 09).
  fwd         - Chronos-2 native covariates with honest damped-drift forward
                paths (script 33). Global Chronos-2 only.

DM: the strict Harvey-Leybourne-Newbold adjusted Diebold-Mariano test with
abs-error (MAE) loss, identical to 07_thesis_critical_dm_recompute.py::dm_hln
(HAC lags h-1, HLN small-sample factor, two-sided Student-t p with df=n-1) --
NOT shared.metrics.diebold_mariano (which uses a circular np.roll autocovariance
and a normal reference). Paired by (origin, fc_date). Read-only on stored
forecasts.

Output: 08_results/before_after_c1.md
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]


def dm_hln_abs(error_a: np.ndarray, error_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """MAE-loss HLN-adjusted Diebold-Mariano (mirror of script 07's dm_hln).

    d = |error_a| - |error_b|; returns (dm_adj, two-sided p). NaN when the loss
    differential has no usable variance (e.g. a no-op overlay where C1 == C0).
    """
    d = np.abs(np.asarray(error_a, dtype=float)) - np.abs(np.asarray(error_b, dtype=float))
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 4:
        return math.nan, math.nan
    d_bar = float(np.mean(d))
    centered = d - d_bar
    long_run = float(np.dot(centered, centered) / n)
    for lag in range(1, horizon):
        if n <= lag:
            break
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2.0 * gamma
    var_d_bar = long_run / n
    if not math.isfinite(var_d_bar) or var_d_bar <= 0:
        return math.nan, math.nan
    dm = d_bar / math.sqrt(var_d_bar)
    hln_factor = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if hln_factor <= 0:
        return math.nan, math.nan
    dm_adj = dm * math.sqrt(hln_factor)
    p_value = 2.0 * stats.t.sf(abs(dm_adj), df=n - 1)
    return float(dm_adj), float(p_value)

# series -> family -> (C0 file, {variant: C1 file})
PLAN = {
    "Spain": {
        "chronos2": ("chronos2_C0_predictions.parquet", {
            "inst": "chronos2_C1_inst_predictions.parquet",
            "validated": "chronos2_C1_validated_predictions.parquet",
            "regime": "chronos2_C1_regime_predictions.parquet",
        }),
        "timesfm": ("timesfm_C0_predictions.parquet", {
            "inst": "timesfm_C1_inst_predictions.parquet",
            "validated": "timesfm_C1_validated_predictions.parquet",
            "regime": "timesfm_C1_regime_predictions.parquet",
        }),
    },
    "Global": {
        "chronos2": ("chronos2_C0_global_predictions.parquet", {
            "inst (flat-hold)": "chronos2_C1_inst_global_predictions.parquet",
            "validated": "chronos2_C1_validated_global_predictions.parquet",
            "regime": "chronos2_C1_regime_global_predictions.parquet",
            "fwd (forward path)": "chronos2_C1_fwd_global_predictions.parquet",
        }),
        "timesfm": ("timesfm_C0_global_predictions.parquet", {
            "inst (ridge nudge)": "timesfm_C1_inst_global_predictions.parquet",
            "validated": "timesfm_C1_validated_global_predictions.parquet",
            "regime": "timesfm_C1_regime_global_predictions.parquet",
        }),
    },
    "Europe": {
        "chronos2": ("chronos2_C0_europe_predictions.parquet", {
            "full": "chronos2_C1_full_europe_predictions.parquet",
            "regime": "chronos2_C1_regime_europe_predictions.parquet",
        }),
        "timesfm": ("timesfm_C0_europe_predictions.parquet", {
            "full": "timesfm_C1_full_europe_predictions.parquet",
            "validated": "timesfm_C1_validated_europe_predictions.parquet",
            "regime": "timesfm_C1_regime_europe_predictions.parquet",
        }),
    },
}


def load(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["origin"] = pd.to_datetime(d["origin"])
    d["fc_date"] = pd.to_datetime(d["fc_date"])
    return d


def compare(c0: pd.DataFrame, c1: pd.DataFrame, h: int) -> tuple | None:
    a = c0[c0["horizon"] == h][["origin", "fc_date", "error"]]
    b = c1[c1["horizon"] == h][["origin", "fc_date", "error"]]
    m = a.merge(b, on=["origin", "fc_date"], suffixes=("_0", "_1"))
    if len(m) < 8:
        return None
    e0, e1 = m["error_0"].values, m["error_1"].values
    mae0, mae1 = float(np.abs(e0).mean()), float(np.abs(e1).mean())
    _, p = dm_hln_abs(e0, e1, h)
    winner = "C1" if mae1 < mae0 else ("C0" if mae1 > mae0 else "=")
    return mae0, mae1, (mae1 - mae0) / mae0 * 100.0, p, winner


def cell(res: tuple | None) -> str:
    if res is None:
        return "-"
    _, _, dpct, p, better = res
    if not math.isfinite(p):
        # no usable loss differential (e.g. a no-op overlay: C1 == C0)
        return f"{dpct:+.1f}% n/a ({better})"
    star = "**" if p < 0.05 else ("*" if p < 0.10 else "")
    return f"{dpct:+.1f}% {p:.3f}{star} ({better})"


def main() -> None:
    lines = [
        "# C1 (with-signals) vs C0 (univariate) - before/after by integration variant",
        "",
        "> Each cell is `dMAE% DM_p (winner)` for that C1 variant vs the matching "
        "foundation C0, paired by (origin, fc_date), MAE-based HLN Diebold-Mariano. "
        "Negative dMAE% = C1 lower error. ** p<0.05, * p<0.10. Rolling origins "
        "2021-01..2024-12, horizons in months. Generated by "
        "`07_evaluation/10_before_after_c1.py` (read-only).",
        "",
        "| Series | Family | Variant | h=1 | h=3 | h=6 | h=12 |",
        "|--------|--------|---------|-----|-----|-----|------|",
    ]
    for series, fams in PLAN.items():
        for family, (c0file, variants) in fams.items():
            c0 = load(c0file)
            if c0 is None:
                continue
            for variant, c1file in variants.items():
                c1 = load(c1file)
                if c1 is None:
                    lines.append(f"| {series} | {family} | {variant} | - | - | - | - |")
                    continue
                cells = [cell(compare(c0, c1, h)) for h in HORIZONS]
                lines.append(f"| {series} | {family} | {variant} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## How to read it",
        "",
        "- **Global** is the only series where signals deliver genuine OOS gains. "
        "Chronos-2 with an honest forward covariate path (`fwd`) cuts MAE 14-24% "
        "vs C0 at every horizon (DM-significant only at h=1, p=0.075, because the "
        "gain is concentrated in the 2022 shock over few origins). The "
        "regime-gated overlay turns the Global Chronos-2 advantage into clean "
        "DM-significant wins at h=3/6/12 (smaller, ~1-1.6%).",
        "- **Spain and Europe (price-index targets)**: signals do not beat C0. The "
        "overlays shrink the original C1 damage to ~0; the regime gate keeps it "
        "non-significant. This is the honest finding - outside the rate target and "
        "the shock regime, exogenous/semantic signals do not improve the foundation "
        "forecast.",
        "- **TimesFM Global**: the always-on overlay (`validated`) helps (h1-6, "
        "p<0.10) but the regime gate removes that benefit, because here the signal "
        "helps broadly rather than only in high-volatility months.",
    ]
    out = RESULTS / "before_after_c1.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
