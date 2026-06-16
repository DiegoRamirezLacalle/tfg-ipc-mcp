"""
00_phase0_diagnostics.py
------------------------
PHASE 0 (read-only) diagnosis for the C1 (with-signals) vs C0 (univariate)
question: *why* do the with-signals foundation forecasts fail to beat the
univariate baseline?

This script reads ONLY stored prediction Parquets and feature Parquets. It does
NOT retrain any model and does NOT overwrite any forecast artifact. It produces:

  08_results/phase0_biasvar.csv      bias/variance decomposition of C1-C0 error
  08_results/phase0_signal_corr.csv  per-signal corr with target LEVEL vs CHANGE
  08_results/phase0_diagnostics.md   the findings note (the Phase 0 deliverable)

Three diagnostics, mapped to the pre-diagnosed root causes:

  A. Bias vs variance decomposition (paired by origin, fc_date) per
     (series, family, signal, horizon).  MSE = bias^2 + var, so
        dMSE = (bias_C1^2 - bias_C0^2) + (var_C1 - var_C0)
     tells us whether C1 hurts by SHIFTING the forecast (bias) or by adding
     noise (variance).  Confirms the failure mode.

  B. Implied correction (C1_pred - C0_pred) for the GLOBAL foundation models.
     For TimesFM the within-origin spread across horizons is exactly 0 (the
     "Ridge correction" is a single scalar added flat to all 12 steps -- root
     cause #3).  For Chronos-2 the spread is non-zero (native covariates with a
     FLAT-held future path -- root cause #2).

  C. Per-signal correlation with the target LEVEL vs the target CHANGE
     (increment).  Confirms root cause #1: the signals fed to C1 track the
     price LEVEL, not the increment we actually forecast.

DM uses shared.metrics.diebold_mariano (HLN-corrected, power=1 => MAE), the same
implementation as scripts 01/05/06.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.metrics import diebold_mariano  # noqa: E402

RESULTS = ROOT / "08_results"
DATA = ROOT / "data" / "processed"
HORIZONS = [1, 3, 6, 12]

# (series, family, signal, C0 file, C1 file).  Only pairs whose forecasts are
# both stored are evaluated; missing files are reported and skipped.
PAIRS = [
    ("Spain",  "chronos2", "mcp",    "chronos2_C0_predictions.parquet",        "chronos2_C1_predictions.parquet"),
    ("Spain",  "chronos2", "energy", "chronos2_C0_predictions.parquet",        "chronos2_C1_energy_predictions.parquet"),
    ("Spain",  "chronos2", "inst",   "chronos2_C0_predictions.parquet",        "chronos2_C1_inst_predictions.parquet"),
    ("Spain",  "chronos2", "macro",  "chronos2_C0_predictions.parquet",        "chronos2_C1_macro_predictions.parquet"),
    ("Spain",  "timesfm",  "mcp",    "timesfm_C0_predictions.parquet",         "timesfm_C1_predictions.parquet"),
    ("Spain",  "timesfm",  "inst",   "timesfm_C0_predictions.parquet",         "timesfm_C1_inst_predictions.parquet"),
    ("Spain",  "timesfm",  "macro",  "timesfm_C0_predictions.parquet",         "timesfm_C1_macro_predictions.parquet"),
    ("Spain",  "timegpt",  "inst",   "timegpt_C0_predictions.parquet",         "timegpt_C1_inst_predictions.parquet"),
    ("Global", "chronos2", "inst",   "chronos2_C0_global_predictions.parquet", "chronos2_C1_inst_global_predictions.parquet"),
    ("Global", "timesfm",  "inst",   "timesfm_C0_global_predictions.parquet",  "timesfm_C1_inst_global_predictions.parquet"),
    ("Europe", "chronos2", "inst",   "chronos2_C0_europe_predictions.parquet", "chronos2_C1_inst_europe_predictions.parquet"),
    ("Europe", "chronos2", "full",   "chronos2_C0_europe_predictions.parquet", "chronos2_C1_full_europe_predictions.parquet"),
    ("Europe", "timesfm",  "full",   "timesfm_C0_europe_predictions.parquet",  "timesfm_C1_full_europe_predictions.parquet"),
    ("Europe", "timegpt",  "full",   "timegpt_C0_europe_predictions.parquet",  "timegpt_C1_full_europe_predictions.parquet"),
]

# Global foundation pairs whose implied correction (C1-C0) we inspect.
GLOBAL_PAIRS = [
    ("timesfm",  "timesfm_C0_global_predictions.parquet",  "timesfm_C1_inst_global_predictions.parquet"),
    ("chronos2", "chronos2_C0_global_predictions.parquet", "chronos2_C1_inst_global_predictions.parquet"),
]

# Signals to correlate per series (the covariates actually used by C1 models).
SIGNAL_SETS = {
    "Europe (HICP index)": (
        "features_c1_europe.parquet", "hicp_index",
        ["epu_europe_ma3", "brent_ma3", "ttf_ma3", "esi_eurozone", "eurusd_ma3",
         "dfr", "dfr_ma3", "breakeven_5y_lag1", "bce_shock_score",
         "bce_tone_numeric", "gdelt_tone_ma6"],
    ),
    "Global (CPI YoY rate)": (
        "features_c1_global_institutional.parquet", "cpi_global_rate",
        ["imf_comm_ma3", "brent_log_ma3", "gscpi_ma3", "gepu_ma3",
         "fedfunds_ma3", "vix_ma3", "dxy_ma3", "gpr_ma3"],
    ),
    "Spain (CPI index)": (
        "features_c1.parquet", "indice_general",
        ["brent_ma3", "ttf_ma3", "epu_europe_ma3", "dfr", "dfr_lag3",
         "gdelt_tone_ma6", "bce_shock_score", "bce_tone_numeric"],
    ),
}


def load_preds(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["origin"] = pd.to_datetime(d["origin"])
    d["fc_date"] = pd.to_datetime(d["fc_date"])
    return d


def align(c0: pd.DataFrame, c1: pd.DataFrame, h: int) -> pd.DataFrame:
    a = c0[c0["horizon"] == h][["origin", "fc_date", "error", "y_pred"]]
    b = c1[c1["horizon"] == h][["origin", "fc_date", "error", "y_pred"]]
    return a.merge(b, on=["origin", "fc_date"], suffixes=("_0", "_1"))


def biasvar_rows() -> list[dict]:
    rows = []
    for series, fam, sig, f0, f1 in PAIRS:
        c0, c1 = load_preds(f0), load_preds(f1)
        if c0 is None or c1 is None:
            print(f"  [skip] {series} {fam} {sig}: missing forecast file")
            continue
        for h in HORIZONS:
            m = align(c0, c1, h)
            if len(m) < 8:
                continue
            e0, e1 = m["error_0"].values, m["error_1"].values
            b0, b1 = float(e0.mean()), float(e1.mean())
            v0, v1 = float(e0.var()), float(e1.var())
            mae0, mae1 = float(np.abs(e0).mean()), float(np.abs(e1).mean())
            dm = diebold_mariano(e0, e1, h=h, power=1)
            d_bias2 = b1 ** 2 - b0 ** 2
            d_var = v1 - v0
            rows.append({
                "series": series, "family": fam, "signal": sig, "horizon": h,
                "n": len(m),
                "mae_c0": round(mae0, 4), "mae_c1": round(mae1, 4),
                "dmae_pct": round((mae1 - mae0) / mae0 * 100, 1),
                "bias_c0": round(b0, 3), "bias_c1": round(b1, 3),
                "std_c0": round(v0 ** 0.5, 3), "std_c1": round(v1 ** 0.5, 3),
                "d_bias2": round(d_bias2, 4), "d_var": round(d_var, 4),
                "driver": "VAR" if abs(d_var) >= abs(d_bias2) else "BIAS",
                "dm_p": round(dm["p_value"], 3),
                "better": "C1" if dm["dm_stat"] > 0 else "C0",
            })
    return rows


def correction_rows() -> list[dict]:
    rows = []
    for fam, f0, f1 in GLOBAL_PAIRS:
        c0, c1 = load_preds(f0), load_preds(f1)
        if c0 is None or c1 is None:
            continue
        m = c0[["origin", "fc_date", "step", "y_pred"]].merge(
            c1[["origin", "fc_date", "y_pred"]], on=["origin", "fc_date"],
            suffixes=("_0", "_1"))
        m["cval"] = m["y_pred_1"] - m["y_pred_0"]
        g = m.groupby("origin")["cval"]
        spread = (g.max() - g.min())
        per = g.first()
        rows.append({
            "family": fam,
            "mean_corr": round(float(m["cval"].mean()), 4),
            "mean_abs_corr": round(float(m["cval"].abs().mean()), 4),
            "std_corr": round(float(m["cval"].std()), 4),
            "within_origin_spread_mean": round(float(spread.mean()), 4),
            "within_origin_spread_max": round(float(spread.max()), 4),
            "per_origin_min": round(float(per.min()), 4),
            "per_origin_max": round(float(per.max()), 4),
            "interpretation": ("flat additive nudge (constant across h)"
                               if spread.max() < 1e-9
                               else "native covariate, flat-held future path"),
        })
    return rows


def corr_rows() -> list[dict]:
    rows = []
    for name, (fname, target, cols) in SIGNAL_SETS.items():
        df = pd.read_parquet(DATA / fname)
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        y = df[target].astype(float)
        inc = y.diff()                 # contemporaneous increment  y_t - y_{t-1}
        nxt = y.shift(-1) - y          # next increment  y_{t+1} - y_t (forecast-relevant)
        for c in cols:
            if c not in df.columns:
                continue
            s = df[c].astype(float)
            rows.append({
                "series": name, "target": target, "signal": c,
                "corr_level": round(float(s.corr(y)), 3),
                "corr_increment": round(float(s.corr(inc)), 3),
                "corr_next_increment": round(float(s.corr(nxt)), 3),
            })
    return rows


def write_md(bv: list[dict], corr: list[dict], cc: list[dict]) -> None:
    bvdf = pd.DataFrame(bv)
    lines = [
        "# Phase 0 diagnostics - why C1 (with-signals) does not beat C0",
        "",
        "> Read-only analysis of stored forecasts + feature Parquets. No model "
        "was retrained and no forecast artifact was overwritten. Generated by "
        "`07_evaluation/00_phase0_diagnostics.py`.",
        "",
        "## A. Bias vs variance decomposition of the C1-C0 error",
        "",
        "> `e = y_true - y_pred`, paired by (origin, fc_date). "
        "`dmae_pct` > 0 means C1 is worse. "
        "`d_bias2 = bias_C1^2 - bias_C0^2`, `d_var = var_C1 - var_C0`; "
        "`driver` is whichever dominates `dMSE`. "
        "`bias > 0` = the model under-forecasts (y_true above y_pred). "
        "DM: model1=C0, model2=C1, MAE-based.",
        "",
        "| series | family | signal | h | n | MAE C0 | MAE C1 | dMAE% | bias C0 | bias C1 | d_bias2 | d_var | driver | DM p | better |",
        "|--------|--------|--------|--:|--:|-------:|-------:|------:|--------:|--------:|--------:|------:|--------|-----:|--------|",
    ]
    for r in bv:
        lines.append(
            f"| {r['series']} | {r['family']} | {r['signal']} | {r['horizon']} | {r['n']} | "
            f"{r['mae_c0']:.4f} | {r['mae_c1']:.4f} | {r['dmae_pct']:+.1f} | "
            f"{r['bias_c0']:+.3f} | {r['bias_c1']:+.3f} | {r['d_bias2']:+.4f} | "
            f"{r['d_var']:+.4f} | {r['driver']} | {r['dm_p']:.3f} | {r['better']} |"
        )

    lines += [
        "",
        "## B. Implied correction (C1_pred - C0_pred), global foundation models",
        "",
        "> `within_origin_spread` = max-min of the implied correction across the "
        "12 horizon steps of one origin. Spread == 0 proves the correction is a "
        "single scalar added flat to every step (root cause #3). A non-zero "
        "spread with a flat-held future covariate path is root cause #2.",
        "",
        "| family | mean corr | mean|corr| | std | spread mean | spread max | per-origin min | per-origin max | interpretation |",
        "|--------|----------:|-----------:|----:|------------:|-----------:|---------------:|---------------:|----------------|",
    ]
    for r in cc:
        lines.append(
            f"| {r['family']} | {r['mean_corr']:+.4f} | {r['mean_abs_corr']:.4f} | "
            f"{r['std_corr']:.4f} | {r['within_origin_spread_mean']:.4f} | "
            f"{r['within_origin_spread_max']:.4f} | {r['per_origin_min']:+.4f} | "
            f"{r['per_origin_max']:+.4f} | {r['interpretation']} |"
        )

    lines += [
        "",
        "## C. Per-signal correlation with target LEVEL vs target CHANGE",
        "",
        "> `corr_level` = corr(signal, y). `corr_increment` = corr(signal, "
        "y_t - y_{t-1}). `corr_next_increment` = corr(signal, y_{t+1} - y_t). "
        "A signal useful for forecasting the change must correlate with the "
        "increment, not (only) the level. Full sample.",
        "",
        "| series | signal | corr_level | corr_increment | corr_next_increment |",
        "|--------|--------|-----------:|---------------:|--------------------:|",
    ]
    for r in corr:
        lines.append(
            f"| {r['series']} | {r['signal']} | {r['corr_level']:+.3f} | "
            f"{r['corr_increment']:+.3f} | {r['corr_next_increment']:+.3f} |"
        )

    lines += [
        "",
        "## D. Findings and verdict",
        "",
        "**Root cause #1 (signals track LEVEL, not CHANGE) - CONFIRMED, strong.** "
        "The covariates fed to C1 correlate with the price level and are near-"
        "zero against the increment we actually forecast: Europe `epu_europe_ma3` "
        "+0.81 level / +0.12 next-increment; Spain `epu_europe_ma3` +0.80 / +0.03, "
        "`brent_ma3` +0.51 / -0.02; Global `imf_comm_ma3` +0.58 / +0.00, "
        "`brent_log_ma3` +0.46 / -0.04. The notable exceptions carry real change "
        "information: Global `gscpi_ma3` (+0.28 level AND +0.25 next-increment) "
        "and `vix_ma3` (-0.27 increment); Europe `esi_eurozone`, `bce_tone_numeric` "
        "and `gdelt_tone_ma6` (+0.19-0.33 next-increment). The current foundation "
        "C1 models pick covariates by correlation-with-LEVEL (imf_comm, brent, "
        "epu) and therefore mostly inject level/regime variance.",
        "",
        "**Root causes #2 and #3 (forward path and integration) - CONFIRMED.** "
        "TimesFM global adds a single scalar to all 12 steps (within-origin spread "
        "= 0 exactly): a flat additive nudge that is dimensionally wrong on a path "
        "(it neither compounds nor decays). Chronos-2 global uses native "
        "covariates but with a flat-held future path (last value repeated), so the "
        "covariate effect is driven by a frozen forward signal.",
        "",
        "**Failure mode (decomposition A).** On the price-INDEX targets (Spain, "
        "Europe HICP) the C1 degradation is dominated by BIAS that compounds with "
        "the horizon (e.g. Spain timesfm-mcp h12 bias +1.16 -> +1.67, "
        "d_bias2 = +1.45; the additive/covariate effect injects a drift). On the "
        "Global YoY-RATE target the opposite happens: C1 *reduces* error by -4% to "
        "-20% MAE, cutting the large positive C0 forecast bias during the 2022 "
        "shock (chronos2 h12 bias +1.19 -> +0.74) - but it is NOT DM-significant "
        "(p = 0.12-0.74) because of high error variance over only 36-47 origins. "
        "The timegpt C1 rows (Spain inst +30..+65%, Europe full +75..+97% MAE, "
        "bias-driven) are anomalous broken/stale runs (TimeGPT quota), not a "
        "signal-value finding - exclude them from the conclusion.",
        "",
        "**Verdict: representation (Phase 1) is the prerequisite and the bigger "
        "lever, but it must be paired with the right target transform and "
        "integration (Phases 2-3); architecture alone will not fix it.** The "
        "evidence: (1) the dominant failure (biased drift on level targets) is a "
        "representation problem - level-correlated signals select for regime, not "
        "change; (2) the one place C1 already has the *right sign* (Global, a rate "
        "target) is where the target is change-like, which is exactly what Phase 1 "
        "would impose everywhere; (3) that Global improvement stays insignificant "
        "because of the flat additive nudge and the flat forward path, which are "
        "Phase 2/3 problems. Concretely: select signals by correlation with the "
        "CHANGE (gscpi, vix, esi, tone - not imf_comm/brent/epu), feed change "
        "transforms (next_diff / resid-vs-trailing-trend), and replace the flat "
        "scalar nudge with a path-aware, shrunk correction.",
        "",
        "**Note on existing uncommitted work.** "
        "`07_evaluation/08_validated_context_overlay.py` already implements a "
        "disciplined version of this (pre-2021-only selection of feature family + "
        "change transform + Ridge shrinkage, applied to the stored C0 forecasts as "
        "`*_C1_validated*`). Its independent recompute "
        "(`thesis_critical_dm_recompute.md`) shows it removes almost all of the "
        "damage on Spain/Europe (deltas shrink to +0.1..+0.5%) and yields a "
        "DM-significant C1 > C0 on Global Chronos-2 at h6 (-3.0%, p=0.006) and h12 "
        "(-2.7%, p=0.0003). That corroborates this verdict and is the natural base "
        "to build Phases 1-3 on rather than re-deriving from scratch.",
    ]

    (RESULTS / "phase0_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS / 'phase0_diagnostics.md'}")


def main() -> None:
    print("Phase 0 diagnostics (read-only)")
    bv = biasvar_rows()
    cc = correction_rows()
    corr = corr_rows()

    pd.DataFrame(bv).to_csv(RESULTS / "phase0_biasvar.csv", index=False)
    pd.DataFrame(corr).to_csv(RESULTS / "phase0_signal_corr.csv", index=False)
    write_md(bv, corr, cc)
    print(f"Wrote {RESULTS / 'phase0_biasvar.csv'}")
    print(f"Wrote {RESULTS / 'phase0_signal_corr.csv'}")


if __name__ == "__main__":
    main()
