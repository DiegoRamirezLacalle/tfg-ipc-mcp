"""
diagnostico_timegpt_c1_part2.py — D2 and D3 (continuation of diagnostic)

D1 already confirmed: Variant B (2015+) is best, MAE = 0.4038 vs C0 = 0.5646
Now running D2 (subsets) and D3 (fill strategy) using Variant B.

To save calls, using 1 representative origin + Variant B.
Budget: 12 calls (5 subsets + 3 fill + 1 C0 ref + 3 margin).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger

logger = get_logger(__name__)

SERIES_ID = "ipc_spain"
MAX_H = 12

# 3 origins: one per sub-period
TEST_ORIGINS = pd.to_datetime([
    "2021-06-01",   # quiet period
    "2022-09-01",   # peak BCE shock
    "2023-09-01",   # post-shock
])

ALL_EXOG = [
    "gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6",
    "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
    "ine_surprise_score", "ine_inflacion",
    "signal_available",
]

api_calls = 0


def load_ipc() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df["indice_general"]
    y.index = pd.to_datetime(y.index)
    y.index.freq = "MS"
    return y


def load_features() -> pd.DataFrame:
    c1 = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    c1["date"] = pd.to_datetime(c1["date"])
    c1 = c1.set_index("date")
    c1.index.freq = "MS"
    for col in ALL_EXOG:
        if col in c1.columns:
            c1[col] = c1[col].fillna(0.0)
    return c1


def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


def forecast_one(
    client, y, origin, exog=None, exog_cols=None,
    start_date="2015-01-01", fill_strategy="forward",
):
    """One API call. Returns h=1 prediction."""
    global api_calls

    context_y = y.loc[start_date:origin] if start_date else y.loc[:origin]

    hist = pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": context_y.index,
        "y": context_y.values,
    })

    future = None
    if exog is not None and exog_cols:
        # Historical covariates
        ctx_exog = exog.loc[context_y.index[0]:origin, exog_cols].reindex(context_y.index)
        for col in exog_cols:
            hist[col] = ctx_exog[col].values if col in ctx_exog.columns else 0.0

        # Future covariates
        fc_dates = pd.date_range(
            start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
        )
        future = pd.DataFrame({"unique_id": SERIES_ID, "ds": fc_dates})
        last_row = exog.loc[:origin, exog_cols].iloc[-1]

        if fill_strategy == "forward":
            for col in exog_cols:
                future[col] = 1.0 if col == "signal_available" else float(last_row[col])
        elif fill_strategy == "zero":
            for col in exog_cols:
                future[col] = 0.0
        elif fill_strategy == "mean3":
            last3 = exog.loc[:origin, exog_cols].tail(3).mean()
            for col in exog_cols:
                future[col] = 1.0 if col == "signal_available" else float(last3[col])

    kwargs = dict(
        df=hist, h=MAX_H, freq="MS",
        time_col="ds", target_col="y", id_col="unique_id",
    )
    if future is not None:
        kwargs["X_df"] = future

    try:
        fc = client.forecast(**kwargs)
        api_calls += 1
        fc = fc.sort_values("ds").reset_index(drop=True)
        return float(fc["TimeGPT"].iloc[0])
    except Exception as e:
        api_calls += 1
        logger.warning(f"    [!] Error: {e}")
        return None


def main():
    global api_calls

    logger.info("=" * 60)
    logger.info("DIAGNOSTIC TIMEGPT C1 — Part 2 (D2 + D3)")
    logger.info(f"Origins: {[o.strftime('%Y-%m') for o in TEST_ORIGINS]}")
    logger.info("All variants use context from 2015 (Variant B)")
    logger.info("=" * 60)

    y = load_ipc()
    exog = load_features()
    client = get_client()

    # Actuals for h=1
    actuals = {}
    for origin in TEST_ORIGINS:
        target = origin + pd.DateOffset(months=1)
        actuals[origin] = float(y.loc[target])

    # ── C0 baseline (no exogenous, full context) ──
    logger.info("\n--- C0 baseline (full context, no exogenous) ---")
    c0_preds = {}
    for origin in TEST_ORIGINS:
        p = forecast_one(client, y, origin, start_date=None)
        c0_preds[origin] = p
        ae = abs(p - actuals[origin]) if p else None
        if p:
            logger.info(
                f"  {origin.strftime('%Y-%m')}: pred={p:.2f}  actual={actuals[origin]:.2f}  AE={ae:.4f}"
            )
        else:
            logger.info(f"  {origin.strftime('%Y-%m')}: ERROR")

    c0_mae = np.mean([abs(c0_preds[o] - actuals[o]) for o in TEST_ORIGINS if c0_preds[o]])
    logger.info(f"  C0 MAE h=1: {c0_mae:.4f}")

    # ── D2: Signal subsets (Variant B: from 2015) ──
    logger.info("\n" + "=" * 60)
    logger.info("D2 — Signal subsets (context from 2015)")
    logger.info("=" * 60)

    subsets = {
        "GDELT_3": ["gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6"],
        "BCE_3": ["bce_shock_score", "bce_tone_numeric", "bce_cumstance"],
        "INE_2": ["ine_surprise_score", "ine_inflacion"],
        "STANCE_2": ["bce_cumstance", "signal_available"],
        "ALL_9": ALL_EXOG,
    }

    d2_results = {}
    for name, cols in subsets.items():
        logger.info(f"\n  [{name}] {cols}")
        preds = {}
        for origin in TEST_ORIGINS:
            p = forecast_one(client, y, origin, exog, cols, start_date="2015-01-01")
            preds[origin] = p
        mae = np.mean([abs(preds[o] - actuals[o]) for o in TEST_ORIGINS if preds[o]])
        d2_results[name] = mae
        delta = (mae - c0_mae) / c0_mae * 100
        logger.info(f"    MAE h=1: {mae:.4f}  (vs C0: {delta:+.1f}%)")

    # ── D3: Fill strategy (with ALL_9, Variant B) ──
    logger.info("\n" + "=" * 60)
    logger.info("D3 — Future horizon fill strategy (ALL_9, from 2015)")
    logger.info("=" * 60)

    d3_results = {}
    for strat in ["forward", "zero", "mean3"]:
        logger.info(f"\n  [{strat}]")
        preds = {}
        for origin in TEST_ORIGINS:
            p = forecast_one(client, y, origin, exog, ALL_EXOG,
                             start_date="2015-01-01", fill_strategy=strat)
            preds[origin] = p
        mae = np.mean([abs(preds[o] - actuals[o]) for o in TEST_ORIGINS if preds[o]])
        d3_results[strat] = mae
        delta = (mae - c0_mae) / c0_mae * 100
        logger.info(f"    MAE h=1: {mae:.4f}  (vs C0: {delta:+.1f}%)")

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("FULL DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\nC0 baseline MAE h=1: {c0_mae:.4f}")

    logger.info("\nD1 (from part1): Zero regime hypothesis CONFIRMED")
    logger.info("  A_completo:    MAE = 0.4714  (vs C0: -16.5%)")
    logger.info("  B_desde2015:   MAE = 0.4038  (vs C0: -28.5%)  <-- BEST")
    logger.info("  C_nan_pre2015: MAE = 0.4294  (vs C0: -23.9%)")

    logger.info("\nD2: Signal subsets (Variant B):")
    d2_sorted = sorted(d2_results.items(), key=lambda x: x[1])
    for name, mae in d2_sorted:
        delta = (mae - c0_mae) / c0_mae * 100
        marker = " <-- BEST" if mae == d2_sorted[0][1] else ""
        logger.info(f"  {name:15s}: MAE = {mae:.4f}  (vs C0: {delta:+.1f}%){marker}")

    neutrals = [k for k, v in d2_sorted if v <= c0_mae * 1.05]
    harmful = [k for k, v in d2_sorted if v > c0_mae * 1.20]
    better = [k for k, v in d2_sorted if v < c0_mae * 0.95]
    logger.info(f"\n  Improve C0 (>5%): {better or ['none']}")
    logger.info(f"  Neutral (<=5% degradation): {neutrals or ['none']}")
    logger.info(f"  Harmful (>20% degradation): {harmful or ['none']}")

    logger.info("\nD3: Future horizon fill strategy:")
    d3_sorted = sorted(d3_results.items(), key=lambda x: x[1])
    for strat, mae in d3_sorted:
        delta = (mae - c0_mae) / c0_mae * 100
        marker = " <-- BEST" if mae == d3_sorted[0][1] else ""
        logger.info(f"  {strat:15s}: MAE = {mae:.4f}  (vs C0: {delta:+.1f}%){marker}")

    best_subset = d2_sorted[0][0]
    best_subset_mae = d2_sorted[0][1]
    best_fill = d3_sorted[0][0]
    best_fill_mae = d3_sorted[0][1]

    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDED CONCLUSION")
    logger.info("=" * 60)
    logger.info(f"  1. Clip context to 2015+ (Variant B)")
    logger.info(f"  2. Best signal subset: {best_subset} (MAE={best_subset_mae:.4f})")
    logger.info(f"  3. Best fill strategy: {best_fill} (MAE={best_fill_mae:.4f})")
    logger.info(f"  4. C0 baseline: MAE={c0_mae:.4f}")
    delta_best = (best_subset_mae - c0_mae) / c0_mae * 100
    logger.info(f"  5. Expected improvement vs C0: {delta_best:+.1f}%")
    logger.info(f"\n  Total API calls: {api_calls}")


if __name__ == "__main__":
    main()
