"""
diagnostico_timegpt_c1.py - Full diagnostic before applying fix to TimeGPT C1

4 diagnostics with a total budget of 20 API calls:
  D1: Zero regime (3 variants x 1 call = 3 calls)
  D2: Signal subsets (5 subsets x 1 call = 5 calls)
  D3: Forward-fill strategy (3 strategies x 1 call = 3 calls)
  D4: C0 baseline (1 call)
  Total: 12 calls (margin of 8 for errors/retries)

Test origins: 2022-07 to 2022-11 (5 origins, peak BCE shock period)
Metric: MAE h=1 (most sensitive to covariate changes)
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

from shared.constants import DATE_TRAIN_END
from shared.logger import get_logger

logger = get_logger(__name__)

SERIES_ID = "ipc_spain"
MAX_H = 12
TEST_ORIGINS = pd.to_datetime([
    "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01",
])

ALL_EXOG = [
    "gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6",
    "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
    "ine_surprise_score", "ine_inflacion",
    "signal_available",
]

api_calls = 0
MAX_API_CALLS = 35


# ── Data ───────────────────────────────────────────────────────

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
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError(
            "NIXTLA_API_KEY not configured. "
            "Edit the .env file at the monorepo root."
        )
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


# ── Common helpers ─────────────────────────────────────────────

def build_hist_df(
    y: pd.Series,
    exog: pd.DataFrame | None,
    origin: pd.Timestamp,
    exog_cols: list[str] | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """Build historical df for Nixtla. If exog=None, series only."""
    context_y = y.loc[:origin]
    if start_date:
        context_y = context_y.loc[start_date:]

    hist = pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": context_y.index,
        "y": context_y.values,
    })

    if exog is not None and exog_cols:
        context_exog = exog.loc[context_y.index[0]:origin, exog_cols]
        context_exog = context_exog.reindex(context_y.index)
        for col in exog_cols:
            if col in context_exog.columns:
                hist[col] = context_exog[col].values
            else:
                hist[col] = 0.0

    return hist


def build_future_df(
    origin: pd.Timestamp,
    exog: pd.DataFrame | None,
    exog_cols: list[str] | None = None,
    fill_strategy: str = "forward",
) -> pd.DataFrame | None:
    """Build future_df for Nixtla. Strategies: forward, zero, mean3."""
    if exog is None or not exog_cols:
        return None

    fc_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
    )
    future = pd.DataFrame({"unique_id": SERIES_ID, "ds": fc_dates})

    last_row = exog.loc[:origin, exog_cols].iloc[-1]

    if fill_strategy == "forward":
        for col in exog_cols:
            if col == "signal_available":
                future[col] = 1.0
            else:
                future[col] = float(last_row[col])

    elif fill_strategy == "zero":
        for col in exog_cols:
            future[col] = 0.0

    elif fill_strategy == "mean3":
        last3 = exog.loc[:origin, exog_cols].tail(3).mean()
        for col in exog_cols:
            if col == "signal_available":
                future[col] = 1.0
            else:
                future[col] = float(last3[col])

    return future


def forecast_one_origin(
    client,
    y: pd.Series,
    origin: pd.Timestamp,
    exog: pd.DataFrame | None = None,
    exog_cols: list[str] | None = None,
    start_date: str | None = None,
    fill_strategy: str = "forward",
) -> float | None:
    """Make ONE API call for an origin. Returns the h=1 prediction."""
    global api_calls
    if api_calls >= MAX_API_CALLS:
        logger.warning(f"  [!] API call limit of {MAX_API_CALLS} reached, skipping.")
        return None

    hist = build_hist_df(y, exog, origin, exog_cols, start_date)
    future = build_future_df(origin, exog, exog_cols, fill_strategy)

    kwargs = dict(
        df=hist,
        h=MAX_H,
        freq="MS",
        time_col="ds",
        target_col="y",
        id_col="unique_id",
    )
    if future is not None:
        kwargs["X_df"] = future

    try:
        fc = client.forecast(**kwargs)
        api_calls += 1
        fc = fc.sort_values("ds").reset_index(drop=True)
        return float(fc["TimeGPT"].iloc[0])  # h=1 prediction
    except Exception as e:
        api_calls += 1
        logger.warning(f"  [!] Error at {origin.date()}: {e}")
        return None


def compute_mae_h1(
    preds: list[float | None],
    actuals: list[float],
) -> float | None:
    """MAE h=1 over valid pairs."""
    pairs = [(p, a) for p, a in zip(preds, actuals) if p is not None]
    if not pairs:
        return None
    return float(np.mean([abs(p - a) for p, a in pairs]))


# ── Diagnostic 1: Zero regime ──────────────────────────────────

def diag1_zero_regime(client, y, exog):
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 1 - Zero regime pre-2015")
    logger.info("=" * 60)

    actuals = []
    for origin in TEST_ORIGINS:
        target_date = origin + pd.DateOffset(months=1)
        actuals.append(float(y.loc[target_date]))

    results = {}

    # Variant A: full dataset (282 rows, with zeros pre-2015)
    logger.info("\n[A] Full dataset (282 rows, zeros pre-2015)...")
    preds_a = []
    for origin in TEST_ORIGINS:
        p = forecast_one_origin(client, y, origin, exog, ALL_EXOG)
        preds_a.append(p)
    results["A_completo"] = compute_mae_h1(preds_a, actuals)

    # Variant B: clipped from 2015 (~120 rows, real signals only)
    logger.info("[B] Dataset from 2015 (~120 rows, real signals only)...")
    preds_b = []
    for origin in TEST_ORIGINS:
        p = forecast_one_origin(client, y, origin, exog, ALL_EXOG,
                                start_date="2015-01-01")
        preds_b.append(p)
    results["B_desde2015"] = compute_mae_h1(preds_b, actuals)

    # Variant C: NaN pre-2015 (TimeGPT interpolates)
    logger.info("[C] Full dataset, NaN pre-2015 (TimeGPT interpolates)...")
    exog_nan = exog.copy()
    mask_pre2015 = exog_nan.index < "2015-01-01"
    for col in ALL_EXOG:
        if col in exog_nan.columns and col != "signal_available":
            exog_nan.loc[mask_pre2015, col] = np.nan
    preds_c = []
    for origin in TEST_ORIGINS:
        p = forecast_one_origin(client, y, origin, exog_nan, ALL_EXOG)
        preds_c.append(p)
    results["C_nan_pre2015"] = compute_mae_h1(preds_c, actuals)

    return results, actuals


# ── Diagnostic 2: Signal subsets ───────────────────────────────

def diag2_subsets(client, y, exog):
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 2 - Signal subsets")
    logger.info("=" * 60)

    actuals = []
    for origin in TEST_ORIGINS:
        target_date = origin + pd.DateOffset(months=1)
        actuals.append(float(y.loc[target_date]))

    subsets = {
        "GDELT": ["gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6"],
        "BCE": ["bce_shock_score", "bce_tone_numeric", "bce_cumstance"],
        "INE": ["ine_surprise_score", "ine_inflacion"],
        "cumstance+avail": ["bce_cumstance", "signal_available"],
        "ALL_9": ALL_EXOG,
    }

    results = {}
    for name, cols in subsets.items():
        logger.info(f"\n[{name}] Covariates: {cols}")
        preds = []
        for origin in TEST_ORIGINS:
            p = forecast_one_origin(client, y, origin, exog, cols)
            preds.append(p)
        results[name] = compute_mae_h1(preds, actuals)

    return results, actuals


# ── Diagnostic 3: Fill strategy ────────────────────────────────

def diag3_fill_strategy(client, y, exog):
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 3 - Future horizon fill strategy")
    logger.info("=" * 60)

    actuals = []
    for origin in TEST_ORIGINS:
        target_date = origin + pd.DateOffset(months=1)
        actuals.append(float(y.loc[target_date]))

    strategies = ["forward", "zero", "mean3"]
    results = {}

    for strat in strategies:
        logger.info(f"\n[{strat}] Filling future horizon with strategy '{strat}'...")
        preds = []
        for origin in TEST_ORIGINS:
            p = forecast_one_origin(client, y, origin, exog, ALL_EXOG,
                                    fill_strategy=strat)
            preds.append(p)
        results[strat] = compute_mae_h1(preds, actuals)

    return results, actuals


# ── Diagnostic 4: C0 baseline ─────────────────────────────────

def diag4_c0_baseline(client, y):
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 4 - C0 baseline (no exogenous)")
    logger.info("=" * 60)

    actuals = []
    for origin in TEST_ORIGINS:
        target_date = origin + pd.DateOffset(months=1)
        actuals.append(float(y.loc[target_date]))

    preds = []
    for origin in TEST_ORIGINS:
        p = forecast_one_origin(client, y, origin)
        preds.append(p)

    mae = compute_mae_h1(preds, actuals)
    return mae, actuals


# ── Main ──────────────────────────────────────────────────────

def main():
    global api_calls

    logger.info("=" * 60)
    logger.info("DIAGNOSTIC TIMEGPT C1 - Root cause identification")
    logger.info(f"Test origins: {[o.strftime('%Y-%m') for o in TEST_ORIGINS]}")
    logger.info(f"API budget: {MAX_API_CALLS} calls")
    logger.info("=" * 60)

    y = load_ipc()
    exog = load_features()
    client = get_client()

    logger.info(f"IPC series: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")
    logger.info(f"C1 features: {exog.index.min().date()} - {exog.index.max().date()} ({len(exog)} obs)")

    # D4 first (1 call, baseline needed for everything else)
    mae_c0, actuals = diag4_c0_baseline(client, y)
    logger.info(f"\n  >>> C0 baseline MAE h=1: {mae_c0:.4f}")
    logger.info(f"  >>> API calls used: {api_calls}/{MAX_API_CALLS}")

    d1_results, _ = diag1_zero_regime(client, y, exog)
    logger.info(f"\n  >>> API calls used: {api_calls}/{MAX_API_CALLS}")

    # D2: optimized with 1 representative origin (2022-09)
    logger.info("\n[!] D2 optimized: using 1 representative origin (2022-09)")

    single_origin = pd.to_datetime(["2022-09-01"])
    target_date = single_origin[0] + pd.DateOffset(months=1)
    single_actual = [float(y.loc[target_date])]

    subsets = {
        "GDELT": ["gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6"],
        "BCE": ["bce_shock_score", "bce_tone_numeric", "bce_cumstance"],
        "INE": ["ine_surprise_score", "ine_inflacion"],
        "cumstance+avail": ["bce_cumstance", "signal_available"],
        "ALL_9": ALL_EXOG,
    }

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 2 - Signal subsets (origin: 2022-09)")
    logger.info("=" * 60)

    d2_results = {}
    for name, cols in subsets.items():
        if api_calls >= MAX_API_CALLS:
            logger.warning(f"  [!] Limit reached, skipping {name}")
            d2_results[name] = None
            continue
        logger.info(f"  [{name}] {cols}")
        p = forecast_one_origin(client, y, single_origin[0], exog, cols)
        if p is not None:
            d2_results[name] = abs(p - single_actual[0])
        else:
            d2_results[name] = None
    logger.info(f"\n  >>> API calls used: {api_calls}/{MAX_API_CALLS}")

    # D3: fill strategy (3 strategies x 1 origin)
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 3 - Future horizon fill strategy (origin: 2022-09)")
    logger.info("=" * 60)

    d3_results = {}
    for strat in ["forward", "zero", "mean3"]:
        if api_calls >= MAX_API_CALLS:
            logger.warning(f"  [!] Limit reached, skipping {strat}")
            d3_results[strat] = None
            continue
        logger.info(f"  [{strat}]")
        p = forecast_one_origin(client, y, single_origin[0], exog, ALL_EXOG,
                                fill_strategy=strat)
        if p is not None:
            d3_results[strat] = abs(p - single_actual[0])
        else:
            d3_results[strat] = None
    logger.info(f"\n  >>> API calls used: {api_calls}/{MAX_API_CALLS}")

    # ── Summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC TIMEGPT C1")
    logger.info("=" * 60)

    # D4 + D1
    if mae_c0:
        logger.info(f"\nC0 baseline MAE h=1 (5 origins): {mae_c0:.4f}")
    else:
        logger.info("\nC0 baseline: ERROR")

    logger.info("\nZero regime hypothesis:")
    for name, mae in d1_results.items():
        if mae is not None and mae_c0 is not None:
            delta_pct = (mae - mae_c0) / mae_c0 * 100
            logger.info(f"  {name:20s}: MAE = {mae:.4f}  (vs C0: {delta_pct:+.1f}%)")
        elif mae is not None:
            logger.info(f"  {name:20s}: MAE = {mae:.4f}")
        else:
            logger.info(f"  {name:20s}: ERROR")

    # Determine hypothesis
    a = d1_results.get("A_completo")
    b = d1_results.get("B_desde2015")
    if a is not None and b is not None:
        if b < a * 0.90:
            hyp1 = "CONFIRMED"
            hyp1_detail = f"Clipping to 2015+ reduces MAE by {(a - b) / a * 100:.1f}%"
        elif b > a * 1.10:
            hyp1 = "NOT CONFIRMED (clipping worsens)"
            hyp1_detail = f"Clipping to 2015+ INCREASES MAE by {(b - a) / a * 100:.1f}%"
        else:
            hyp1 = "INDETERMINATE (difference < 10%)"
            hyp1_detail = f"Difference: {(b - a) / a * 100:+.1f}%"
    else:
        hyp1 = "ERROR in calculation"
        hyp1_detail = ""

    logger.info(f"\n  >>> Zero regime hypothesis: {hyp1}")
    if hyp1_detail:
        logger.info(f"      {hyp1_detail}")

    # D2
    logger.info("\nIndividual signals (AE h=1, origin 2022-09):")
    d2_sorted = sorted(
        [(k, v) for k, v in d2_results.items() if v is not None],
        key=lambda x: x[1]
    )
    # C0 reference for this origin
    c0_ref_p = forecast_one_origin(client, y, single_origin[0])
    c0_ref_ae = abs(c0_ref_p - single_actual[0]) if c0_ref_p is not None else None

    if c0_ref_ae is not None:
        logger.info(f"  {'C0 (no exog)':20s}: AE = {c0_ref_ae:.4f}  (reference)")
    for name, ae in d2_sorted:
        if c0_ref_ae is not None:
            delta = (ae - c0_ref_ae) / c0_ref_ae * 100
            marker = " <-- BEST" if ae < c0_ref_ae else ""
            logger.info(f"  {name:20s}: AE = {ae:.4f}  (vs C0: {delta:+.1f}%){marker}")
        else:
            logger.info(f"  {name:20s}: AE = {ae:.4f}")

    if d2_sorted:
        neutrals = [k for k, v in d2_sorted if c0_ref_ae and v <= c0_ref_ae * 1.05]
        harmful = [k for k, v in d2_sorted if c0_ref_ae and v > c0_ref_ae * 1.20]
        logger.info(f"\n  Neutral signals (<=5% degradation): {neutrals or ['none']}")
        logger.info(f"  Harmful signals (>20% degradation): {harmful or ['none']}")

    # D3
    logger.info("\nFuture horizon fill strategy (AE h=1, origin 2022-09):")
    best_strat = None
    best_ae = float("inf")
    for strat in ["forward", "zero", "mean3"]:
        ae = d3_results.get(strat)
        if ae is not None:
            marker = ""
            if ae < best_ae:
                best_ae = ae
                best_strat = strat
                marker = " <-- BEST"
            if c0_ref_ae is not None:
                delta = (ae - c0_ref_ae) / c0_ref_ae * 100
                logger.info(f"  {strat:15s}: AE = {ae:.4f}  (vs C0: {delta:+.1f}%){marker}")
            else:
                logger.info(f"  {strat:15s}: AE = {ae:.4f}{marker}")
        else:
            logger.info(f"  {strat:15s}: ERROR")

    # Conclusion
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDED CONCLUSION")
    logger.info("=" * 60)

    best_subset = d2_sorted[0][0] if d2_sorted else "ALL_9"
    best_subset_ae = d2_sorted[0][1] if d2_sorted else None

    recs = []
    if hyp1 == "CONFIRMED":
        recs.append("Clip context to 2015+ (remove zero regime)")
    elif "NOT CONFIRMED" in hyp1:
        recs.append("Keep full context (zeros are not the problem)")

    if best_strat and best_strat != "forward":
        recs.append(f"Change fill strategy to '{best_strat}'")
    else:
        recs.append("Keep forward-fill (or best of the 3)")

    if best_subset != "ALL_9" and best_subset_ae is not None and c0_ref_ae is not None:
        if best_subset_ae < c0_ref_ae * 1.05:
            recs.append(f"Use only '{best_subset}' covariates (neutral)")
        else:
            recs.append("All signals degrade; consider falling back to C0 pure")

    for i, r in enumerate(recs, 1):
        logger.info(f"  {i}. {r}")

    logger.info(f"\n  Total API calls: {api_calls}/{MAX_API_CALLS}")


if __name__ == "__main__":
    main()
