"""
07_timegpt_C1_energy_only.py - TimeGPT C1 with energy variables only

Ablation experiment: isolate the predictive effect of energy without MCP signals.

Covariates (3 total, energy only):
    brent_ma3       # corr 0.715 with IPC(t+1) in 2015+
    brent_ret       # captures rapid energy shocks
    ttf_ma3         # corr 0.541 with IPC(t+1) in 2015+

No signal_available, no bce_*, no gdelt_*. Energy prices only.
All columns have real data from 2002 (Brent via WTI proxy,
TTF via Henry Hub proxy). No NaN.

Context: full IPC from 2002 + 3 complete energy covariates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
MAX_H = max(HORIZONS)
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "timegpt_C1_energy_only"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SERIES_ID = "ipc_spain"

EXOG_COLS = ["brent_ma3", "brent_ret", "ttf_ma3"]


# API client

def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError("NIXTLA_API_KEY not configured.")
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


# Data

def load_data() -> tuple[pd.Series, pd.DataFrame]:
    ipc_df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = ipc_df["indice_general"]
    y.index = pd.to_datetime(y.index)
    y.index.freq = "MS"

    c1 = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    c1["date"] = pd.to_datetime(c1["date"])
    c1 = c1.set_index("date")
    c1.index.freq = "MS"
    return y, c1


def build_nixtla_df(
    y: pd.Series,
    exog: pd.DataFrame,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full IPC context + 3 complete energy covariates (no NaN)."""
    context_y = y.loc[:origin]

    hist_df = pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": context_y.index,
        "y": context_y.values,
    })

    for col in EXOG_COLS:
        col_vals = exog.loc[:origin, col].reindex(context_y.index)
        hist_df[col] = col_vals.values

    last_row = exog.loc[:origin, EXOG_COLS].iloc[-1]
    fc_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
    )
    future_df = pd.DataFrame({"unique_id": SERIES_ID, "ds": fc_dates})
    for col in EXOG_COLS:
        future_df[col] = float(last_row[col])

    return hist_df, future_df


# Rolling backtesting

def run_rolling(
    y: pd.Series,
    exog: pd.DataFrame,
    client,
    test_run: bool = False,
) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run:
        origins = origins[:5]
        logger.info(f"[test-run] {len(origins)} origins")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="TimeGPT C1 energy_only"):
        hist_df, future_df = build_nixtla_df(y, exog, origin)

        try:
            fc = client.forecast(
                df=hist_df,
                X_df=future_df,
                h=MAX_H,
                freq="MS",
                time_col="ds",
                target_col="y",
                id_col="unique_id",
            )
            fc = fc.sort_values("ds").reset_index(drop=True)
            pred_values = fc["TimeGPT"].values
        except Exception as e:
            logger.warning(f"\n[!] Error at {origin.date()}: {e}")
            continue

        for h in HORIZONS:
            horizon_end = origin + pd.DateOffset(months=h)
            if horizon_end > TEST_END_TS:
                continue

            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue

            y_true = y_actual.values
            y_pred = pred_values[:h]

            for i, (date, real, pred) in enumerate(
                zip(fc_dates, y_true, y_pred), start=1
            ):
                records.append({
                    "origin": origin,
                    "fc_date": date,
                    "step": i,
                    "horizon": h,
                    "model": MODEL_NAME,
                    "y_true": float(real),
                    "y_pred": float(pred),
                    "error": float(real - pred),
                    "abs_error": float(abs(real - pred)),
                })

    return pd.DataFrame(records), mase_scale


# Metrics

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for h in HORIZONS:
        h_df = df_preds[df_preds["horizon"] == h]
        if h_df.empty:
            continue
        y_true = h_df["y_true"].values
        y_pred = h_df["y_pred"].values
        results[f"h{h}"] = {
            "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
            "n_evals": int(len(h_df["origin"].unique())),
        }
    return results


def log_table(metrics: dict) -> None:
    logger.info(f"\n{'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    logger.info("-" * 45)
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics:
            m = metrics[key]
            logger.info(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                        f"{m['MASE']:8.4f} {m['n_evals']:5d}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    n_origins = 5 if args.test_run else 48
    logger.info("=" * 60)
    logger.info(f"BACKTESTING - {MODEL_NAME}")
    logger.info(f"Origins: {n_origins} ({'TEST' if args.test_run else 'FULL'})")
    logger.info(f"Covariates: {EXOG_COLS}")
    logger.info(f"Estimated cost: {n_origins} TimeGPT API calls")
    logger.info("=" * 60)

    y, exog = load_data()
    logger.info(f"IPC data: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")

    client = get_client()

    df_preds, mase_scale = run_rolling(y, exog, client, test_run=args.test_run)
    logger.info(f"\nPredictions: {len(df_preds)}")

    if df_preds.empty:
        logger.warning("[!] No predictions generated.")
        return

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS {MODEL_NAME}")
    logger.info("=" * 60)
    log_table(metrics)

    # Comparison vs C0
    c0_path = RESULTS_DIR / "timegpt_C0_metrics.json"
    if c0_path.exists():
        with open(c0_path) as f:
            c0_m = json.load(f).get("timegpt_C0", {})
        logger.info("\n--- C0 vs C1_energy_only (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            c0_mae = c0_m.get(key, {}).get("MAE")
            c1_mae = metrics.get(key, {}).get("MAE")
            if c0_mae and c1_mae:
                delta = c1_mae - c0_mae
                logger.info(f"  h={h}: C0={c0_mae:.4f}  energy_only={c1_mae:.4f}  "
                            f"delta={delta:+.4f} ({delta/c0_mae*100:+.1f}%)")

    if args.test_run:
        logger.info("\n[test-run] OK. Run without --test-run for full backtesting.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    logger.info(f"\nPredictions: {preds_path}")

    metrics_path = RESULTS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
