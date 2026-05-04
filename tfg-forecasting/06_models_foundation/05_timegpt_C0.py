"""
05_timegpt_C0.py — TimeGPT condition C0 (historical IPC series only)

Rolling-origin backtesting identical to the rest of the models:
  - 48 origins: 2021-01 to 2024-12
  - Horizons: h = 1, 3, 6, 12
  - MASE scale: seasonal naive lag-12 over train 2002-2020

Nixtla API: client.forecast(df, h=h, freq='MS')
  df requires columns ['unique_id', 'ds', 'y'] (Nixtla long format).

Cost control:
  --test-run    runs only 5 origins to verify functionality
  --full        runs all 48 origins (default)

API key loaded from .env at the monorepo root (NIXTLA_API_KEY).
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
MODEL_NAME = "timegpt_C0"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SERIES_ID = "ipc_spain"


# API client

def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError(
            "NIXTLA_API_KEY not configured. "
            "Edit the .env file at the monorepo root."
        )
    from nixtla import NixtlaClient
    client = NixtlaClient(api_key=api_key)
    return client


# Data

def load_data() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df["indice_general"]
    y.index = pd.to_datetime(y.index)
    y.index.freq = "MS"
    return y


def to_nixtla_df(y: pd.Series) -> pd.DataFrame:
    """Convert series to Nixtla long format: unique_id, ds, y."""
    return pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": y.index,
        "y": y.values,
    })


# Rolling backtesting

def run_rolling(
    y: pd.Series,
    client,
    test_run: bool = False,
) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run:
        origins = origins[:5]
        logger.info(f"[test-run] Running with {len(origins)} origins")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="TimeGPT C0 rolling"):
        context = y.loc[:origin]
        df_input = to_nixtla_df(context)

        # API call: forecast h=MAX_H at once, slice by horizon
        try:
            fc = client.forecast(
                df=df_input,
                h=MAX_H,
                freq="MS",
                time_col="ds",
                target_col="y",
                id_col="unique_id",
            )
            # fc columns: unique_id, ds, TimeGPT
            fc = fc.sort_values("ds").reset_index(drop=True)
            pred_values = fc["TimeGPT"].values  # shape (MAX_H,)
            pred_dates = pd.to_datetime(fc["ds"].values)
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
    parser = argparse.ArgumentParser(description="TimeGPT C0 rolling backtesting")
    parser.add_argument("--test-run", action="store_true",
                        help="Run only 5 origins to verify cost/functionality")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"ROLLING BACKTESTING — {MODEL_NAME}")
    logger.info(f"Origins: {ORIGINS_START} - {ORIGINS_END} ({'5 TEST' if args.test_run else '48 FULL'})")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info("=" * 60)

    y = load_data()
    logger.info(f"IPC data: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")

    client = get_client()
    logger.info("[timegpt] Nixtla client initialized")

    df_preds, mase_scale = run_rolling(y, client, test_run=args.test_run)
    logger.info(f"\nPredictions generated: {len(df_preds)}")

    if df_preds.empty:
        logger.warning("[!] No predictions generated.")
        return

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS {MODEL_NAME}")
    logger.info("=" * 60)
    log_table(metrics)

    # Comparison vs TimesFM C0
    tfm_path = RESULTS_DIR / "timesfm_C0_metrics.json"
    if tfm_path.exists():
        with open(tfm_path) as f:
            tfm = json.load(f).get("timesfm_C0", {})
        logger.info("\n--- vs timesfm_C0 (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            tgpt = metrics.get(key, {}).get("MAE")
            tfm_mae = tfm.get(key, {}).get("MAE")
            if tgpt and tfm_mae:
                delta = tgpt - tfm_mae
                logger.info(f"  h={h}: TimeGPT={tgpt:.4f}  TimesFM={tfm_mae:.4f}  "
                            f"delta={delta:+.4f} ({delta/tfm_mae*100:+.1f}%)")

    if args.test_run:
        logger.info("\n[test-run] Test results. Run without --test-run for full backtesting.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    logger.info(f"\nPredictions: {preds_path}")

    metrics_path = RESULTS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
