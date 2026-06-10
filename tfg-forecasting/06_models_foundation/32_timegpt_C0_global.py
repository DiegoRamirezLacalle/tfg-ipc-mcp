"""
32_timegpt_C0_global.py - TimeGPT condition C0, GLOBAL CPI (historical only)

Global counterpart of 05_timegpt_C0.py (which is Spain CPI).

Target : data/processed/cpi_global_monthly.parquet :: cpi_global_rate
Model ID / outputs : timegpt_C0_global

Nixtla API: client.forecast(df, h=h, freq='MS'), long format ['unique_id','ds','y'].
API key from .env at the monorepo root (NIXTLA_API_KEY).

A target-integrity guard verifies y_true == cpi_global_rate before writing.

Cost control:
  --test-run    runs only 5 origins
  (default)     runs all 48 origins
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
MODEL_NAME = "timegpt_C0_global"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SERIES_ID = "cpi_global"

TARGET_FILE = ROOT / "data" / "processed" / "cpi_global_monthly.parquet"
TARGET_COL = "cpi_global_rate"


def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError("NIXTLA_API_KEY not configured in .env")
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


def load_data() -> pd.Series:
    df = pd.read_parquet(TARGET_FILE)
    df.index = pd.to_datetime(df.index)
    df.index.freq = "MS"
    return df[TARGET_COL]


def to_nixtla_df(y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": SERIES_ID, "ds": y.index, "y": y.values})


def run_rolling(y: pd.Series, client, test_run: bool = False) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run:
        origins = origins[:5]
        logger.info(f"[test-run] {len(origins)} origins")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []
    for origin in tqdm(origins, desc="TimeGPT C0 GLOBAL rolling"):
        df_input = to_nixtla_df(y.loc[:origin])
        try:
            fc = client.forecast(
                df=df_input, h=MAX_H, freq="MS",
                time_col="ds", target_col="y", id_col="unique_id",
            )
            fc = fc.sort_values("ds").reset_index(drop=True)
            pred_values = fc["TimeGPT"].values
        except Exception as e:
            logger.warning(f"\n[!] Error at {origin.date()}: {e}")
            continue

        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS:
                continue
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue
            for i, (date, real, pred) in enumerate(
                zip(fc_dates, y_actual.values, pred_values[:h]), start=1
            ):
                records.append({
                    "origin": origin, "fc_date": date, "step": i, "horizon": h,
                    "model": MODEL_NAME,
                    "y_true": float(real), "y_pred": float(pred),
                    "error": float(real - pred), "abs_error": float(abs(real - pred)),
                })

    return pd.DataFrame(records), mase_scale


def assert_target_integrity(df_preds: pd.DataFrame, y: pd.Series) -> None:
    if df_preds.empty:
        raise ValueError(f"[{MODEL_NAME}] No predictions to verify.")
    expected = y.reindex(pd.to_datetime(df_preds["fc_date"]).values).values
    actual = df_preds["y_true"].values
    if np.isnan(expected).any():
        bad = df_preds.loc[np.isnan(expected), "fc_date"].tolist()[:5]
        raise ValueError(f"[{MODEL_NAME}] fc_date(s) not in target series: {bad}")
    if not np.allclose(actual, expected, atol=1e-6):
        n_bad = int(np.sum(~np.isclose(actual, expected, atol=1e-6)))
        raise ValueError(
            f"[{MODEL_NAME}] TARGET-INTEGRITY FAILURE: {n_bad} rows where y_true "
            f"!= {TARGET_COL}. Predictions NOT written (possible Spain/Global mix)."
        )
    lo, hi = float(np.nanmin(actual)), float(np.nanmax(actual))
    if lo > 40:
        raise ValueError(
            f"[{MODEL_NAME}] y_true range [{lo:.2f},{hi:.2f}] looks like Spain index "
            f"scale (80-100), not a CPI rate. Refusing to write."
        )
    logger.info(f"[{MODEL_NAME}] target-integrity OK: y_true matches {TARGET_COL} "
                f"(range [{lo:.3f}, {hi:.3f}], n={len(df_preds)})")


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


def main():
    parser = argparse.ArgumentParser(description="TimeGPT C0 GLOBAL rolling backtesting")
    parser.add_argument("--test-run", action="store_true", help="Run only 5 origins")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"ROLLING BACKTESTING - {MODEL_NAME} ({'5 TEST' if args.test_run else '48 FULL'})")
    logger.info(f"Target: {TARGET_FILE.name} :: {TARGET_COL}")
    logger.info(f"Origins: {ORIGINS_START} - {ORIGINS_END} | Horizons: {HORIZONS}")
    logger.info("=" * 60)

    y = load_data()
    logger.info(f"Data: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")

    client = get_client()
    logger.info("[timegpt] Nixtla client initialized")

    df_preds, mase_scale = run_rolling(y, client, test_run=args.test_run)
    logger.info(f"\nPredictions generated: {len(df_preds)}")
    if df_preds.empty:
        logger.warning("[!] No predictions generated.")
        return

    assert_target_integrity(df_preds, y)

    metrics = compute_metrics(df_preds, mase_scale)
    logger.info(f"\nRESULTS {MODEL_NAME}")
    log_table(metrics)

    if args.test_run:
        logger.info("\n[test-run] OK. Run without --test-run for full backtesting.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    metrics_path = RESULTS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"\nPredictions: {preds_path.name}")
    logger.info(f"Metrics:     {metrics_path.name}")


if __name__ == "__main__":
    main()
