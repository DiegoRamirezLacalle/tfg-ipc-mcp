"""
03_chronos2_C0.py - Chronos-2 condition C0 (historical only)

Rolling-origin backtesting:
  - 48 origins: 2021-01 to 2024-12
  - Horizons: h=1, 3, 6, 12
  - Metrics: MAE, RMSE, MASE (seasonal naive lag-12)

Model: amazon/chronos-2 (2nd generation, 21 quantiles: 0.01-0.99)
Output: quantiles with p10, p50, p90 for prediction intervals.
Supports native covariates via use_reg_token.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
MODEL_NAME = "chronos2_C0"
CHRONOS_MODEL_ID = "amazon/chronos-2"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

# 21 quantiles: [0.01, 0.05, 0.1, ..., 0.5, ..., 0.9, 0.95, 0.99]
# p10 = idx 2, p50 = idx 10, p90 = idx 18
Q_IDX = {"p10": 2, "p50": 10, "p90": 18}


# Data

def load_data() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_exog.parquet")
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df["indice_general"]


# Model

def load_model():
    from chronos import Chronos2Pipeline

    logger.info(f"[chronos2] Loading {CHRONOS_MODEL_ID} ...")
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        device_map="cpu",
    )
    logger.info("[chronos2] Model loaded (21 quantiles, use_reg_token=True)")
    return pipeline


# Rolling backtesting

def run_rolling(y: pd.Series, model) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="Chronos C0 rolling"):
        context = torch.tensor(y.loc[:origin].values, dtype=torch.float32)

        preds = model.predict([context], prediction_length=MAX_H)
        # Shape: (n_variates, n_quantiles, pred_len) = (1, 21, 12) for univariate
        quantiles = preds[0].numpy()  # (1, 21, 12)
        q = quantiles[0]  # (21, 12) - remove variates dimension

        p50 = q[Q_IDX["p50"]]
        p10 = q[Q_IDX["p10"]]
        p90 = q[Q_IDX["p90"]]

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

            for i, (date, real) in enumerate(zip(fc_dates, y_true), start=1):
                records.append({
                    "origin": origin,
                    "fc_date": date,
                    "step": i,
                    "horizon": h,
                    "model": MODEL_NAME,
                    "y_true": float(real),
                    "y_pred": float(p50[i - 1]),
                    "y_pred_p10": float(p10[i - 1]),
                    "y_pred_p90": float(p90[i - 1]),
                    "error": float(real - p50[i - 1]),
                    "abs_error": float(abs(real - p50[i - 1])),
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
        # p10-p90 interval coverage (should be ~80%)
        p10 = h_df["y_pred_p10"].values
        p90 = h_df["y_pred_p90"].values
        coverage = float(np.mean((y_true >= p10) & (y_true <= p90)))

        results[f"h{h}"] = {
            "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
            "coverage_80": round(coverage, 4),
            "n_evals": int(len(h_df["origin"].unique())),
        }
    return results


def log_table(metrics: dict) -> None:
    logger.info(f"\n{'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'Cov80':>6} {'N':>5}")
    logger.info("-" * 52)
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics:
            m = metrics[key]
            logger.info(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                        f"{m['MASE']:8.4f} {m['coverage_80']:6.2%} {m['n_evals']:5d}")


# Main

def main():
    logger.info("=" * 60)
    logger.info(f"ROLLING BACKTESTING - {MODEL_NAME}")
    logger.info(f"Model: {CHRONOS_MODEL_ID}")
    logger.info(f"Origins: {ORIGINS_START} - {ORIGINS_END}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info("=" * 60)

    y = load_data()
    logger.info(f"Data: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")

    model = load_model()

    df_preds, mase_scale = run_rolling(y, model)
    logger.info(f"\nPredictions generated: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS {MODEL_NAME}")
    logger.info("=" * 60)
    log_table(metrics)

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
