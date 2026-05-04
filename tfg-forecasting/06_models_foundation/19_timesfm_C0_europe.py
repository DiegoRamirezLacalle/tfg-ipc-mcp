"""
19_timesfm_C0_europe.py — TimesFM 2.5 condition C0 (historical only) HICP Eurozone

Rolling-origin backtesting:
  - 48 origins: 2021-01 to 2024-12
  - Horizons: h=1, 3, 6, 12
  - Metrics: MAE, RMSE, MASE (seasonal naive lag-12)

Model: google/timesfm-2.5-200m-pytorch
Series: hicp_europe_index.parquet (level index, base 2015=100)

Output:
  08_results/timesfm_C0_europe_predictions.parquet
  08_results/timesfm_C0_europe_metrics.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
MODEL_NAME = "timesfm_C0_europe"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)


# Data

def load_data() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    return df["hicp_index"]


# Model

def load_model():
    import timesfm

    logger.info("[timesfm] Loading model google/timesfm-2.5-200m-pytorch ...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
    )
    tfm.compile(
        forecast_config=timesfm.ForecastConfig(
            max_context=512,
            max_horizon=MAX_H,
            per_core_batch_size=1,
        )
    )
    logger.info("[timesfm] Model loaded and compiled (max_horizon=12)")
    return tfm


# Rolling backtesting

def run_rolling(y: pd.Series, model) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))
    logger.info(f"  MASE scale: {mase_scale:.4f}")

    records = []

    for origin in tqdm(origins, desc="TimesFM C0 europe rolling"):
        context = y.loc[:origin].values.astype(np.float32)

        point_out, _ = model.forecast(horizon=MAX_H, inputs=[context])
        full_pred = point_out[0]  # shape: (MAX_H,)

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
            y_pred = full_pred[:h]

            for i, (date, real, pred) in enumerate(
                zip(fc_dates, y_true, y_pred), start=1
            ):
                records.append({
                    "origin":    origin,
                    "fc_date":   date,
                    "step":      i,
                    "horizon":   h,
                    "model":     MODEL_NAME,
                    "y_true":    float(real),
                    "y_pred":    float(pred),
                    "error":     float(real - pred),
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
            "MAE":     round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE":    round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MASE":    round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
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
    logger.info("=" * 60)
    logger.info(f"ROLLING BACKTESTING — {MODEL_NAME}")
    logger.info(f"Origins: {ORIGINS_START} - {ORIGINS_END}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info("=" * 60)

    y = load_data()
    logger.info(f"HICP Europe: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")

    model = load_model()

    df_preds, mase_scale = run_rolling(y, model)
    logger.info(f"\nPredictions generated: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS {MODEL_NAME}")
    logger.info("=" * 60)
    log_table(metrics)

    # Comparison vs SARIMA baseline and Chronos2
    baseline_path = RESULTS_DIR / "rolling_metrics_europe.json"
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text())
        logger.info("\n--- vs SARIMA (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            tfm = metrics.get(key, {}).get("MAE")
            sar = baselines.get("sarima", {}).get(key, {}).get("MAE")
            if tfm and sar:
                delta = tfm - sar
                logger.info(f"  h={h}: TimesFM={tfm:.4f}  SARIMA={sar:.4f}  "
                            f"delta={delta:+.4f} ({delta/sar*100:+.1f}%)")

    c2_path = RESULTS_DIR / "chronos2_C0_europe_metrics.json"
    if c2_path.exists():
        c2 = json.loads(c2_path.read_text()).get("chronos2_C0_europe", {})
        logger.info("\n--- vs Chronos2 C0 europe (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            tfm = metrics.get(key, {}).get("MAE")
            c2m = c2.get(key, {}).get("MAE")
            if tfm and c2m:
                delta = tfm - c2m
                logger.info(f"  h={h}: TimesFM={tfm:.4f}  Chronos2={c2m:.4f}  "
                            f"delta={delta:+.4f} ({delta/c2m*100:+.1f}%)")

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
