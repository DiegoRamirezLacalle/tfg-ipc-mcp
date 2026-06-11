"""
30_chronos2_C0_global.py - Chronos-2 condition C0, GLOBAL CPI (historical only)

Global counterpart of 03_chronos2_C0.py (which is Spain CPI).

Target : data/processed/cpi_global_monthly.parquet :: cpi_global_rate
Model ID / outputs : chronos2_C0_global

Rolling-origin backtesting (same protocol as all C0 scripts):
  - origins: 2021-01 to DATE_TEST_END
  - horizons: h = 1, 3, 6, 12
  - MAE, RMSE, MASE (seasonal naive lag-12 over the GLOBAL train series)

A target-integrity guard verifies that every prediction row's y_true equals
cpi_global_rate at the same fc_date BEFORE anything is written. This makes it
impossible to silently emit Spain values under a Global name.
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
MODEL_NAME = "chronos2_C0_global"
CHRONOS_MODEL_ID = "amazon/chronos-2"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

TARGET_FILE = ROOT / "data" / "processed" / "cpi_global_monthly.parquet"
TARGET_COL = "cpi_global_rate"

Q_IDX = {"p10": 2, "p50": 10, "p90": 18}


def load_data() -> pd.Series:
    df = pd.read_parquet(TARGET_FILE)
    df.index = pd.to_datetime(df.index)
    df.index.freq = "MS"
    return df[TARGET_COL]


def load_model():
    from chronos import Chronos2Pipeline
    logger.info(f"[chronos2] Loading {CHRONOS_MODEL_ID} ...")
    pipeline = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL_ID, device_map="cpu")
    logger.info("[chronos2] Model loaded")
    return pipeline


def run_rolling(y: pd.Series, model) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []
    for origin in tqdm(origins, desc="Chronos C0 GLOBAL rolling"):
        context = torch.tensor(y.loc[:origin].values, dtype=torch.float32)
        preds = model.predict([context], prediction_length=MAX_H)
        q = preds[0].numpy()[0]  # (21, 12)
        p50, p10, p90 = q[Q_IDX["p50"]], q[Q_IDX["p10"]], q[Q_IDX["p90"]]

        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS:
                continue
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue
            for i, (date, real) in enumerate(zip(fc_dates, y_actual.values), start=1):
                records.append({
                    "origin": origin, "fc_date": date, "step": i, "horizon": h,
                    "model": MODEL_NAME,
                    "y_true": float(real), "y_pred": float(p50[i - 1]),
                    "y_pred_p10": float(p10[i - 1]), "y_pred_p90": float(p90[i - 1]),
                    "error": float(real - p50[i - 1]),
                    "abs_error": float(abs(real - p50[i - 1])),
                })

    return pd.DataFrame(records), mase_scale


def assert_target_integrity(df_preds: pd.DataFrame, y: pd.Series) -> None:
    """
    Guard: every prediction row's y_true must equal cpi_global_rate at its
    fc_date. Raises and refuses to write if Spain (or any other) values slipped in.
    """
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
            f"[{MODEL_NAME}] y_true range [{lo:.2f},{hi:.2f}] looks like the Spain "
            f"index scale (80-100), not a CPI rate. Refusing to write."
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


def main():
    logger.info("=" * 60)
    logger.info(f"ROLLING BACKTESTING - {MODEL_NAME}")
    logger.info(f"Target: {TARGET_FILE.name} :: {TARGET_COL}")
    logger.info(f"Origins: {ORIGINS_START} - {ORIGINS_END} | Horizons: {HORIZONS}")
    logger.info("=" * 60)

    y = load_data()
    logger.info(f"Data: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")

    model = load_model()
    df_preds, mase_scale = run_rolling(y, model)
    logger.info(f"\nPredictions generated: {len(df_preds)}")

    # GUARD before writing anything
    assert_target_integrity(df_preds, y)

    metrics = compute_metrics(df_preds, mase_scale)
    logger.info(f"\nRESULTS {MODEL_NAME}")
    log_table(metrics)

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
