"""
02_timesfm_C1.py — TimesFM 2.5 condition C1 (historical + MCP signals)

Fix 1 — XReg restricted to the period with real signals (2015+):
  The base TimesFM model receives the COMPLETE IPC context (282 obs,
  identical to C0). The MCP signal correction is computed via an
  external Ridge fitted ONLY on df.loc['2015':origin], where all
  covariates have real values. The correction is the difference
  between the Ridge prediction with current signals and with neutral
  signals (zeros). This correctly implements the base-TimesFM / XReg-MCP
  separation.

Fix 2 — Covariate selection:
  Ridge input columns: gdelt_avg_tone, gdelt_tone_ma3,
  gdelt_tone_ma6, bce_shock_score, bce_tone_numeric, bce_cumstance,
  ine_surprise_score, ine_inflacion, signal_available.
  Removed: bce_uncertainty (5 values), gdelt_goldstein/n_articles
  (correlated/noisy), dfr_diff/lag3/6/12 (redundant with rates
  already in IPC context).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
MODEL_NAME = "timesfm_C1"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

# Fix 1: XReg fitted ONLY from this date onward
SIGNAL_START = "2015-01-01"

# Fix 2: XReg external covariates (exactly as specified)
XREG_COVS = [
    "gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6",
    "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
    "ine_surprise_score", "ine_inflacion",
    "signal_available",
]

# Ridge regularization (avoids overfitting with ~60-120 obs and 9 features)
RIDGE_ALPHA = 1.0


# Data

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    for col in XREG_COVS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


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
    logger.info("[timesfm] Base model loaded (no internal XReg, identical to C0)")
    return tfm


# Fix 1: external Ridge XReg fitted over 2015:origin

def compute_xreg_correction(
    df: pd.DataFrame,
    origin: pd.Timestamp,
) -> float:
    """
    Fit Ridge on df.loc[SIGNAL_START:origin] using only real signals.
    Target: monthly IPC change (first difference, stationary ±1.5 pp).
    Returns the marginal correction:
      correction = beta @ current_signals
                 = Ridge.predict(current) - Ridge.predict(zeros)

    Using first differences prevents the Ridge from overfitting the IPC
    level trend (range 79-100) to the signals, which would produce
    huge corrections (±5-10 pp).
    Returns 0.0 if insufficient data (< 13 months with signals).
    """
    signal_start_ts = pd.Timestamp(SIGNAL_START)
    window = df.loc[signal_start_ts:origin].copy()

    window = window[window["signal_available"] > 0]
    if len(window) < 13:
        return 0.0

    ipc_mom = window["indice_general"].diff(1)
    valid = ~ipc_mom.isna()
    X = window.loc[valid, XREG_COVS].values.astype(np.float64)
    y_diff = ipc_mom[valid].values.astype(np.float64)

    reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    reg.fit(X, y_diff)

    current = df.loc[origin:origin, XREG_COVS].values.astype(np.float64)
    neutral = np.zeros_like(current)

    correction = float(reg.predict(current)[0] - reg.predict(neutral)[0])
    return correction


# Rolling backtesting

def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["indice_general"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="TimesFM C1 rolling"):
        context = y.loc[SIGNAL_START:origin].values.astype(np.float32)
        try:
            point_out, _ = model.forecast(horizon=MAX_H, inputs=[context])
            base_pred = np.array(point_out[0])  # shape (MAX_H,)
        except Exception as e:
            logger.warning(f"\n[!] Base error at {origin.date()}: {e}")
            continue

        xreg_correction = compute_xreg_correction(df, origin)

        full_pred = base_pred + xreg_correction

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
    logger.info("=" * 60)
    logger.info(f"ROLLING BACKTESTING — {MODEL_NAME}")
    logger.info(f"Fix 1: base TimesFM C0 (282 obs) + external Ridge over 2015:origin")
    logger.info(f"Fix 2: XReg covariates = {XREG_COVS}")
    logger.info(f"Origins: {ORIGINS_START} - {ORIGINS_END}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info("=" * 60)

    df = load_data()
    logger.info(f"Data: {df.index.min().date()} - {df.index.max().date()} ({len(df)} obs)")

    model = load_model()

    df_preds, mase_scale = run_rolling(df, model)
    logger.info(f"\nPredictions generated: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS {MODEL_NAME}")
    logger.info("=" * 60)
    log_table(metrics)

    c0_path = RESULTS_DIR / "timesfm_C0_metrics.json"
    if c0_path.exists():
        with open(c0_path) as f:
            c0_metrics = json.load(f).get("timesfm_C0", {})
        logger.info("\n--- C0 vs C1 (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            c0_mae = c0_metrics.get(key, {}).get("MAE", "N/A")
            c1_mae = metrics.get(key, {}).get("MAE", "N/A")
            if isinstance(c0_mae, float) and isinstance(c1_mae, float):
                delta = c1_mae - c0_mae
                pct = (delta / c0_mae) * 100
                logger.info(f"  h={h}: C0={c0_mae:.4f}  C1={c1_mae:.4f}  "
                            f"delta={delta:+.4f} ({pct:+.1f}%)")

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
