"""ARIMAX C1_institutional - Global CPI with institutional signals.

ARIMA(3,1,0) with global institutional signals selected by
correlation >= 0.2 with cpi_global_rate(t+1).

Top signals (full-sample correlation):
  imf_comm_ma3  (0.586) - IMF All Commodity Index
  brent_log_ma3 (0.456) - Brent crude
  dfr_ma3       (0.376) - ECB DFR
  gscpi_ma3     (0.324) - NY Fed Supply Chain
  fedfunds_ma3  (0.279) - Fed Funds
  usg10y_ma3    (0.250) - 10Y UST

Rolling expanding-window protocol identical to baseline script 04.

Output:
  08_results/arimax_C1_inst_global_metrics.json
  08_results/rolling_predictions_C1_inst_global.parquet
  08_results/rolling_metrics_C1_inst_global.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"

ARIMA_ORDER   = (3, 1, 0)
HORIZONS      = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)

# Selected signals (available from 2002, no NaN after bfill)
EXOG_COLS = [
    "imf_comm_ma3",
    "brent_log_ma3",
    "dfr_ma3",
    "gscpi_ma3",
    "fedfunds_ma3",
    "usg10y_ma3",
]


def load_data():
    feat = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_global_institutional.parquet")
    feat.index = pd.to_datetime(feat.index)
    feat.index.freq = "MS"

    y = feat["cpi_global_rate"]
    X = feat[EXOG_COLS].copy()

    # Fill residual NaNs
    X = X.ffill().bfill()

    return y, X


def fit_arimax(y_train, x_train):
    mod = SM_SARIMAX(y_train, exog=x_train, order=ARIMA_ORDER, trend="n")
    return mod.fit(disp=False)


def run_static_val(y, X):
    """Static evaluation: train 2002-2020 / val 2021-01 to 2022-06."""
    train_mask = y.index <= DATE_TRAIN_END
    val_mask   = (y.index > DATE_TRAIN_END) & (y.index <= DATE_VAL_END)

    y_train, y_val = y[train_mask], y[val_mask]
    X_train, X_val = X[train_mask], X[val_mask]

    res = fit_arimax(y_train, X_train)

    fc = res.get_forecast(steps=len(y_val), exog=X_val)
    y_pred = fc.predicted_mean.values
    y_true = y_val.values

    mase_scale = float(np.mean(np.abs(y_train.values[12:] - y_train.values[:-12])))

    metrics = {
        "model":    "arimax_C1_inst_global",
        "order":    list(ARIMA_ORDER),
        "exog":     EXOG_COLS,
        "n_train":  int(train_mask.sum()),
        "metrics_val": {
            "MAE":  round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
        }
    }
    return metrics


def run_rolling(y, X):
    origins    = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(y_train_init.values[12:] - y_train_init.values[:-12])))
    logger.info(f"  MASE scale: {mase_scale:.4f} pp")

    records = []
    for origin in tqdm(origins, desc="ARIMAX C1_inst global"):
        y_train = y.loc[:origin]
        x_train = X.loc[:origin]

        try:
            res = fit_arimax(y_train, x_train)
        except Exception as e:
            logger.warning(f"\n[!] {origin.date()}: {e}")
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

            x_future = X.reindex(fc_dates).ffill().bfill()
            fc = res.get_forecast(steps=h, exog=x_future)
            y_pred = fc.predicted_mean.values
            y_true = y_actual.values

            for i, (date, real, pred) in enumerate(zip(fc_dates, y_true, y_pred), 1):
                records.append({
                    "origin":    origin,
                    "fc_date":   date,
                    "step":      i,
                    "horizon":   h,
                    "model":     "arimax_C1_inst",
                    "y_true":    real,
                    "y_pred":    pred,
                    "error":     real - pred,
                    "abs_error": abs(real - pred),
                })

    return pd.DataFrame(records), mase_scale


def compute_metrics(df_preds, mase_scale):
    results = {"arimax_C1_inst": {}}
    m_df = df_preds[df_preds["model"] == "arimax_C1_inst"]
    for h in HORIZONS:
        h_df = m_df[m_df["horizon"] == h]
        if h_df.empty:
            continue
        yt = h_df["y_true"].values
        yp = h_df["y_pred"].values
        results["arimax_C1_inst"][f"h{h}"] = {
            "MAE":     round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE":    round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE":    round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(h_df["origin"].unique())),
        }
    return results


def main():
    logger.info("=" * 60)
    logger.info("ARIMAX C1_institutional - CPI Global")
    logger.info(f"  Order: {ARIMA_ORDER}  Exogenous: {len(EXOG_COLS)}")
    logger.info(f"  {EXOG_COLS}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    y, X = load_data()
    logger.info(f"\nCPI Global: {len(y)} obs  ({y.index.min().date()} - {y.index.max().date()})")
    logger.info(f"Exogenous:   {X.shape[1]} cols, NaN={X.isna().sum().sum()}\n")

    logger.info("1. Static evaluation...")
    static = run_static_val(y, X)
    mv = static["metrics_val"]
    logger.info(f"   MAE={mv['MAE']}  RMSE={mv['RMSE']}  MASE={mv['MASE']}")

    with open(RESULTS_DIR / "arimax_C1_inst_global_metrics.json", "w") as f:
        json.dump(static, f, indent=2)

    logger.info("\n2. Rolling expanding-window...")
    df_preds, mase_scale = run_rolling(y, X)
    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info("ROLLING RESULTS C1_institutional")
    logger.info("=" * 60)
    logger.info(f"\n  {'h':>4} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    logger.info(f"  {'-'*38}")
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics["arimax_C1_inst"]:
            m = metrics["arimax_C1_inst"][key]
            logger.info(f"  {h:>4} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MASE']:>8.4f} {m['n_evals']:>5d}")

    try:
        with open(RESULTS_DIR / "rolling_metrics_global.json") as f:
            c0 = json.load(f)
        logger.info("\n  Delta MAE vs ARIMAX C0 (positive = C1 improvement):")
        logger.info(f"  {'h':>4} {'C0 ARIMAX':>12} {'C1 inst':>12} {'Delta%':>10}")
        logger.info(f"  {'-'*42}")
        for h in HORIZONS:
            key = f"h{h}"
            if key in metrics["arimax_C1_inst"] and key in c0.get("arimax", {}):
                c0_mae = c0["arimax"][key]["MAE"]
                c1_mae = metrics["arimax_C1_inst"][key]["MAE"]
                pct    = (c0_mae - c1_mae) / c0_mae * 100
                mark   = " <-- improvement" if pct > 0 else ""
                logger.info(f"  {h:>4} {c0_mae:>12.4f} {c1_mae:>12.4f} {pct:>+9.1f}%{mark}")
    except FileNotFoundError:
        pass

    preds_path = RESULTS_DIR / "rolling_predictions_C1_inst_global.parquet"
    df_preds.to_parquet(preds_path, index=False)
    logger.info(f"\nPredictions: {preds_path}")

    metrics_path = RESULTS_DIR / "rolling_metrics_C1_inst_global.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics:     {metrics_path}")

    logger.info("\n" + "=" * 60)
    logger.info("ARIMAX C1_inst COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
