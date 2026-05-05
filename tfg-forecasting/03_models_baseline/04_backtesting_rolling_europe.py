"""Rolling expanding-window backtesting - HICP Eurozone.

Identical design to Spain and Global pipelines:
  - Expanding window: train on all data up to origin t
  - Fixed orders from auto_arima (script 01)
  - Horizons: h = 1, 3, 6, 12 months
  - Origins: 2021-01 to 2024-12 (48 points)

Models:
  naive   - seasonal lag-12
  sarima  - SARIMA with auto_arima order
  sarimax - SARIMA + DFR (real values known, no leakage)

MASE scale: fixed over initial train set (2002-01 to 2020-12).

Output:
  08_results/rolling_predictions_europe.parquet
  08_results/rolling_metrics_europe.json
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

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR   = ROOT / "08_results"
HORIZONS      = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)
MODELS        = ["naive", "sarima", "sarimax"]


def load_orders():
    path = RESULTS_DIR / "arima_europe_metrics.json"
    if path.exists():
        saved = json.loads(path.read_text())
        order          = tuple(saved["order"])
        seasonal_order = tuple(saved["seasonal_order"])
        logger.info(f"  auto_arima order: SARIMA{order}x{seasonal_order}")
    else:
        order          = (2, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        logger.info(f"  Fallback order:   SARIMA{order}x{seasonal_order}")
    return order, seasonal_order


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    y = df["hicp_index"]

    ecb = pd.read_parquet(ROOT / "data" / "processed" / "ecb_rates_monthly.parquet")
    dfr = ecb["dfr"].reindex(y.index).ffill()

    return y, dfr


def forecast_naive(y_train: pd.Series, h: int) -> np.ndarray:
    preds = []
    for s in range(1, h + 1):
        idx = -12 + ((s - 1) % 12)
        preds.append(float(y_train.iloc[idx]))
    return np.array(preds)


def run_rolling(y: pd.Series, dfr: pd.Series, order: tuple, seasonal_order: tuple):
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))
    logger.info(f"  MASE scale: {mase_scale:.4f}")

    records = []
    for origin in tqdm(origins, desc="Origins"):
        y_train   = y.loc[:origin]
        dfr_train = dfr.loc[:origin].values.reshape(-1, 1)

        try:
            res_sarima = SM_SARIMAX(
                y_train, order=order, seasonal_order=seasonal_order, trend="n"
            ).fit(disp=False)

            res_sarimax = SM_SARIMAX(
                y_train, exog=dfr_train,
                order=order, seasonal_order=seasonal_order, trend="n"
            ).fit(disp=False)
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

            y_true    = y_actual.values
            dfr_fut   = dfr.reindex(fc_dates).values.reshape(-1, 1)

            preds = {
                "naive":   forecast_naive(y_train, h),
                "sarima":  res_sarima.get_forecast(steps=h).predicted_mean.values,
                "sarimax": res_sarimax.get_forecast(
                    steps=h, exog=dfr_fut
                ).predicted_mean.values,
            }

            for model_name, y_pred in preds.items():
                for i, (date, real, pred) in enumerate(
                    zip(fc_dates, y_true, y_pred), start=1
                ):
                    records.append({
                        "origin":    origin,
                        "fc_date":   date,
                        "step":      i,
                        "horizon":   h,
                        "model":     model_name,
                        "y_true":    real,
                        "y_pred":    pred,
                        "error":     real - pred,
                        "abs_error": abs(real - pred),
                    })

    return pd.DataFrame(records), mase_scale


def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for model in MODELS:
        results[model] = {}
        m_df = df_preds[df_preds["model"] == model]
        for h in HORIZONS:
            h_df = m_df[m_df["horizon"] == h]
            if h_df.empty:
                continue
            yt = h_df["y_true"].values
            yp = h_df["y_pred"].values
            results[model][f"h{h}"] = {
                "MAE":     round(float(np.mean(np.abs(yt - yp))), 4),
                "RMSE":    round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
                "MASE":    round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
                "n_evals": int(len(h_df["origin"].unique())),
            }
    return results


def print_results(metrics: dict) -> None:
    header = f"\n  {'Model':<10}"
    for h in HORIZONS:
        header += f"   h={h:>2} MAE"
    logger.info(header)
    logger.info(f"  {'-'*55}")
    for model in MODELS:
        row = f"  {model:<10}"
        for h in HORIZONS:
            key = f"h{h}"
            if key in metrics.get(model, {}):
                v = metrics[model][key]["MAE"]
                ref = metrics["naive"][key]["MAE"]
                mark = "*" if v < ref else " "
                row += f"  {v:>7.4f}{mark}"
            else:
                row += f"  {'N/A':>8}"
        logger.info(row)
    logger.info("  (* = beats lag-12 naive)")

    logger.info(f"\n  DFR BENEFIT (SARIMAX vs SARIMA):")
    logger.info(f"  {'h':>4}  {'MAE SARIMA':>12}  {'MAE SARIMAX':>13}  {'Delta%':>8}")
    logger.info(f"  {'-'*42}")
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics.get("sarima", {}) and key in metrics.get("sarimax", {}):
            ms = metrics["sarima"][key]["MAE"]
            mx = metrics["sarimax"][key]["MAE"]
            pct = (mx - ms) / ms * 100
            mark = " <-- improvement" if pct < 0 else ""
            logger.info(f"  {h:>4}  {ms:>12.4f}  {mx:>13.4f}  {pct:>+7.1f}%{mark}")


def main():
    logger.info("=" * 60)
    logger.info("ROLLING BACKTESTING - HICP Eurozone")
    logger.info(f"  Origins:  {ORIGINS_START} - {ORIGINS_END}")
    logger.info(f"  Horizons: {HORIZONS}")
    logger.info("=" * 60)

    order, seasonal_order = load_orders()
    y, dfr = load_data()
    logger.info(f"\nHICP: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")
    logger.info(f"DFR:  {dfr.index.min().date()} - {dfr.index.max().date()}\n")

    df_preds, mase_scale = run_rolling(y, dfr, order, seasonal_order)
    logger.info(f"\nRecords: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    print_results(metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR / "rolling_predictions_europe.parquet", index=False)
    with open(RESULTS_DIR / "rolling_metrics_europe.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
