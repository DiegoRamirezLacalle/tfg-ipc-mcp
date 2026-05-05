"""Rolling expanding-window backtesting for baseline models.

Design:
  - Expanding window: at each origin t, train on all data up to t
  - Fixed orders determined by auto_arima in 01/02/03 (no re-selection
    at each step - avoids look-ahead bias and reduces compute time)
  - Models: ARIMA(1,1,2), SARIMA(0,1,1)(0,1,1)12, SARIMAX with dfr, seasonal naive
  - Horizons: h = 1, 3, 6, 12 months
  - Origins: 2021-01 to 2024-12 (48 points; for h=12 last useful origin is 2023-12)

Note SARIMAX: DFR is public in real time (ECB decisions published same day),
so passing real DFR values as future exogenous does not introduce look-ahead bias.

Output:
  results/rolling_predictions.parquet  - tidy predictions (origin, horizon, model)
  results/rolling_metrics.json         - MAE/RMSE/MASE per model x horizon
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
from shared.metrics import mae, rmse, mase

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Fixed orders from auto_arima
ARIMA_ORDER   = (1, 1, 2)
SARIMA_ORDER  = (0, 1, 1)
SARIMA_SORDER = (0, 1, 1, 12)
EXOG_COL      = "dfr"

HORIZONS      = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
MODELS        = ["naive", "arima", "sarima", "sarimax"]
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_exog.parquet")
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df


def fit_arima(y_train: pd.Series):
    mod = SM_SARIMAX(y_train, order=ARIMA_ORDER, trend="c")
    return mod.fit(disp=False)


def fit_sarima(y_train: pd.Series):
    mod = SM_SARIMAX(y_train, order=SARIMA_ORDER,
                     seasonal_order=SARIMA_SORDER, trend="c")
    return mod.fit(disp=False)


def fit_sarimax(y_train: pd.Series, x_train: pd.DataFrame):
    mod = SM_SARIMAX(y_train, exog=x_train, order=SARIMA_ORDER,
                     seasonal_order=SARIMA_SORDER, trend="c")
    return mod.fit(disp=False)


def forecast_fixed(result, h: int, x_future=None) -> np.ndarray:
    """Forecast h steps from end of train."""
    fc = result.forecast(steps=h, exog=x_future)
    return fc.values


def forecast_naive(y_train: pd.Series, h: int) -> np.ndarray:
    """Seasonal naive: y[t+s] = y[t+s-12] for s=1..h."""
    preds = []
    for s in range(1, h + 1):
        # s=1 -> t-11, ..., s=12 -> t (monthly seasonality of 12)
        idx = -12 + ((s - 1) % 12)
        preds.append(float(y_train.iloc[idx]))
    return np.array(preds)


def run_rolling(df: pd.DataFrame) -> pd.DataFrame:
    y = df["indice_general"]
    X = df[[EXOG_COL]]

    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    # MASE scale: seasonal naive over initial train set (fixed)
    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="Origins"):
        y_train = y.loc[:origin]
        x_train = X.loc[:origin]

        try:
            res_arima   = fit_arima(y_train)
            res_sarima  = fit_sarima(y_train)
            res_sarimax = fit_sarimax(y_train, x_train)
        except Exception as e:
            logger.warning(f"\n[!] Error at {origin.date()}: {e}")
            continue

        for h in HORIZONS:
            # Full horizon must fit within test end.
            horizon_end = origin + pd.DateOffset(months=h)
            if horizon_end > TEST_END_TS:
                continue

            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )

            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue   # actuals not yet available

            y_true   = y_actual.values
            x_future = X.reindex(fc_dates)  # real DFR (no look-ahead bias)

            preds = {
                "naive":   forecast_naive(y_train, h),
                "arima":   forecast_fixed(res_arima, h),
                "sarima":  forecast_fixed(res_sarima, h),
                "sarimax": forecast_fixed(res_sarimax, h, x_future=x_future),
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
    """MAE, RMSE, MASE per model x horizon."""
    results = {}
    for model in MODELS:
        results[model] = {}
        m_df = df_preds[df_preds["model"] == model]
        for h in HORIZONS:
            h_df = m_df[m_df["horizon"] == h]
            if h_df.empty:
                continue
            y_true = h_df["y_true"].values
            y_pred = h_df["y_pred"].values
            results[model][f"h{h}"] = {
                "MAE":     round(float(np.mean(np.abs(y_true - y_pred))), 4),
                "RMSE":    round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
                "MASE":    round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
                "n_evals": int(len(h_df["origin"].unique())),
            }
    return results


def print_table(metrics: dict) -> None:
    for h in HORIZONS:
        key = f"h{h}"
        logger.info(f"\n--- Horizon h={h} ---")
        logger.info(f"{'Model':<10} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
        logger.info("-" * 42)
        for model in MODELS:
            if key in metrics.get(model, {}):
                m = metrics[model][key]
                logger.info(f"{model:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                             f"{m['MASE']:8.4f} {m['n_evals']:5d}")


def main():
    logger.info("=" * 60)
    logger.info("ROLLING BACKTESTING - baseline models")
    logger.info(f"Origins: {ORIGINS_START} - {ORIGINS_END}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Models: {MODELS}")
    logger.info("=" * 60)

    df = load_data()
    logger.info(f"Data loaded: {df.index.min().date()} - {df.index.max().date()} ({len(df)} obs)")

    df_preds, mase_scale = run_rolling(df)

    logger.info(f"\nTotal predictions generated: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info("ROLLING BACKTESTING RESULTS")
    logger.info("=" * 60)
    print_table(metrics)

    # MASE relative to seasonal naive (relative benchmark)
    logger.info("\n--- MASE relative to seasonal naive (naive=1.00) ---")
    header = f"{'Model':<10}"
    for h in HORIZONS:
        header += f"  h={h:>2}"
    logger.info(header)
    logger.info("-" * 42)
    for model in ["arima", "sarima", "sarimax"]:
        row = f"{model:<10}"
        for h in HORIZONS:
            key = f"h{h}"
            if key in metrics.get(model, {}) and key in metrics.get("naive", {}):
                ratio = metrics[model][key]["MASE"] / metrics["naive"][key]["MASE"]
                row += f"  {ratio:>5.3f}"
            else:
                row += f"  {'N/A':>5}"
        logger.info(row)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = RESULTS_DIR / "rolling_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    logger.info(f"\nPredictions saved: {preds_path}")

    metrics_path = RESULTS_DIR / "rolling_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved:     {metrics_path}")


if __name__ == "__main__":
    main()
