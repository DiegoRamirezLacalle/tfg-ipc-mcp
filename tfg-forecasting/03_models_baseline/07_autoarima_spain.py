"""AutoSARIMA rolling backtesting — Spain CPI.

Key difference vs fixed SARIMA (04_backtesting_rolling.py):
  - At each rolling origin, pmdarima.auto_arima re-selects the optimal
    (p,d,q)(P,D,Q) orders via AIC + stepwise.
  - Fixed SARIMA uses orders determined once on the initial training set.

Design:
  - Expanding window: at each origin t, train on all data up to t
  - Seasonal auto_arima (m=12) re-fitted at each origin
  - Horizons: h = 1, 3, 6, 12 months
  - Origins: 2021-01 to 2024-12 (48 points)
  - No exogenous (pure AutoSARIMA, comparable to SARIMA baseline)

Output:
  08_results/autoarima_spain_predictions.parquet
  08_results/autoarima_spain_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pmdarima import auto_arima
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


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_exog.parquet")
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df["indice_general"]


def forecast_naive(y_train: pd.Series, h: int) -> np.ndarray:
    preds = []
    for s in range(1, h + 1):
        idx = -12 + ((s - 1) % 12)
        preds.append(float(y_train.iloc[idx]))
    return np.array(preds)


def run_rolling(y: pd.Series):
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))
    logger.info(f"  MASE scale: {mase_scale:.4f} pp")

    records = []
    orders_log = []

    for origin in tqdm(origins, desc="AutoARIMA Spain"):
        y_train = y.loc[:origin]

        try:
            model = auto_arima(
                y_train,
                seasonal=True,
                m=12,
                stepwise=True,
                information_criterion="aic",
                max_p=3, max_q=3,
                max_P=2, max_Q=2,
                max_d=2, max_D=1,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
            )
            orders_log.append({
                "origin": str(origin.date()),
                "order": list(model.order),
                "seasonal_order": list(model.seasonal_order),
            })
        except Exception as e:
            logger.warning(f"\n[!] auto_arima error at {origin.date()}: {e}")
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

            try:
                y_pred_auto = model.predict(n_periods=h)
            except Exception:
                continue

            y_pred_naive = forecast_naive(y_train, h)

            for model_name, y_pred in [("auto_arima", y_pred_auto), ("naive", y_pred_naive)]:
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

    return pd.DataFrame(records), mase_scale, orders_log


def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for model in ["auto_arima", "naive"]:
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


def main():
    logger.info("=" * 60)
    logger.info("AutoARIMA ROLLING — Spain CPI")
    logger.info(f"  Origins:  {ORIGINS_START} - {ORIGINS_END}")
    logger.info(f"  Horizons: {HORIZONS}")
    logger.info(f"  Method:   pmdarima.auto_arima re-fitted at each origin")
    logger.info("=" * 60)

    y = load_data()
    logger.info(f"\nSpain CPI: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)\n")

    df_preds, mase_scale, orders_log = run_rolling(y)
    logger.info(f"\nTotal records: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info("AutoARIMA Spain RESULTS")
    logger.info("=" * 60)
    ref = metrics.get("naive", {})
    logger.info(f"\n  {'h':>4}  {'MAE':>8}  {'RMSE':>8}  {'MASE':>8}  {'vs naive':>9}")
    logger.info(f"  {'-'*48}")
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics.get("auto_arima", {}):
            m = metrics["auto_arima"][key]
            n_mae = ref.get(key, {}).get("MAE", float("nan"))
            ratio = m["MAE"] / n_mae if n_mae else float("nan")
            mark = " *" if ratio < 1.0 else ""
            logger.info(f"  {h:>4}  {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  "
                        f"{m['MASE']:>8.4f}  {ratio:>8.3f}x{mark}")

    logger.info("\n  (* = beats seasonal lag-12 naive)")

    logger.info("\n  Sample of auto_arima orders (every 12 origins):")
    for entry in orders_log[::12]:
        logger.info(f"    {entry['origin']}: SARIMA{tuple(entry['order'])}x{tuple(entry['seasonal_order'])}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR / "autoarima_spain_predictions.parquet", index=False)
    with open(RESULTS_DIR / "autoarima_spain_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_DIR / "autoarima_spain_orders.json", "w") as f:
        json.dump(orders_log, f, indent=2)

    logger.info(f"\nSaved to {RESULTS_DIR}")
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
