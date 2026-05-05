"""Rolling expanding-window backtesting for Global CPI baseline models.

Identical design to the Spain pipeline (04_backtesting_rolling.py):
  - Expanding window: at each origin t, train on all data up to t
  - Fixed orders determined by auto_arima in scripts 01-03 (no re-selection)
  - Horizons: h = 1, 3, 6, 12 months
  - Origins: 2021-01 to 2024-12 (48 points)

Models:
  naive    - Seasonal lag-12 naive (benchmark)
  arima    - ARIMA(3,1,0)  [AIC winner from auto_arima, script 01]
  arima111 - ARIMA(1,1,1)  [simple reference, script 02]
  arimax   - ARIMA(3,1,0) + FEDFUNDS [script 03, evaluated with real exogenous]

Note ARIMAX:
  Fed decisions are published on the same day as the FOMC meeting.
  Using real FEDFUNDS values as future exogenous does not introduce
  look-ahead bias (same assumption as DFR/ECB in Spain).

MASE scale:
  Computed over the initial train set (2002-2020) as the mean absolute
  error of the lag-12 naive: mean(|y[t] - y[t-12]|). Fixed across origins.

Output:
  08_results/rolling_predictions_global.parquet
  08_results/rolling_metrics_global.json
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

RESULTS_DIR = ROOT / "08_results"

# Fixed orders from auto_arima (scripts 01-02)
ARIMA_ORDER    = (3, 1, 0)
ARIMA111_ORDER = (1, 1, 1)
EXOG_COL       = "fedfunds"

HORIZONS      = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
MODELS        = ["naive", "arima", "arima111", "arimax"]
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)


def load_data():
    # Target series
    df_y = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df_y["cpi_global_rate"]
    y.index = pd.DatetimeIndex(y.index, freq="MS")

    # FEDFUNDS exogenous
    ff = pd.read_parquet(ROOT / "data" / "processed" / "fedfunds_monthly.parquet")[EXOG_COL]
    ff.index = pd.DatetimeIndex(ff.index, freq="MS")
    ff = ff.reindex(y.index).ffill()

    return y, ff


def fit_arima(y_train: pd.Series, order: tuple):
    mod = SM_SARIMAX(y_train, order=order, trend="n")
    return mod.fit(disp=False)


def fit_arimax(y_train: pd.Series, x_train: pd.DataFrame, order: tuple):
    mod = SM_SARIMAX(y_train, exog=x_train, order=order, trend="n")
    return mod.fit(disp=False)


def forecast_fixed(result, h: int, x_future=None) -> np.ndarray:
    fc = result.get_forecast(steps=h, exog=x_future)
    return fc.predicted_mean.values


def forecast_naive(y_train: pd.Series, h: int) -> np.ndarray:
    """Seasonal lag-12 naive: y[t+s] = y[t+s-12] for s=1..h."""
    preds = []
    for s in range(1, h + 1):
        idx = -12 + ((s - 1) % 12)
        preds.append(float(y_train.iloc[idx]))
    return np.array(preds)


def run_rolling(y: pd.Series, ff: pd.Series):
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    # MASE scale: fixed over initial train set
    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))
    logger.info(f"  MASE scale (naive lag-12, train): {mase_scale:.4f} pp")

    records = []

    for origin in tqdm(origins, desc="Origins"):
        y_train = y.loc[:origin]
        x_train = ff.loc[:origin].to_frame()

        # Fit parametric models
        try:
            res_arima    = fit_arima(y_train,  ARIMA_ORDER)
            res_arima111 = fit_arima(y_train,  ARIMA111_ORDER)
            res_arimax   = fit_arimax(y_train, x_train, ARIMA_ORDER)
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

            y_true   = y_actual.values
            x_future = ff.reindex(fc_dates).to_frame()

            preds = {
                "naive":    forecast_naive(y_train, h),
                "arima":    forecast_fixed(res_arima,    h),
                "arima111": forecast_fixed(res_arima111, h),
                "arimax":   forecast_fixed(res_arimax,   h, x_future=x_future),
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


def print_table(metrics: dict) -> None:
    for h in HORIZONS:
        key = f"h{h}"
        n   = metrics.get("arima", {}).get(key, {}).get("n_evals", "?")
        logger.info(f"\n  h={h} ({n} evaluations)")
        logger.info(f"  {'Model':<10} {'MAE':>8} {'RMSE':>8} {'MASE':>8}  vs naive")
        logger.info(f"  {'-'*48}")
        for model in MODELS:
            if key not in metrics.get(model, {}):
                continue
            m     = metrics[model][key]
            ratio = m["MAE"] / metrics["naive"][key]["MAE"]
            mark  = " *" if ratio < 1.0 else ""
            logger.info(f"  {model:<10} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                        f"{m['MASE']:>8.4f}  {ratio:.3f}x{mark}")


def print_mase_table(metrics: dict) -> None:
    """MASE relative to naive (naive=1.00)."""
    header = f"  {'Model':<10}"
    for h in HORIZONS:
        header += f"   h={h:>2}"
    logger.info(header)
    logger.info(f"  {'-'*42}")
    for model in ["arima", "arima111", "arimax"]:
        row = f"  {model:<10}"
        for h in HORIZONS:
            key = f"h{h}"
            if key in metrics.get(model, {}) and key in metrics.get("naive", {}):
                ratio = metrics[model][key]["MASE"] / metrics["naive"][key]["MASE"]
                mark = "*" if ratio < 1.0 else " "
                row += f"  {ratio:>5.3f}{mark}"
            else:
                row += f"  {'N/A':>6}"
        logger.info(row)
    logger.info("  (* = beats seasonal lag-12 naive)")


def main():
    logger.info("=" * 60)
    logger.info("ROLLING BACKTESTING - Baseline CPI Global")
    logger.info(f"  Origins:       {ORIGINS_START} - {ORIGINS_END}")
    logger.info(f"  Horizons:      {HORIZONS}")
    logger.info(f"  Models:        {MODELS}")
    logger.info(f"  ARIMA order:   {ARIMA_ORDER}")
    logger.info(f"  ARIMA111 order:{ARIMA111_ORDER}")
    logger.info(f"  ARIMAX exog:   {EXOG_COL.upper()}")
    logger.info("=" * 60)

    y, ff = load_data()
    logger.info(f"\nCPI Global: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")
    logger.info(f"FEDFUNDS:   {ff.index.min().date()} - {ff.index.max().date()} ({len(ff)} obs)\n")

    df_preds, mase_scale = run_rolling(y, ff)
    logger.info(f"\nTotal records generated: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info("ROLLING RESULTS - MAE / RMSE / MASE by model x horizon")
    logger.info("=" * 60)
    print_table(metrics)

    logger.info("\n" + "=" * 60)
    logger.info("MASE relative to naive (naive = 1.000)")
    logger.info("=" * 60)
    print_mase_table(metrics)

    # Quantify FEDFUNDS contribution (ARIMA vs ARIMAX)
    logger.info("\n" + "=" * 60)
    logger.info("FEDFUNDS BENEFIT (ARIMA vs ARIMAX, delta MAE)")
    logger.info("=" * 60)
    logger.info(f"  {'h':>4}  {'MAE ARIMA':>12}  {'MAE ARIMAX':>12}  {'Delta':>8}  {'Improv%':>8}")
    logger.info(f"  {'-'*52}")
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics.get("arima", {}) and key in metrics.get("arimax", {}):
            mae_a  = metrics["arima"][key]["MAE"]
            mae_ax = metrics["arimax"][key]["MAE"]
            delta  = mae_ax - mae_a
            pct    = -delta / mae_a * 100
            mark   = " <-- improvement" if pct > 0 else ""
            logger.info(f"  {h:>4}  {mae_a:>12.4f}  {mae_ax:>12.4f}  {delta:>+8.4f}  {pct:>+7.1f}%{mark}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = RESULTS_DIR / "rolling_predictions_global.parquet"
    df_preds.to_parquet(preds_path, index=False)
    logger.info(f"\nPredictions saved: {preds_path}")

    metrics_path = RESULTS_DIR / "rolling_metrics_global.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved:     {metrics_path}")

    logger.info("\n" + "=" * 60)
    logger.info("BACKTESTING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
