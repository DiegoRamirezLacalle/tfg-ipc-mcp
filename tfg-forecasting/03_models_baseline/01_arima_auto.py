"""ARIMA baseline (non-seasonal) via auto_arima (pmdarima).

Search ranges informed by ACF/PACF analysis (notebook 04):
  - d=1 confirmed by ADF/KPSS/PP
  - p candidates: 0-3 (PACF significant at lag 1)
  - q candidates: 0-3 (ACF significant at lag 1)

Final selection by AIC. Seasonal component handled in 02_sarima.

Input:  data/processed/ipc_spain_index.parquet
Output: 03_models_baseline/results/arima_summary.txt
        03_models_baseline/results/arima_metrics.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END, FORECAST_HORIZON
from shared.logger import get_logger
from shared.metrics import mae, rmse, mase

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df["indice_general"]
    y.index.freq = "MS"

    train = y.loc[:DATE_TRAIN_END]
    val   = y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    test  = y.loc[DATE_VAL_END:].iloc[1:]

    return y, train, val, test


def fit_arima(train):
    """Fit auto_arima with ranges informed by ACF/PACF."""
    model = pm.auto_arima(
        train,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=1,                    # fixed by stationarity tests
        seasonal=False,         # seasonal component handled in 02_sarima
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )
    return model


def diagnose_residuals(model, model_name):
    """Residual diagnostics: Ljung-Box and basic statistics."""
    resid = model.resid()
    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)

    logger.info(f"\n--- Residual diagnostics ({model_name}) ---")
    logger.info(f"  Mean:  {resid.mean():.6f}")
    logger.info(f"  Std:   {resid.std():.4f}")
    logger.info(f"  Ljung-Box:")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "AUTOCORRELATION"
        logger.info(f"    Lag {lag:2d}: stat={row['lb_stat']:.2f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return lb


def forecast_and_evaluate(model, train, val):
    """Forecast on validation set and compute metrics."""
    n_val = len(val)
    fc, ci = model.predict(n_periods=n_val, return_conf_int=True, alpha=0.05)

    fc_series = pd.Series(fc, index=val.index, name="forecast")

    metrics = {
        "MAE":  round(mae(val.values, fc_series.values), 4),
        "RMSE": round(rmse(val.values, fc_series.values), 4),
        "MASE": round(mase(val.values, fc_series.values, train.values, m=12), 4),
    }

    return fc_series, ci, metrics


def save_results(model, metrics, model_name):
    """Save model summary and metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Model summary
    summary_path = RESULTS_DIR / f"{model_name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    logger.info(f"\nSummary saved: {summary_path}")

    # Metrics JSON
    metrics_out = {
        "model":      model_name,
        "order":      list(model.order),
        "aic":        round(model.aic(), 4),
        "bic":        round(model.bic(), 4),
        "n_train":    int(model.nobs_),
        "metrics_val": metrics,
    }
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    return metrics_out


def main():
    logger.info("=" * 60)
    logger.info("ARIMA AUTO - non-seasonal baseline")
    logger.info("=" * 60)

    y, train, val, test = load_data()
    logger.info(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    logger.info(f"Val:   {val.index.min().date()} -> {val.index.max().date()} ({len(val)} obs)")
    logger.info(f"Test:  {test.index.min().date()} -> {test.index.max().date()} ({len(test)} obs)")

    # Fit
    logger.info("\n--- auto_arima search ---")
    model = fit_arima(train)
    logger.info(f"\nSelected model: ARIMA{model.order}")
    logger.info(f"AIC: {model.aic():.2f}  |  BIC: {model.bic():.2f}")
    logger.info(str(model.summary()))

    # Diagnostics
    diagnose_residuals(model, "arima")

    # Forecast on validation
    logger.info(f"\n--- Forecast on validation ({len(val)} months) ---")
    fc, ci, metrics = forecast_and_evaluate(model, train, val)

    logger.info("\nValidation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # Point-by-point comparison
    logger.info(f"\n{'Date':>12} {'Actual':>10} {'Pred':>10} {'Error':>10}")
    logger.info("-" * 45)
    for date, real, pred in zip(val.index, val.values, fc.values):
        logger.info(f"{str(date.date()):>12} {real:10.3f} {pred:10.3f} {real - pred:10.3f}")

    # Save
    result = save_results(model, metrics, "arima")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY: ARIMA{model.order}")
    logger.info(f"  AIC={result['aic']}  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
