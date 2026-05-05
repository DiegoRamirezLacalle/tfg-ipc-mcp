"""ARIMA non-seasonal baseline via auto_arima (pmdarima) - Global CPI.

Target series: cpi_global_rate (YoY % rate, median of 186 countries)

Parameters fixed by EDA (notebooks 02-04 _global):
  d = 1   (ADF p=0.010 marginal, PP p=0.145 does not reject, KPSS rejects level stationarity)
  D = 0   (Fs = -0.079: no seasonality, global median cancels national patterns)
  p in [0..4], q in [0..4]  (PACF cuts at lag 3, ACF tails to lag 4 on diff(1))

Key difference vs Spain:
  - Spain:   level index -> SARIMA with D=1
  - Global:  YoY rate already computed -> ARIMA without seasonal component (D=0)

Final selection by AIC.

Input:  data/processed/cpi_global_monthly.parquet
Output: 08_results/arima_global_summary.txt
        08_results/arima_global_metrics.json
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

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.logger import get_logger
from shared.metrics import mae, rmse, mase

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df["cpi_global_rate"]
    y.index.freq = "MS"

    train = y.loc[:DATE_TRAIN_END]
    val   = y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    test  = y.loc[DATE_VAL_END:].iloc[1:]

    return y, train, val, test


def fit_arima(train: pd.Series):
    """auto_arima with d=1, D=0, seasonal=False. p/q up to 4 from ACF/PACF on diff(1)."""
    model = pm.auto_arima(
        train,
        start_p=0, max_p=4,
        start_q=0, max_q=4,
        d=1,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )
    return model


def diagnose_residuals(model, name="arima_global"):
    resid = model.resid()
    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)

    logger.info(f"\n--- Residual diagnostics ({name}) ---")
    logger.info(f"  Mean:  {resid.mean():.6f}")
    logger.info(f"  Std:   {resid.std():.4f}")
    logger.info(f"  Min:   {resid.min():.4f}   Max: {resid.max():.4f}")
    logger.info(f"  Ljung-Box (H0: no autocorrelation):")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "RESIDUAL AUTOCORRELATION"
        logger.info(f"    Lag {lag:2d}: stat={row['lb_stat']:7.3f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return resid, lb


def forecast_and_evaluate(model, train: pd.Series, val: pd.Series):
    n_val = len(val)
    fc, ci = model.predict(n_periods=n_val, return_conf_int=True, alpha=0.05)

    fc_series = pd.Series(fc, index=val.index, name="forecast")

    metrics = {
        "MAE":  round(float(mae(val.values, fc)), 4),
        "RMSE": round(float(rmse(val.values, fc)), 4),
        "MASE": round(float(mase(val.values, fc, train.values, m=12)), 4),
    }
    return fc_series, ci, metrics


def save_results(model, metrics: dict, resid: pd.Series, lb: pd.DataFrame):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "arima_global_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    logger.info(f"\nSummary saved: {summary_path}")

    lb_dict = {
        f"lag_{lag}": {"stat": round(row["lb_stat"], 4), "pvalue": round(row["lb_pvalue"], 4)}
        for lag, row in lb.iterrows()
    }
    out = {
        "model":    "arima_global",
        "order":    list(model.order),
        "aic":      round(float(model.aic()), 4),
        "bic":      round(float(model.bic()), 4),
        "n_train":  int(model.nobs_),
        "residuals": {
            "mean": round(float(resid.mean()), 6),
            "std":  round(float(resid.std()), 4),
            "ljung_box": lb_dict,
        },
        "metrics_val": metrics,
    }
    metrics_path = RESULTS_DIR / "arima_global_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    return out


def main():
    logger.info("=" * 60)
    logger.info("ARIMA AUTO GLOBAL - Baseline CPI Global (d=1, D=0)")
    logger.info("=" * 60)

    y, train, val, test = load_data()
    logger.info(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    logger.info(f"Val:   {val.index.min().date()} -> {val.index.max().date()} ({len(val)} obs)")
    logger.info(f"Test:  {test.index.min().date()} -> {test.index.max().date()} ({len(test)} obs)")
    logger.info(f"\nTrain stats:  mean={train.mean():.4f}  std={train.std():.4f}  "
                f"min={train.min():.4f}  max={train.max():.4f}")

    logger.info("\n--- auto_arima search (d=1, D=0, seasonal=False, p/q max=4) ---")
    model = fit_arima(train)

    order = model.order
    logger.info(f"\nSelected model: ARIMA{order}")
    logger.info(f"AIC: {model.aic():.4f}  |  BIC: {model.bic():.4f}")
    logger.info(f"Parameters:\n{model.summary()}")

    resid, lb = diagnose_residuals(model, f"ARIMA{order}")

    logger.info(f"\n--- Forecast on validation ({len(val)} months) ---")
    fc, ci, metrics = forecast_and_evaluate(model, train, val)

    logger.info(f"\nValidation metrics (2021-01 to 2022-06):")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\n{'Date':>12} {'Actual':>10} {'Pred':>10} {'Error':>10} {'CI_low':>8} {'CI_hi':>8}")
    logger.info("-" * 65)
    for date, real, pred, lo, hi in zip(val.index, val.values, fc.values, ci[:, 0], ci[:, 1]):
        err = real - pred
        flag = " <--" if abs(err) > 1.5 else ""
        logger.info(f"{str(date.date()):>12} {real:10.4f} {pred:10.4f} {err:10.4f} "
                    f"{lo:8.4f} {hi:8.4f}{flag}")

    result = save_results(model, metrics, resid, lb)

    logger.info(f"\n{'=' * 60}")
    logger.info("ARIMA GLOBAL SUMMARY")
    logger.info(f"  Model:    ARIMA{order}")
    logger.info(f"  AIC:      {result['aic']}")
    logger.info(f"  BIC:      {result['bic']}")
    logger.info(f"  MAE val:  {metrics['MAE']}")
    logger.info(f"  RMSE val: {metrics['RMSE']}")
    logger.info(f"  MASE val: {metrics['MASE']}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
