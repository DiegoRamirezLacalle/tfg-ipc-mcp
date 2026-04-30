"""SARIMAX baseline with ECB Deposit Facility Rate (DFR) as exogenous variable.

Main exogenous variable: dfr (Deposit Facility Rate)
Economic justification: primary ECB monetary policy instrument since 2014,
directly related to consumer inflation.

Note on static evaluation:
  Validation forecast uses the REAL DFR values for 2021-2022 (oracle assumption).
  This is correct for static baseline evaluation; rolling backtesting uses values
  known at each forecast origin, which are also real (DFR is public on the same day).

Input:  data/processed/features_exog.parquet
Output: 03_models_baseline/results/sarimax_summary.txt
        03_models_baseline/results/sarimax_metrics.json
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

RESULTS_DIR = Path(__file__).resolve().parent / "results"

EXOG_COL = "dfr"


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_exog.parquet")
    df.index.freq = "MS"

    y = df["indice_general"]
    X = df[[EXOG_COL]]

    train_mask = y.index <= DATE_TRAIN_END
    val_mask   = (y.index > DATE_TRAIN_END) & (y.index <= DATE_VAL_END)

    y_train, y_val = y[train_mask], y[val_mask]
    X_train, X_val = X[train_mask], X[val_mask]

    return y, X, y_train, y_val, X_train, X_val


def fit_sarimax(y_train, X_train):
    """Fit auto_arima with exogenous variable. Same ranges as SARIMA for comparability."""
    model = pm.auto_arima(
        y_train,
        exogenous=X_train,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=1,
        seasonal=True,
        m=12,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        D=1,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )
    return model


def diagnose_residuals(model):
    resid = model.resid()
    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)

    logger.info("\n--- Residual diagnostics (sarimax) ---")
    logger.info(f"  Mean:  {resid.mean():.6f}")
    logger.info(f"  Std:   {resid.std():.4f}")
    logger.info("  Ljung-Box:")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "AUTOCORRELATION"
        logger.info(f"    Lag {lag:2d}: stat={row['lb_stat']:.2f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return lb


def forecast_and_evaluate(model, y_train, y_val, X_val):
    n_val = len(y_val)
    fc, ci = model.predict(
        n_periods=n_val,
        exogenous=X_val,
        return_conf_int=True,
        alpha=0.05,
    )

    fc_series = pd.Series(fc, index=y_val.index, name="forecast")
    metrics = {
        "MAE":  round(mae(y_val.values, fc), 4),
        "RMSE": round(rmse(y_val.values, fc), 4),
        "MASE": round(mase(y_val.values, fc, y_train.values, m=12), 4),
    }
    return fc_series, ci, metrics


def load_previous_metrics():
    """Load ARIMA and SARIMA metrics for comparison."""
    results = {}
    for name in ["arima", "sarima"]:
        path = RESULTS_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def save_results(model, metrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "sarimax_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    logger.info(f"\nSummary saved: {summary_path}")

    out = {
        "model":          "sarimax",
        "exog":           EXOG_COL,
        "order":          list(model.order),
        "seasonal_order": list(model.seasonal_order),
        "aic":            round(model.aic(), 4),
        "bic":            round(model.bic(), 4),
        "n_train":        int(model.nobs_),
        "metrics_val":    metrics,
    }
    metrics_path = RESULTS_DIR / "sarimax_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    return out


def main():
    logger.info("=" * 60)
    logger.info(f"SARIMAX — baseline with exogenous: {EXOG_COL}")
    logger.info("=" * 60)

    y, X, y_train, y_val, X_train, X_val = load_data()
    logger.info(f"Train: {y_train.index.min().date()} -> {y_train.index.max().date()} ({len(y_train)} obs)")
    logger.info(f"Val:   {y_val.index.min().date()} -> {y_val.index.max().date()} ({len(y_val)} obs)")
    logger.info(f"DFR train: min={X_train[EXOG_COL].min():.2f}  max={X_train[EXOG_COL].max():.2f}")
    logger.info(f"DFR val:   min={X_val[EXOG_COL].min():.2f}  max={X_val[EXOG_COL].max():.2f}")

    logger.info("\n--- auto_arima search with exogenous ---")
    model = fit_sarimax(y_train, X_train)

    order   = model.order
    s_order = model.seasonal_order
    logger.info(f"\nSelected model: SARIMAX{order}x{s_order}")
    logger.info(f"AIC: {model.aic():.2f}  |  BIC: {model.bic():.2f}")
    logger.info(str(model.summary()))

    diagnose_residuals(model)

    logger.info(f"\n--- Forecast on validation ({len(y_val)} months) ---")
    fc, ci, metrics = forecast_and_evaluate(model, y_train, y_val, X_val)

    logger.info("\nValidation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\n{'Date':>12} {'Actual':>10} {'Pred':>10} {'Error':>10} {'DFR':>8}")
    logger.info("-" * 50)
    for date, real, pred, dfr_val in zip(y_val.index, y_val.values, fc.values, X_val[EXOG_COL].values):
        logger.info(f"{str(date.date()):>12} {real:10.3f} {pred:10.3f} {real - pred:10.3f} {dfr_val:8.2f}")

    result = save_results(model, metrics)

    # Comparison with ARIMA and SARIMA
    prev = load_previous_metrics()
    if prev:
        logger.info(f"\n{'=' * 60}")
        logger.info("ARIMA / SARIMA / SARIMAX comparison (validation)")
        logger.info(f"{'=' * 60}")
        header = f"{'Metric':<8}"
        for name in ["arima", "sarima", "sarimax"]:
            if name in prev or name == "sarimax":
                header += f" {name.upper():>10}"
        logger.info(header)
        logger.info("-" * 50)

        all_metrics = {**prev, "sarimax": result}
        for m_name in ["MAE", "RMSE", "MASE"]:
            row = f"{m_name:<8}"
            for name in ["arima", "sarima", "sarimax"]:
                if name in all_metrics:
                    row += f" {all_metrics[name]['metrics_val'][m_name]:10.4f}"
            logger.info(row)

        logger.info("\nAIC:")
        for name in ["arima", "sarima", "sarimax"]:
            if name in all_metrics:
                logger.info(f"  {name.upper():<8}: {all_metrics[name]['aic']:.2f}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY: SARIMAX{order}x{s_order}  exog={EXOG_COL}")
    logger.info(f"  AIC={result['aic']}  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
