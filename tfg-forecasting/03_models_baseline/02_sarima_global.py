"""ARIMA(1,1,1) fixed as simple reference model - Global CPI.

Purpose: compare the auto-selected ARIMA(3,1,0) from script 01 against a
lower-complexity ARIMA(1,1,1), which is the canonical Box-Jenkins starting
point and serves as a parsimony benchmark.

File is named sarima_global to maintain symmetry with Spain, where script 02
is the seasonal SARIMA. Here there is no seasonality (D=0, Fs=-0.08), so the
"seasonal version" for global is simply a simpler ARIMA.

Models fitted:
  - ARIMA(1,1,1): simple reference model
  - ARIMA(3,1,0): best by AIC (recovered from script 01 JSON)

Input:  data/processed/cpi_global_monthly.parquet
        08_results/arima_global_metrics.json
Output: 08_results/arima111_global_summary.txt
        08_results/arima111_global_metrics.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.logger import get_logger
from shared.metrics import mae, rmse, mase

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"

# Fixed reference order
ARIMA_REF_ORDER = (1, 1, 1)


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df["cpi_global_rate"]
    y.index.freq = "MS"
    train = y.loc[:DATE_TRAIN_END]
    val   = y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    test  = y.loc[DATE_VAL_END:].iloc[1:]
    return y, train, val, test


def fit_arima_fixed(y_train: pd.Series, order: tuple):
    """Fixed-order fit using statsmodels SARIMAX as pure ARIMA (no seasonal)."""
    mod = SARIMAX(y_train, order=order, trend="n")
    return mod.fit(disp=False)


def diagnose_residuals(result, name: str):
    resid = result.resid
    lb = acorr_ljungbox(resid.dropna(), lags=[6, 12, 24], return_df=True)

    logger.info(f"\n--- Residual diagnostics ({name}) ---")
    logger.info(f"  Mean:  {resid.mean():.6f}")
    logger.info(f"  Std:   {resid.std():.4f}")
    logger.info(f"  Min:   {resid.min():.4f}   Max: {resid.max():.4f}")
    logger.info(f"  Ljung-Box (H0: no autocorrelation):")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "RESIDUAL AUTOCORRELATION"
        logger.info(f"    Lag {lag:2d}: stat={row['lb_stat']:7.3f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return resid, lb


def forecast_val(result, n_steps: int, val_index):
    """Forecast n_steps from end of train."""
    fc_obj = result.get_forecast(steps=n_steps)
    fc   = fc_obj.predicted_mean.values
    ci   = fc_obj.conf_int(alpha=0.05).values
    return pd.Series(fc, index=val_index), ci


def compute_metrics(y_true, y_pred, y_train):
    return {
        "MAE":  round(float(mae(y_true, y_pred)),              4),
        "RMSE": round(float(rmse(y_true, y_pred)),             4),
        "MASE": round(float(mase(y_true, y_pred, y_train, m=12)), 4),
    }


def save_results(result, metrics: dict, lb: pd.DataFrame, order: tuple):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"arima{''.join(str(x) for x in order)}_global"

    summary_path = RESULTS_DIR / f"{name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(result.summary()))
    logger.info(f"Summary saved: {summary_path}")

    lb_dict = {
        f"lag_{lag}": {"stat": round(row["lb_stat"], 4), "pvalue": round(row["lb_pvalue"], 4)}
        for lag, row in lb.iterrows()
    }
    out = {
        "model":    name,
        "order":    list(order),
        "aic":      round(float(result.aic), 4),
        "bic":      round(float(result.bic), 4),
        "n_train":  int(result.nobs),
        "residuals": {
            "mean": round(float(result.resid.mean()), 6),
            "std":  round(float(result.resid.std()),  4),
            "ljung_box": lb_dict,
        },
        "metrics_val": metrics,
    }
    metrics_path = RESULTS_DIR / f"{name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")
    return out


def compare_with_auto():
    """Load auto-selected ARIMA metrics from script 01."""
    path = RESULTS_DIR / "arima_global_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    logger.info("=" * 60)
    logger.info(f"ARIMA{ARIMA_REF_ORDER} GLOBAL - Simple reference model")
    logger.info("=" * 60)

    y, train, val, test = load_data()
    logger.info(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    logger.info(f"Val:   {val.index.min().date()} -> {val.index.max().date()} ({len(val)} obs)")

    logger.info(f"\n--- Fitting ARIMA{ARIMA_REF_ORDER} (fixed order, no constant) ---")
    result = fit_arima_fixed(train, ARIMA_REF_ORDER)
    logger.info(result.summary())

    aic = result.aic
    bic = result.bic
    logger.info(f"\nAIC: {aic:.4f}  |  BIC: {bic:.4f}")

    resid, lb = diagnose_residuals(result, f"ARIMA{ARIMA_REF_ORDER}")

    logger.info(f"\n--- Forecast on validation ({len(val)} months) ---")
    fc, ci = forecast_val(result, len(val), val.index)
    metrics = compute_metrics(val.values, fc.values, train.values)

    logger.info(f"\nValidation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\n{'Date':>12} {'Actual':>10} {'Pred':>10} {'Error':>10}")
    logger.info("-" * 45)
    for date, real, pred in zip(val.index, val.values, fc.values):
        flag = " <--" if abs(real - pred) > 1.5 else ""
        logger.info(f"{str(date.date()):>12} {real:10.4f} {pred:10.4f} {real-pred:10.4f}{flag}")

    result_dict = save_results(result, metrics, lb, ARIMA_REF_ORDER)

    auto = compare_with_auto()
    if auto:
        auto_order = tuple(auto["order"])
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ARIMA{auto_order} (auto) vs ARIMA{ARIMA_REF_ORDER} (reference) comparison")
        logger.info(f"{'=' * 60}")
        logger.info(f"{'Metric':<8} {'ARIMA'+str(auto_order):>14} {'ARIMA'+str(ARIMA_REF_ORDER):>14} {'Diff':>10}")
        logger.info("-" * 50)
        for m_name in ["MAE", "RMSE", "MASE"]:
            a_val = auto["metrics_val"][m_name]
            r_val = metrics[m_name]
            diff  = r_val - a_val
            logger.info(f"{m_name:<8} {a_val:14.4f} {r_val:14.4f} {diff:+10.4f}")
        logger.info(f"\n{'AIC':<8} {auto['aic']:14.4f} {aic:14.4f} {aic - auto['aic']:+10.4f}")
        logger.info(f"{'BIC':<8} {auto['bic']:14.4f} {bic:14.4f} {bic - auto['bic']:+10.4f}")
        if auto["aic"] < aic:
            margin = aic - auto["aic"]
            logger.info(f"=> ARIMA{auto_order} preferred by AIC ({margin:.2f} points better).")
            logger.info(f"   ARIMA{auto_order} will be used in backtesting.")
        else:
            margin = aic - auto["aic"]
            logger.info(f"=> ARIMA{ARIMA_REF_ORDER} preferred by AIC ({-margin:.2f} points better).")
            logger.info(f"   Review model choice for backtesting.")
    else:
        logger.info("\n(Run 01_arima_auto_global.py first to see the comparison)")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ARIMA{ARIMA_REF_ORDER} GLOBAL SUMMARY")
    logger.info(f"  AIC: {aic:.4f}  BIC: {bic:.4f}")
    logger.info(f"  MAE val: {metrics['MAE']}  RMSE: {metrics['RMSE']}  MASE: {metrics['MASE']}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
