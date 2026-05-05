"""auto_arima SARIMA for HICP Eurozone.

Series: hicp_europe_index.parquet (level index, base 2015=100)

Parameters fixed by EDA (notebooks 02-04 _europe):
  d=1, D=1, m=12  (same as Spain)
  p in {0,1,2,3}, q in {0,1,2}  (PACF/ACF on diff(1,12))
  P in {1,2},     Q in {1}

Output:
  08_results/arima_europe_summary.txt
  08_results/arima_europe_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.logger import get_logger
from shared.metrics import mae, rmse, mase

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    y = df["hicp_index"]
    train = y.loc[:DATE_TRAIN_END]
    val   = y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    return y, train, val


def fit_sarima(train: pd.Series):
    model = pm.auto_arima(
        train,
        start_p=0, max_p=3,
        start_q=0, max_q=2,
        d=1, D=1, m=12,
        start_P=1, max_P=2,
        start_Q=0, max_Q=1,
        seasonal=True,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )
    return model


def diagnose(model):
    resid = model.resid()
    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)
    logger.info("\nLjung-Box:")
    logger.info(str(lb[["lb_stat", "lb_pvalue"]].round(4)))
    return resid


def main():
    logger.info("=" * 60)
    logger.info("auto_arima SARIMA - HICP Eurozone")
    logger.info("=" * 60)

    y, train, val = load_data()
    logger.info(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    logger.info(f"Val:   {val.index.min().date()} -> {val.index.max().date()} ({len(val)} obs)\n")

    logger.info("Searching best SARIMA(p,1,q)(P,1,Q)12 by AIC...")
    model = fit_sarima(train)

    logger.info(f"\nSelected model: {model.order} x {model.seasonal_order}")
    logger.info(f"AIC: {model.aic():.2f}  |  BIC: {model.bic():.2f}")
    logger.info(f"\nParameters:\n{model.summary()}")

    diagnose(model)

    fc = model.predict(n_periods=len(val))
    m_val = {
        "MAE":  round(mae(val.values, fc), 4),
        "RMSE": round(rmse(val.values, fc), 4),
        "MASE": round(mase(val.values, fc, train.values), 4),
    }
    logger.info(f"\nValidation metrics: {m_val}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    order_str = f"SARIMA{model.order}x{model.seasonal_order}"
    summary_txt = (
        f"{order_str}\n"
        f"AIC={model.aic():.2f}  BIC={model.bic():.2f}\n"
        f"MAE_val={m_val['MAE']}  RMSE_val={m_val['RMSE']}  MASE_val={m_val['MASE']}\n"
    )
    (RESULTS_DIR / "arima_europe_summary.txt").write_text(summary_txt)

    result = {
        "order": list(model.order),
        "seasonal_order": list(model.seasonal_order),
        "aic": round(model.aic(), 2),
        "bic": round(model.bic(), 2),
        "metrics_val": m_val,
    }
    with open(RESULTS_DIR / "arima_europe_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResults saved to {RESULTS_DIR}")
    return model.order, model.seasonal_order


if __name__ == "__main__":
    main()
