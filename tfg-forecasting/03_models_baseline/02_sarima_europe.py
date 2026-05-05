"""SARIMA canonical model for HICP Eurozone.

Reference model: SARIMA(0,1,1)(0,1,1)12
Same as the canonical Spain model (airline model).

Output:
  08_results/sarima_europe_summary.txt
  08_results/sarima_europe_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.logger import get_logger
from shared.metrics import mae, rmse, mase

logger = get_logger(__name__)

RESULTS_DIR    = ROOT / "08_results"
ORDER          = (0, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 12)


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    y = df["hicp_index"]
    return y.loc[:DATE_TRAIN_END], y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]


def main():
    logger.info("=" * 60)
    logger.info(f"SARIMA{ORDER}x{SEASONAL_ORDER} - HICP Eurozone")
    logger.info("=" * 60)

    train, val = load_data()
    logger.info(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")

    mod = SARIMAX(train, order=ORDER, seasonal_order=SEASONAL_ORDER, trend="n")
    res = mod.fit(disp=False)
    logger.info(res.summary())

    lb = acorr_ljungbox(res.resid, lags=[6, 12, 24], return_df=True)
    logger.info("\nLjung-Box:")
    logger.info(str(lb[["lb_stat", "lb_pvalue"]].round(4)))

    fc = res.get_forecast(steps=len(val)).predicted_mean
    m_val = {
        "MAE":  round(mae(val.values, fc.values), 4),
        "RMSE": round(rmse(val.values, fc.values), 4),
        "MASE": round(mase(val.values, fc.values, train.values), 4),
    }
    logger.info(f"\nValidation metrics: {m_val}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "sarima_europe_summary.txt").write_text(
        f"SARIMA{ORDER}x{SEASONAL_ORDER}\n"
        f"AIC={res.aic:.2f}  BIC={res.bic:.2f}\n"
        f"MAE_val={m_val['MAE']}  MASE_val={m_val['MASE']}\n"
    )
    with open(RESULTS_DIR / "sarima_europe_metrics.json", "w") as f:
        json.dump({"order": list(ORDER), "seasonal_order": list(SEASONAL_ORDER),
                   "aic": round(res.aic, 2), "metrics_val": m_val}, f, indent=2)

    logger.info(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
