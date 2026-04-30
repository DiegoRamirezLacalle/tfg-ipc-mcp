"""Shared utilities for Block 04 deep models — CPI Global pipeline.

Analogous to _helpers.py (Spain) but loads cpi_global_monthly.parquet
with column cpi_global_rate and writes to 08_results/ with _global suffix.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.logger import get_logger
from shared.metrics import mae, rmse, mase

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"


def load_nf_format_global():
    """Load CPI Global and return DataFrames in NeuralForecast long format.

    Returns:
        df_train:       long format up to DATE_TRAIN_END
        df_val:         long format DATE_TRAIN_END+1 to DATE_VAL_END
        df_full:        full long format (for rolling)
        y_train_values: 1D array of train set (for MASE scale)
    """
    df = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df[["cpi_global_rate"]].copy()
    y.index.name = "ds"
    y = y.reset_index()
    y.columns = ["ds", "y"]
    y["unique_id"] = "CPI_GLOBAL"
    y = y[["unique_id", "ds", "y"]]
    y["ds"] = pd.to_datetime(y["ds"])

    train = y[y["ds"] <= DATE_TRAIN_END].copy()
    val   = y[(y["ds"] > DATE_TRAIN_END) & (y["ds"] <= DATE_VAL_END)].copy()

    y_train_values = train["y"].values

    return train, val, y, y_train_values


def evaluate_forecast(y_true, y_pred, y_train_values, model_name):
    """Compute MAE, RMSE, MASE and return formatted dict."""
    return {
        "model": model_name,
        "MAE":   round(float(mae(y_true, y_pred)), 4),
        "RMSE":  round(float(rmse(y_true, y_pred)), 4),
        "MASE":  round(float(mase(y_true, y_pred, y_train_values, m=12)), 4),
    }


def print_comparison(dates, y_true, y_pred):
    """Log point-by-point comparison table."""
    logger.info(f"\n{'Date':>12} {'Actual':>10} {'Pred':>10} {'Error':>10}")
    logger.info("-" * 45)
    for d, real, pred in zip(dates, y_true, y_pred):
        date_str = str(d.date()) if hasattr(d, "date") else str(d)[:10]
        logger.info(f"{date_str:>12} {real:10.4f} {pred:10.4f} {real - pred:10.4f}")
