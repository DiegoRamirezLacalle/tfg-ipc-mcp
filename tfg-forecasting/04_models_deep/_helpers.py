"""Shared utilities for Block 04 deep models.

NeuralForecast expects long format (unique_id, ds, y).
Centralises conversion and evaluation to avoid duplication
across LSTM, N-BEATS, and N-HiTS scripts.
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

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_nf_format():
    """Load IPC Spain and return DataFrames in NeuralForecast long format.

    Returns:
        df_train:       long format up to DATE_TRAIN_END
        df_val:         long format DATE_TRAIN_END+1 to DATE_VAL_END
        df_full:        full long format (for rolling)
        y_train_values: 1D array of train set (for MASE scale)
    """
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df[["indice_general"]].copy()
    y.index.name = "ds"
    y = y.reset_index()
    y.columns = ["ds", "y"]
    y["unique_id"] = "IPC_ESP"
    y = y[["unique_id", "ds", "y"]]

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
        logger.info(f"{date_str:>12} {real:10.3f} {pred:10.3f} {real - pred:10.3f}")
