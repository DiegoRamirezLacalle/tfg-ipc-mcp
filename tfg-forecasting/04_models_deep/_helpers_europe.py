"""Shared utilities for deep models - HICP Eurozone."""

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
UNIQUE_ID   = "HICP_EUROPE"


def load_nf_format_europe():
    """Load HICP Eurozone in NeuralForecast long format.

    Returns:
        train, val, df_full, y_train_values
    """
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "ds", "hicp_index": "y"})
    df["unique_id"] = UNIQUE_ID
    df = df[["unique_id", "ds", "y"]].sort_values("ds").reset_index(drop=True)

    train = df[df["ds"] <= DATE_TRAIN_END].copy()
    val   = df[(df["ds"] > DATE_TRAIN_END) & (df["ds"] <= DATE_VAL_END)].copy()

    return train, val, df, train["y"].values


def evaluate_forecast(y_true, y_pred, y_train_values, model_name):
    return {
        "model": model_name,
        "MAE":   round(float(mae(y_true, y_pred)), 4),
        "RMSE":  round(float(rmse(y_true, y_pred)), 4),
        "MASE":  round(float(mase(y_true, y_pred, y_train_values, m=12)), 4),
    }
