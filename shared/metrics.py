"""Shared evaluation metrics: MAE, RMSE, MASE, and Diebold-Mariano test."""

import numpy as np
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 12) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scale = MAE of the seasonal naive forecast on the training set (lag m).
    m=12 for monthly seasonality.
    """
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        raise ValueError("Naive scale is 0; check the training series.")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> dict:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold correction.

    Parameters
    ----------
    e1, e2 : forecast errors for the two models
    h      : forecast horizon (steps)
    power  : 1 → absolute errors, 2 → squared errors

    Returns
    -------
    dict with 'dm_stat', 'p_value', and 'better' ('model1' | 'model2' | 'tie')
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_bar = np.mean(d)

    # variance with autocovariance correction up to lag h-1
    gamma = [np.mean((d - d_bar) * np.roll(d - d_bar, lag)) for lag in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        var_d = gamma[0] / n  # fallback

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * stats.norm.sf(np.abs(dm_stat))

    better = "tie"
    if p_value < 0.05:
        better = "model2" if dm_stat > 0 else "model1"

    return {"dm_stat": round(dm_stat, 4), "p_value": round(p_value, 4), "better": better}


def summary(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 12) -> dict:
    return {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, m),
    }
