"""
Métricas de evaluación compartidas.
MAE, RMSE, MASE y test de Diebold-Mariano.
"""

import numpy as np
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 12) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006). m=12 para estacionalidad mensual.

    La escala es el MAE del naive estacional sobre el train set:
    scale = mean(|y_t - y_{t-m}|) para t = m, ..., T
    """
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_errors)
    if scale == 0:
        raise ValueError("Escala naive es 0; revisa la serie de entrenamiento.")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> dict:
    """
    Test Diebold-Mariano (Harvey et al., 1997).

    Parámetros
    ----------
    e1, e2 : errores de predicción de los dos modelos
    h      : horizonte de predicción (pasos)
    power  : 1 → errores absolutos, 2 → errores cuadráticos

    Devuelve
    --------
    dict con 'dm_stat', 'p_value' y 'better' ('model1' | 'model2' | 'tie')
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_bar = np.mean(d)

    # varianza con corrección de autocovarianza hasta lag h-1
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
