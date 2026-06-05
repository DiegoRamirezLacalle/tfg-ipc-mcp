"""
02_midas_global.py  —  MIDAS-ADL regression for Global CPI forecasting
=======================================================================

Same MIDAS-ADL architecture as 01_midas_spain.py but applied to the
Global CPI series with global institutional exogenous variables.

Variables (K=12 monthly lags each):
  · brent_log     — Brent crude log-price (global energy cost)
  · imf_comm      — IMF All-Commodity Index (broad commodity signal)
  · gscpi         — NY Fed Global Supply Chain Pressure Index
  · fedfunds      — US Federal Funds Rate (global monetary anchor)

These variables were selected by correlation with Global CPI changes
(see c1_global_inst_selected_cols.json) and are the same used by the
best-performing Chronos-2 C1_inst_global (MASE h=12 = 0.976).

MIDAS improves over the existing Ridge approach by:
  1. Estimating the optimal lag aggregation weights (not imposing equal or
     fixed ad-hoc weights like MA3/lag1).
  2. Enforcing smooth temporal decay via the Beta polynomial, preventing
     the lag-12 coefficient from overpowering more recent lags.

State-of-art context
--------------------
ML-MIDAS (Babii et al., 2020) demonstrates MIDAS outperforms standard
distributed lag models for macro nowcasting.  The Factor-augmented sparse
MIDAS variant (2023) further reduces overfitting with high-dimensional
input — directly analogous to our multi-variable global setup.

Reference: Babii, Ghysels & Striaukas (2020) arXiv:2005.14057
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "09_future_work" / "results"
HORIZONS = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "midas_adl_global"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
TARGET_COL = "cpi_global_rate"

K = 12
XREG_VARS = ["brent_log", "imf_comm", "gscpi", "fedfunds"]
_N_VARS = len(XREG_VARS)
_N_PARAMS = 2 + _N_VARS * 3


def beta_weights(log_t1: float, log_t2: float, K: int) -> np.ndarray:
    theta1 = np.exp(log_t1) + 0.5
    theta2 = np.exp(log_t2) + 0.5
    u = np.arange(1, K + 1) / K
    w = u ** (theta1 - 1) * (1 - u + 1e-9) ** (theta2 - 1)
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(K) / K


def build_lag_matrix(series: np.ndarray, K: int) -> np.ndarray:
    T = len(series)
    out = np.full((T, K), np.nan)
    for k in range(1, K + 1):
        out[k:, k - 1] = series[: T - k]
    return out


def midas_predict(params: np.ndarray, y_ar: np.ndarray,
                  lag_matrices: list[np.ndarray]) -> np.ndarray:
    alpha = params[0]
    rho = params[1]
    pred = alpha + rho * y_ar
    idx = 2
    for X_lags in lag_matrices:
        beta = params[idx]
        log_t1 = params[idx + 1]
        log_t2 = params[idx + 2]
        w = beta_weights(log_t1, log_t2, X_lags.shape[1])
        pred = pred + beta * (X_lags @ w)
        idx += 3
    return pred


def nls_loss(params, y_target, y_ar, lag_matrices):
    pred = midas_predict(params, y_ar, lag_matrices)
    return float(np.sum((y_target - pred) ** 2))


def fit_midas(y_target, y_ar, lag_matrices):
    x0 = np.zeros(_N_PARAMS)
    x0[1] = 0.8
    for i in range(_N_VARS):
        x0[2 + i * 3 + 1] = 0.0
        x0[2 + i * 3 + 2] = np.log(2.0)
    try:
        res = minimize(nls_loss, x0, args=(y_target, y_ar, lag_matrices),
                       method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5, "disp": False})
        return res.x
    except Exception:
        return None


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_global_institutional.parquet")
    df.index = pd.to_datetime(df.index)
    df.index.freq = "MS"
    # Some columns may use different names — normalise to base names
    rename = {
        "brent_log_ma3": "brent_log",
        "imf_comm_ma3": "imf_comm",
        "gscpi_ma3": "gscpi",
        "fedfunds_ma3": "fedfunds",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df


def run_rolling(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    y_series = df[TARGET_COL]
    y_arr = y_series.values.astype(np.float64)
    dates = y_series.index

    x_arrays = {}
    for v in XREG_VARS:
        if v in df.columns:
            x_arrays[v] = df[v].ffill().bfill().values.astype(np.float64)
        else:
            logger.warning("[!] Column %s not found — skipping.", v)

    active_vars = [v for v in XREG_VARS if v in x_arrays]
    n_active = len(active_vars)
    n_params = 2 + n_active * 3

    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    y_train = y_series.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(y_train.values[12:] - y_train.values[:-12])))

    records = []
    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
        if origin not in dates:
            continue
        t_idx = dates.get_loc(origin)
        lag_mats = [build_lag_matrix(x_arrays[v], K) for v in active_vars]

        for h in HORIZONS:
            target_ts = origin + pd.DateOffset(months=h)
            if target_ts > TEST_END_TS or target_ts not in dates:
                continue
            t_h_idx = dates.get_loc(target_ts)

            valid_rows = [i for i in range(K, t_idx - h + 1)
                          if i + h < len(y_arr) and
                          not any(np.any(np.isnan(lm[i, :])) for lm in lag_mats)]
            if len(valid_rows) < 30:
                continue

            valid_rows = np.array(valid_rows)

            # Use n_active-aware version
            def _loss(params):
                alpha = params[0]; rho = params[1]
                pred = alpha + rho * y_arr[valid_rows]
                idx = 2
                for lm in lag_mats:
                    b = params[idx]; lt1 = params[idx+1]; lt2 = params[idx+2]
                    w = beta_weights(lt1, lt2, K)
                    pred = pred + b * (lm[valid_rows, :] @ w)
                    idx += 3
                return float(np.sum((y_arr[valid_rows + h] - pred)**2))

            x0 = np.zeros(n_params); x0[1] = 0.8
            for i in range(n_active):
                x0[2 + i*3 + 2] = np.log(2.0)
            try:
                res = minimize(_loss, x0, method="Nelder-Mead",
                               options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5, "disp": False})
                params_opt = res.x
            except Exception:
                continue

            y_ar_now = np.array([y_arr[t_idx]])
            lm_now = [lm[t_idx:t_idx+1, :] for lm in lag_mats]
            if any(np.any(np.isnan(l)) for l in lm_now):
                continue

            alpha = params_opt[0]; rho = params_opt[1]
            pred = alpha + rho * y_arr[t_idx]
            idx = 2
            for lm in lm_now:
                b = params_opt[idx]; lt1 = params_opt[idx+1]; lt2 = params_opt[idx+2]
                w = beta_weights(lt1, lt2, K)
                pred = pred + b * float(lm @ w)
                idx += 3

            y_pred = float(pred)
            y_true = float(y_arr[t_h_idx])
            records.append({
                "origin": origin, "fc_date": target_ts,
                "step": h, "horizon": h,
                "model": MODEL_NAME,
                "y_true": y_true, "y_pred": y_pred,
                "error": y_true - y_pred, "abs_error": abs(y_true - y_pred),
            })

    return pd.DataFrame(records), mase_scale


def compute_metrics(df_preds, mase_scale):
    res = {}
    for h in HORIZONS:
        hd = df_preds[df_preds["horizon"] == h]
        if hd.empty:
            continue
        yt, yp = hd["y_true"].values, hd["y_pred"].values
        res[f"h{h}"] = {
            "MAE": round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp)**2))), 4),
            "MASE": round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def compare_to_baselines(new):
    path = ROOT / "08_results" / "rolling_metrics_global.json"
    if not path.exists():
        return
    base = json.load(open(path))
    logger.info("\n%-30s %9s %9s %9s %9s", "model", "h1", "h3", "h6", "h12")
    logger.info("-" * 65)
    for ref_name in ["arima", "auto_arima"]:
        ref = base.get(ref_name, {})
        if not ref:
            continue
        row = "%-30s" % ref_name
        for h in HORIZONS:
            v = ref.get(f"h{h}", {}).get("MAE")
            row += " %9.4f" % v if v else "          ?"
        logger.info(row)
    # Chronos-2 C1_inst (best existing model)
    p = ROOT / "08_results" / "chronos2_C1_inst_global_metrics.json"
    if p.exists():
        c = json.load(open(p)).get("chronos2_C1_inst_global", {})
        row = "%-30s" % "chronos2_C1_inst_global"
        for h in HORIZONS:
            v = c.get(f"h{h}", {}).get("MAE")
            row += " %9.4f" % v if v else "          ?"
        logger.info(row)
    row = "%-30s" % MODEL_NAME
    for h in HORIZONS:
        v = new.get(f"h{h}", {}).get("MAE")
        row += " %9.4f" % v if v else "          ?"
    logger.info(row)
    logger.info("\nΔ MIDAS vs ARIMA:")
    arima = base.get("arima", {})
    for h in HORIZONS:
        m = new.get(f"h{h}", {}).get("MAE")
        r = arima.get(f"h{h}", {}).get("MAE")
        if m and r:
            logger.info("  h=%d: %+.1f%%", h, (m - r) / r * 100)


def main():
    logger.info("=" * 62)
    logger.info("MIDAS-ADL  —  Global CPI  —  Beta polynomial lag weights")
    logger.info("Variables : %s  |  K=%d lags", XREG_VARS, K)
    logger.info("=" * 62)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    logger.info("Data: %d obs (%s → %s)", len(df),
                df.index.min().date(), df.index.max().date())

    df_preds, mase_scale = run_rolling(df)
    if df_preds.empty:
        logger.warning("[!] No predictions generated.")
        return

    metrics = compute_metrics(df_preds, mase_scale)
    logger.info("\n%-12s %8s %8s %8s %5s", "Horizon", "MAE", "RMSE", "MASE", "N")
    logger.info("-" * 45)
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]
            logger.info("h=%-10d %8.4f %8.4f %8.4f %5d",
                        h, m["MAE"], m["RMSE"], m["MASE"], m["n_evals"])

    compare_to_baselines(metrics)

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    out = RESULTS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(out, "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info("\nSaved: %s", out.name)


if __name__ == "__main__":
    main()
