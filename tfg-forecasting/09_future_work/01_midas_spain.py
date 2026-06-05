"""
01_midas_spain.py  —  MIDAS-ADL regression for Spain CPI forecasting
=====================================================================

MIDAS (Mixed Data Sampling) constrains the lag polynomial of exogenous
variables to a low-dimensional Beta function instead of estimating one
free coefficient per lag.  This enforces smooth temporal decay and avoids
overfitting even when K (number of lags) is large relative to the sample.

    y_{t+h} = α + ρ·y_t  +  Σ_j  β_j · (w_j(θ_j) · x_j_{t:t-K})  + ε

where  w_k(θ₁,θ₂) ∝ (k/K)^(θ₁-1) · (1-k/K)^(θ₂-1)   (Beta polynomial)

Three exogenous variables, K=12 monthly lags each:
  · brent_log     — Brent crude log-price (energy input cost signal)
  · epu_europe_log — Baker-Bloom-Davis EPU index (policy uncertainty)
  · dfr           — ECB deposit-facility rate (monetary tightening)

Rolling-origin backtest: 48 monthly origins 2021-01 → 2024-12.
Direct multi-horizon: separate MIDAS model per h ∈ {1, 3, 6, 12}.

State-of-art context
--------------------
MIDAS is the canonical mixed-frequency benchmark dating to Ghysels et al.
(2004) and its ML extensions (ML-MIDAS, 2020).  The Beta polynomial weight
constraint is still used as a building block inside more recent methods
(SpecTF 2026 spectral text-series fusion, TimeCMA 2024).  Compared to the
Ridge-XReg correction in scripts 11–12, MIDAS does not assume fixed pre-
computed lag aggregates — it jointly optimises the aggregation weights.

References
----------
· Ghysels, Santa-Clara & Valkanov (2004) — original MIDAS
· Clements & Galvão (2009) — MIDAS for macro forecasting
· Babii, Ghysels & Striaukas (2020) — ML-MIDAS (arXiv:2005.14057)
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
MODEL_NAME = "midas_adl_spain"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

# MIDAS configuration
K = 12          # number of lags per variable
XREG_VARS = ["brent_log", "epu_europe_log", "dfr"]   # exogenous series

# Beta polynomial parameter bounds: θ ∈ (0.5, 12)
# We optimise log(θ - 0.5) to keep θ > 0.5 unconditionally.
_N_VARS = len(XREG_VARS)
_N_PARAMS = 2 + _N_VARS * 3   # intercept + AR(1) + 3 per variable (β, logθ₁, logθ₂)


# ── Beta polynomial ────────────────────────────────────────────────────────────

def beta_weights(log_t1: float, log_t2: float, K: int) -> np.ndarray:
    """Normalised Beta polynomial weights for K lag positions (1..K)."""
    theta1 = np.exp(log_t1) + 0.5
    theta2 = np.exp(log_t2) + 0.5
    u = np.arange(1, K + 1) / K          # grid on (0,1]
    w = u ** (theta1 - 1) * (1 - u + 1e-9) ** (theta2 - 1)
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(K) / K


# ── Lag matrix builder ─────────────────────────────────────────────────────────

def build_lag_matrix(series: np.ndarray, K: int) -> np.ndarray:
    """
    Return (T, K) array where row t contains [series[t-1], ..., series[t-K]].
    Rows with insufficient history are filled with NaN.
    """
    T = len(series)
    out = np.full((T, K), np.nan)
    for k in range(1, K + 1):
        out[k:, k - 1] = series[: T - k]
    return out


# ── NLS objective ─────────────────────────────────────────────────────────────

def midas_predict(params: np.ndarray, y_ar: np.ndarray,
                  lag_matrices: list[np.ndarray]) -> np.ndarray:
    alpha = params[0]
    rho   = params[1]
    pred  = alpha + rho * y_ar
    idx = 2
    for X_lags in lag_matrices:
        beta   = params[idx]
        log_t1 = params[idx + 1]
        log_t2 = params[idx + 2]
        w = beta_weights(log_t1, log_t2, X_lags.shape[1])
        z = X_lags @ w
        pred = pred + beta * z
        idx += 3
    return pred


def nls_loss(params: np.ndarray, y_target: np.ndarray, y_ar: np.ndarray,
             lag_matrices: list[np.ndarray]) -> float:
    pred = midas_predict(params, y_ar, lag_matrices)
    return float(np.sum((y_target - pred) ** 2))


def fit_midas(y_target: np.ndarray, y_ar: np.ndarray,
              lag_matrices: list[np.ndarray]) -> np.ndarray | None:
    """Fit MIDAS parameters by NLS. Returns optimised params or None on failure."""
    x0 = np.zeros(_N_PARAMS)
    x0[1] = 0.8       # AR coefficient near 1 (inflation is persistent)
    # Initialise θ₁=1.5, θ₂=2.5 → gently decaying, recent lags slightly favoured
    for i in range(_N_VARS):
        x0[2 + i * 3 + 1] = np.log(1.0)   # log(θ₁ - 0.5) = 0  → θ₁ ≈ 1.5
        x0[2 + i * 3 + 2] = np.log(2.0)   # log(θ₂ - 0.5)      → θ₂ ≈ 2.5

    try:
        res = minimize(
            nls_loss, x0,
            args=(y_target, y_ar, lag_matrices),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5, "disp": False},
        )
        return res.x
    except Exception:
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    return df


# ── Rolling-origin backtesting ────────────────────────────────────────────────

def run_rolling(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    y_series = df["indice_general"]
    y_arr = y_series.values.astype(np.float64)
    dates = y_series.index

    # Exogenous series as arrays (same index as y)
    x_arrays = {v: df[v].ffill().bfill().values.astype(np.float64) for v in XREG_VARS}

    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train = y_series.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(y_train.values[12:] - y_train.values[:-12])))

    records = []

    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
        # Index of origin in the full series
        if origin not in dates:
            continue
        t_idx = dates.get_loc(origin)

        # Build full-history lag matrices up to t_idx (inclusive)
        lag_mats = [build_lag_matrix(x_arrays[v], K) for v in XREG_VARS]

        for h in HORIZONS:
            target_ts = origin + pd.DateOffset(months=h)
            if target_ts > TEST_END_TS:
                continue
            if target_ts not in dates:
                continue
            t_h_idx = dates.get_loc(target_ts)

            # Training window: all observations strictly before origin, that have
            # valid lags (need at least K months of history) AND a target at t+h
            # We use expanding window: train on everything up to origin - h
            train_end_idx = t_idx  # the target at each train point is y[i + h]
            # Valid train rows: rows i where lag matrices are complete and y[i+h] exists
            valid_rows = []
            for i in range(K, train_end_idx - h + 1):
                if i + h < len(y_arr) and not any(
                    np.any(np.isnan(lm[i, :])) for lm in lag_mats
                ):
                    valid_rows.append(i)

            if len(valid_rows) < 30:   # need enough data to fit
                continue

            valid_rows = np.array(valid_rows)
            y_target_train = y_arr[valid_rows + h]
            y_ar_train = y_arr[valid_rows]
            lag_mats_train = [lm[valid_rows, :] for lm in lag_mats]

            params = fit_midas(y_target_train, y_ar_train, lag_mats_train)
            if params is None:
                continue

            # Predict: at origin t, AR = y[t], lags = last K values of each x
            y_ar_now = np.array([y_arr[t_idx]])
            lag_mats_now = [lm[t_idx : t_idx + 1, :] for lm in lag_mats]

            if any(np.any(np.isnan(lm)) for lm in lag_mats_now):
                continue

            y_pred = float(midas_predict(params, y_ar_now, lag_mats_now)[0])
            y_true = float(y_arr[t_h_idx])

            records.append({
                "origin": origin, "fc_date": target_ts,
                "step": h, "horizon": h,
                "model": MODEL_NAME,
                "y_true": y_true, "y_pred": y_pred,
                "error": y_true - y_pred,
                "abs_error": abs(y_true - y_pred),
            })

    return pd.DataFrame(records), mase_scale


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    res = {}
    for h in HORIZONS:
        hd = df_preds[df_preds["horizon"] == h]
        if hd.empty:
            continue
        yt, yp = hd["y_true"].values, hd["y_pred"].values
        res[f"h{h}"] = {
            "MAE":  round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def compare_to_baselines(new: dict) -> None:
    """Print delta vs ARIMA and TimesFM C0 from stored metrics."""
    baseline_path = ROOT / "08_results" / "metrics_summary_final.json"
    if not baseline_path.exists():
        return
    base = json.load(open(baseline_path))
    logger.info("\n%s", "-" * 62)
    logger.info("%-14s  %s", "model", "  ".join(f"h={h:2d}" for h in HORIZONS))
    logger.info("-" * 62)
    for ref_name in ["arima", "timesfm_C0"]:
        ref = base.get(ref_name, {})
        row = f"{ref_name:<14}"
        for h in HORIZONS:
            v = ref.get(f"h{h}", {}).get("MAE")
            row += f"  {v:7.4f}" if v else "       ?"
        logger.info(row)
    row = f"{MODEL_NAME:<14}"
    for h in HORIZONS:
        v = new.get(f"h{h}", {}).get("MAE")
        row += f"  {v:7.4f}" if v else "       ?"
    logger.info(row)
    logger.info("\nΔ MIDAS vs ARIMA:")
    for h in HORIZONS:
        m = new.get(f"h{h}", {}).get("MAE")
        r = base.get("arima", {}).get(f"h{h}", {}).get("MAE")
        if m and r:
            logger.info("  h=%d: %+.1f%%", h, (m - r) / r * 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 62)
    logger.info("MIDAS-ADL  —  Spain CPI  —  Beta polynomial lag weights")
    logger.info("Variables : %s  |  K=%d lags each", XREG_VARS, K)
    logger.info("Origins   : %s → %s", ORIGINS_START, ORIGINS_END)
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
