"""
01c_midas_daily_spain.py  —  Daily-frequency MIDAS for Spain CPI
=================================================================

MIDAS in its intended setting: a daily input frequency aggregated to a
monthly target via a Beta lag polynomial.  This is what distinguishes MIDAS
from the Ridge-XReg approach (scripts 11-12) and from the monthly-proxy
version (01_midas_spain.py).

Three variants are compared, all predicting the h-step CPI *change*
(delta = y_{t+h} - y_t, stationary) and reconstructing the level as
y_t + delta_pred:

  (A) midas_daily_beta     Beta-MIDAS on daily WTI log-PRICE levels
  (B) midas_daily_returns  Beta-MIDAS on daily WTI log-RETURNS  <- improvement
  (C) midas_daily_ridge    Unconstrained Ridge over the daily lag block (levels)

Why a returns variant (B)?
--------------------------
Daily oil log-PRICES are near-unit-root: within a single month the daily
values barely differ from their monthly mean, so the daily-level lags carry
almost no information beyond what the monthly average already encodes — which
is why variant (A) does not beat the monthly proxy.  Daily log-RETURNS, by
contrast, capture *within-month momentum and volatility* (how fast and how
erratically oil moved), a genuinely sub-monthly signal.  Variant (B) tests
whether that dynamic information helps where the level does not — the correct
"second attempt" at making daily MIDAS competitive.

Data
----
  · Target    : Spain CPI `indice_general` (monthly, features_c1.parquet)
  · HF signal : WTI Crude Oil daily close (CL=F via yfinance, 2002-2024).
                WTI is a Brent proxy (corr ~ 0.99); single NaN on 2020-04-20
                (negative-price day) is forward-filled.
  · Monthly covariates : ECB DFR, EPU Europe

Implementation notes (lessons from the first version)
-----------------------------------------------------
  * Target is the h-step delta, not the level → no scale mismatch, stationary.
  * Daily input and monthly covariates are StandardScaler-normalised per origin
    (training window only) → matches the Ridge-correction convention elsewhere.
  * The NLS loss is fully vectorised (one matrix-vector product per evaluation)
    so the 48 origins x 4 horizons x 3 variants run in a few minutes, not hours.
  * Only training rows with a complete daily sequence are kept (avoids the
    zero-variance / NaN columns that broke the naive version).

Reference: Ghysels et al. (2004); ML-MIDAS, Babii et al. (2020) arXiv:2005.14057
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

K_MONTHS = 3                              # months of daily history per origin
MONTHLY_COVS = ["epu_europe_log", "dfr"]
RIDGE_ALPHA = 0.5

WTI_CACHE = RESULTS_DIR / "wti_daily_2002_2024.parquet"


# ── WTI daily data ─────────────────────────────────────────────────────────────

def fetch_wti_daily() -> pd.Series:
    """Daily WTI log-price; cached. Single 2020-04-20 NaN is forward-filled."""
    if WTI_CACHE.exists():
        s = pd.read_parquet(WTI_CACHE)["close"]
    else:
        logger.info("Downloading WTI (CL=F) from yfinance ...")
        import yfinance as yf
        raw = yf.download("CL=F", start="2002-01-01", end="2024-12-31", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0).str.lower()
        else:
            raw.columns = raw.columns.str.lower()
        s = np.log(raw["close"].dropna())
        s.name = "close"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"close": s}).to_parquet(WTI_CACHE)
    s.index = pd.to_datetime(s.index)
    s = s.ffill()   # fix the 2020-04-20 negative-WTI NaN
    logger.info("WTI: %d days (%s -> %s)", len(s), s.index.min().date(), s.index.max().date())
    return s


# ── Beta polynomial ────────────────────────────────────────────────────────────

def beta_weights(log_t1: float, log_t2: float, n: int) -> np.ndarray:
    """Normalised Beta polynomial weights for n daily lag positions."""
    t1 = np.exp(log_t1) + 0.5
    t2 = np.exp(log_t2) + 0.5
    u = np.arange(1, n + 1) / n
    w = u ** (t1 - 1) * (1 - u + 1e-9) ** (t2 - 1)
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


# ── Daily window extraction (no look-ahead) ─────────────────────────────────────

def daily_window(signal: pd.Series, origin: pd.Timestamp, n_days: int) -> np.ndarray | None:
    """Last `n_days` values of `signal` strictly before `origin`."""
    end = origin - pd.DateOffset(months=1)
    start = end - pd.DateOffset(months=K_MONTHS) + pd.DateOffset(days=1)
    seg = signal.loc[(signal.index >= start) & (signal.index <= end + pd.offsets.MonthEnd(0))].values
    if len(seg) >= n_days:
        return seg[-n_days:].astype(np.float64)
    return None


def raw_window_len(signal: pd.Series, origin: pd.Timestamp) -> int:
    end = origin - pd.DateOffset(months=1)
    start = end - pd.DateOffset(months=K_MONTHS) + pd.DateOffset(days=1)
    return int(((signal.index >= start) & (signal.index <= end + pd.offsets.MonthEnd(0))).sum())


def load_monthly() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    return df


# ── One variant's rolling backtest ──────────────────────────────────────────────

def run_variant(df: pd.DataFrame, signal: pd.Series, n_days: int,
                method: str, model_name: str) -> tuple[pd.DataFrame, float]:
    """method: 'beta' (Beta-MIDAS NLS) or 'ridge' (unconstrained free lags)."""
    y = df["indice_general"].values
    dates = df.index
    mon = {v: df[v].ffill().bfill().values.astype(np.float64) for v in MONTHLY_COVS}
    y_train = df["indice_general"].loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(y_train.values[12:] - y_train.values[:-12])))

    origins = pd.date_range(ORIGINS_START, ORIGINS_END, freq="MS")
    records = []

    for origin in tqdm(origins, desc=model_name):
        if origin not in dates:
            continue
        ti = dates.get_loc(origin)
        sn = daily_window(signal, origin, n_days)
        if sn is None:
            continue

        for h in HORIZONS:
            tt = origin + pd.DateOffset(months=h)
            if tt > TEST_END_TS or tt not in dates:
                continue
            thi = dates.get_loc(tt)

            seqs, rows = [], []
            for i in range(2, ti - h + 1):
                if i + h >= len(y):
                    continue
                s = daily_window(signal, dates[i], n_days)
                if s is not None:
                    seqs.append(s)
                    rows.append(i)
            if len(rows) < 20:
                continue
            rows = np.array(rows)

            X_raw = np.stack(seqs)                       # (T, n_days)
            delta = y[rows + h] - y[rows]                # stationary target
            mX = np.column_stack([mon[v][rows] for v in MONTHLY_COVS])

            sc_d = StandardScaler().fit(X_raw)
            sc_m = StandardScaler().fit(mX)
            Xd = sc_d.transform(X_raw)
            mXs = sc_m.transform(mX)
            sp = sc_d.transform(sn.reshape(1, -1))[0]
            mn = sc_m.transform(np.array([[mon[v][ti] for v in MONTHLY_COVS]]))[0]

            if method == "beta":
                def loss(p):
                    w = beta_weights(p[2], p[3], n_days)
                    pred = p[0] + p[1] * (Xd @ w) + mXs @ p[4:]
                    return float(np.sum((delta - pred) ** 2))
                x0 = np.zeros(4 + len(MONTHLY_COVS))
                x0[3] = np.log(2.0)
                try:
                    res = minimize(loss, x0, method="Nelder-Mead",
                                   options={"maxiter": 2000, "xatol": 1e-4,
                                            "fatol": 1e-4, "disp": False})
                    p = res.x
                    w = beta_weights(p[2], p[3], n_days)
                    dpred = p[0] + p[1] * float(sp @ w) + mn @ p[4:]
                except Exception:
                    continue
            else:  # ridge
                try:
                    rg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(
                        np.column_stack([Xd, mXs]), delta)
                    dpred = float(rg.predict(np.concatenate([sp, mn]).reshape(1, -1))[0])
                except Exception:
                    continue

            yhat = y[ti] + dpred
            yt = float(y[thi])
            records.append({
                "origin": origin, "fc_date": tt, "step": h, "horizon": h,
                "model": model_name, "y_true": yt, "y_pred": float(yhat),
                "error": yt - float(yhat), "abs_error": abs(yt - float(yhat)),
            })

    return pd.DataFrame(records), mase_scale


def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    res = {}
    for h in HORIZONS:
        hd = df_preds[df_preds["horizon"] == h]
        if hd.empty:
            continue
        yt, yp = hd["y_true"].values, hd["y_pred"].values
        res[f"h{h}"] = {
            "MAE": round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 66)
    logger.info("DAILY MIDAS  —  Spain CPI  —  Beta polynomial + returns variant")
    logger.info("HF signal : WTI crude daily (CL=F) | K_months=%d | monthly covs=%s",
                K_MONTHS, MONTHLY_COVS)
    logger.info("=" * 66)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wti = fetch_wti_daily()
    wti_ret = wti.diff().dropna()           # daily log-returns
    df = load_monthly()

    lens = [raw_window_len(wti, o) for o in pd.date_range(ORIGINS_START, ORIGINS_END, freq="MS")]
    n_days = int(np.percentile(lens, 25))
    logger.info("N_DAYS = %d (p25 of window lengths %d-%d)", n_days, min(lens), max(lens))

    variants = [
        ("midas_daily_beta",    wti,     "beta"),
        ("midas_daily_returns", wti_ret, "beta"),
        ("midas_daily_ridge",   wti,     "ridge"),
    ]

    all_metrics = {}
    for model_name, sig, method in variants:
        df_preds, scale = run_variant(df, sig, n_days, method, model_name)
        if df_preds.empty:
            logger.warning("[!] %s: no predictions", model_name)
            continue
        m = compute_metrics(df_preds, scale)
        all_metrics[model_name] = m
        df_preds.to_parquet(RESULTS_DIR / f"{model_name}_predictions.parquet", index=False)
        with open(RESULTS_DIR / f"{model_name}_metrics.json", "w") as f:
            json.dump({model_name: m}, f, indent=2)

    # ── Comparison table ──
    refs = {"ARIMA": {"h1": 0.4781, "h3": 0.6716, "h6": 0.9660, "h12": 1.5410}}
    mp = RESULTS_DIR / "midas_adl_spain_metrics.json"
    if mp.exists():
        refs["MIDAS_monthly_proxy"] = {h: v["MAE"] for h, v in
                                       json.load(open(mp))["midas_adl_spain"].items()}

    logger.info("\n%-26s %8s %8s %8s %8s", "Model", "h1 MAE", "h3 MAE", "h6 MAE", "h12 MAE")
    logger.info("-" * 62)
    for name, m in {**refs, **all_metrics}.items():
        row = "%-26s" % name
        for h in HORIZONS:
            v = m.get(f"h{h}")
            v = v["MAE"] if isinstance(v, dict) else v
            row += " %8.4f" % v if v else "        ?"
        logger.info(row)

    logger.info("\nDoes within-month oil momentum (returns) help vs the level?")
    lvl = all_metrics.get("midas_daily_beta", {}).get("h1", {}).get("MAE")
    ret = all_metrics.get("midas_daily_returns", {}).get("h1", {}).get("MAE")
    if lvl and ret:
        logger.info("  h=1  level=%.4f  returns=%.4f  -> returns %+.1f%%",
                    lvl, ret, (ret - lvl) / lvl * 100)
    logger.info("\nSaved metrics + predictions to %s", RESULTS_DIR.name)


if __name__ == "__main__":
    main()
