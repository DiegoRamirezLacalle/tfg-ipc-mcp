"""ARIMAX C1 (institutional / full) - HICP Eurozone with exogenous signals.

Classical counterpart of the Global institutional ARIMAX (script 06). Fills the
European HICP "C1" cell, which previously only had the single-rate SARIMAX+DFR
baseline (script 04). Two exogenous conditions are evaluated:

  C1_inst  (8 institutional signals)
    epu_europe_ma3, brent_ma3, esi_eurozone, eurusd_ma3,
    dfr, dfr_ma3, ttf_ma3, breakeven_5y_lag1

  C1_full  (C1_inst + 5 MCP signals)
    + bce_shock_score, bce_tone_numeric, bce_cumstance,
      gdelt_tone_ma6, signal_available

Design (controlled comparison vs the European C0 baseline):
  - SAME SARIMA order/seasonal_order as the baseline `sarima`/`sarimax`
    (script 04, loaded from arima_europe_metrics.json). The ONLY difference
    vs C0 is the exogenous set, so any delta is attributable to the signals.
  - Rolling expanding-window, origins 2021-01 to 2024-12, h = 1, 3, 6, 12.
  - Future exogenous = real values (oracle assumption), identical to the
    Global institutional ARIMAX (06) and the European DFR SARIMAX (04), so the
    cross-series comparison is apples-to-apples. MCP signals are absent before
    ~2015 and are filled with 0.0 ("signal unavailable", neutral, no backward
    leakage), matching the foundation C1_full script (26).
  - MASE scale fixed on the initial train set (2002-01 to 2020-12).

Output (per condition COND in {C1_inst, C1_full}):
  08_results/rolling_predictions_<COND>_europe.parquet
  08_results/rolling_metrics_<COND>_europe.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR   = ROOT / "08_results"
HORIZONS      = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)

C1_INST = ["epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3",
           "dfr", "dfr_ma3", "ttf_ma3", "breakeven_5y_lag1"]
C1_MCP  = ["bce_shock_score", "bce_tone_numeric", "bce_cumstance",
           "gdelt_tone_ma6", "signal_available"]

# (model_key, exogenous columns) per condition
CONDITIONS = [
    ("arimax_C1_inst_europe", C1_INST),
    ("arimax_C1_full_europe", C1_INST + C1_MCP),
]


def load_orders() -> tuple[tuple, tuple]:
    path = RESULTS_DIR / "arima_europe_metrics.json"
    if path.exists():
        saved = json.loads(path.read_text())
        order          = tuple(saved["order"])
        seasonal_order = tuple(saved["seasonal_order"])
        logger.info(f"  auto_arima order: SARIMA{order}x{seasonal_order}")
    else:
        order          = (2, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        logger.info(f"  Fallback order:   SARIMA{order}x{seasonal_order}")
    return order, seasonal_order


def load_data() -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_europe.parquet")
    df.index = pd.to_datetime(df.index)
    df.index.freq = "MS"
    y = df["hicp_index"]
    # Institutional signals span the full sample -> ffill/bfill only fixes edges.
    # MCP signals are absent pre-2015 -> fill with 0.0 (neutral "unavailable").
    X = df[C1_INST + C1_MCP].copy()
    X[C1_INST] = X[C1_INST].ffill().bfill()
    X[C1_MCP] = X[C1_MCP].fillna(0.0)
    return y, X


def run_rolling(
    y: pd.Series, X: pd.DataFrame, exog_cols: list[str],
    order: tuple, seasonal_order: tuple, model_key: str,
) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))
    logger.info(f"  MASE scale: {mase_scale:.4f}")

    records = []
    for origin in tqdm(origins, desc=model_key):
        y_train = y.loc[:origin]
        x_train = X.loc[:origin, exog_cols].values

        try:
            res = SM_SARIMAX(
                y_train, exog=x_train,
                order=order, seasonal_order=seasonal_order, trend="n",
            ).fit(disp=False)
        except Exception as e:
            logger.warning(f"\n[!] {origin.date()}: {e}")
            continue

        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS:
                continue
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue

            x_future = X.reindex(fc_dates)[exog_cols].values
            y_pred = res.get_forecast(steps=h, exog=x_future).predicted_mean.values
            y_true = y_actual.values

            for i, (date, real, pred) in enumerate(zip(fc_dates, y_true, y_pred), 1):
                records.append({
                    "origin":    origin,
                    "fc_date":   date,
                    "step":      i,
                    "horizon":   h,
                    "model":     model_key,
                    "y_true":    real,
                    "y_pred":    pred,
                    "error":     real - pred,
                    "abs_error": abs(real - pred),
                })

    return pd.DataFrame(records), mase_scale


def compute_metrics(df_preds: pd.DataFrame, mase_scale: float, model_key: str) -> dict:
    results = {model_key: {}}
    m_df = df_preds[df_preds["model"] == model_key]
    for h in HORIZONS:
        h_df = m_df[m_df["horizon"] == h]
        if h_df.empty:
            continue
        yt = h_df["y_true"].values
        yp = h_df["y_pred"].values
        results[model_key][f"h{h}"] = {
            "MAE":     round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE":    round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE":    round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(h_df["origin"].unique())),
        }
    return results


def _load_c0_baseline() -> dict:
    """European C0 (no exog) = `sarima` from the baseline rolling backtest."""
    p = RESULTS_DIR / "rolling_metrics_europe.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("sarima", {})


def main():
    logger.info("=" * 60)
    logger.info("ARIMAX C1 (institutional / full) - HICP Eurozone")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    order, seasonal_order = load_orders()
    y, X = load_data()
    logger.info(f"\nHICP: {len(y)} obs  ({y.index.min().date()} - {y.index.max().date()})")
    logger.info(f"Exog matrix: {X.shape[1]} cols, residual NaN={int(X.isna().sum().sum())}")

    c0 = _load_c0_baseline()

    for model_key, exog_cols in CONDITIONS:
        logger.info("\n" + "=" * 60)
        logger.info(f"{model_key}  ({len(exog_cols)} exogenous)")
        logger.info(f"  {exog_cols}")
        logger.info("=" * 60)

        df_preds, mase_scale = run_rolling(
            y, X, exog_cols, order, seasonal_order, model_key
        )
        if df_preds.empty:
            logger.warning(f"[!] No predictions for {model_key}; skipping.")
            continue

        metrics = compute_metrics(df_preds, mase_scale, model_key)

        logger.info(f"\n  {'h':>4} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
        logger.info(f"  {'-'*38}")
        for h in HORIZONS:
            key = f"h{h}"
            if key in metrics[model_key]:
                m = metrics[model_key][key]
                logger.info(f"  {h:>4} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                            f"{m['MASE']:>8.4f} {m['n_evals']:>5d}")

        if c0:
            logger.info("\n  Delta MAE vs C0 (sarima, no exog) — positive = C1 improvement:")
            logger.info(f"  {'h':>4} {'C0 sarima':>12} {'C1':>12} {'Delta%':>10}")
            logger.info(f"  {'-'*42}")
            for h in HORIZONS:
                key = f"h{h}"
                if key in metrics[model_key] and key in c0:
                    c0_mae = c0[key]["MAE"]
                    c1_mae = metrics[model_key][key]["MAE"]
                    pct    = (c0_mae - c1_mae) / c0_mae * 100
                    mark   = " <-- improvement" if pct > 0 else ""
                    logger.info(f"  {h:>4} {c0_mae:>12.4f} {c1_mae:>12.4f} {pct:>+9.1f}%{mark}")

        cond = model_key.replace("arimax_", "").replace("_europe", "")  # C1_inst / C1_full
        preds_path   = RESULTS_DIR / f"rolling_predictions_{cond}_europe.parquet"
        metrics_path = RESULTS_DIR / f"rolling_metrics_{cond}_europe.json"
        df_preds.to_parquet(preds_path, index=False)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\n  Saved: {preds_path.name}")
        logger.info(f"  Saved: {metrics_path.name}")

    logger.info("\n" + "=" * 60)
    logger.info("ARIMAX C1 EUROPE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
