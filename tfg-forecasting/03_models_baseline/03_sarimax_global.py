"""ARIMAX with Federal Funds Rate (FEDFUNDS) as exogenous variable - Global CPI.

Exogenous variable: FEDFUNDS (Federal Funds Rate, % annual)
Source: FRED (Federal Reserve Bank of St. Louis)
URL:    https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS

Economic justification:
  The Fed reference rate is the dominant global monetary policy instrument.
  A rate increase reduces aggregate demand and, with a lag, inflation.
  Exact analogy with ECB DFR in the Spain model.

Note on look-ahead bias:
  Fed decisions are published on the same day as the FOMC meeting.
  Passing real FEDFUNDS values as future exogenous in static evaluation
  (validation oracle) is correct; rolling backtesting uses values known
  at each forecast origin.

Data saved at:
  data/raw/fedfunds_raw.csv
  data/processed/fedfunds_monthly.parquet

Output:
  08_results/arimax_global_summary.txt
  08_results/arimax_global_metrics.json
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
import requests
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.logger import get_logger
from shared.metrics import mae, rmse, mase

logger = get_logger(__name__)

RESULTS_DIR  = ROOT / "08_results"
RAW_DIR      = ROOT / "data" / "raw"
PROC_DIR     = ROOT / "data" / "processed"
FEDFUNDS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
EXOG_COL     = "fedfunds"


def download_fedfunds() -> pd.Series:
    """Download FEDFUNDS from FRED (public CSV, no API key). Returns monthly series indexed by MS from 2001-01."""
    raw_path = RAW_DIR / "fedfunds_raw.csv"

    logger.info("  Downloading FEDFUNDS from FRED...")
    r = requests.get(FEDFUNDS_URL, timeout=60)
    r.raise_for_status()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(r.text, encoding="utf-8")
    logger.info(f"  Raw saved: {raw_path}")

    df = pd.read_csv(io.StringIO(r.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "FEDFUNDS": EXOG_COL})
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthBegin(0)
    df = df.set_index("date").sort_index()
    df.index.freq = "MS"

    series = df[EXOG_COL].astype(float)
    return series


def prepare_fedfunds(series: pd.Series, date_start="2001-01-01",
                     date_end="2025-01-01") -> pd.Series:
    """Filter to project date range and save to processed/. No shift applied: Fed rate is known on announcement day."""
    s = series.loc[date_start:date_end]
    # Forward-fill possible gaps (rate does not change between meetings)
    target_idx = pd.date_range(date_start, date_end, freq="MS")
    s = s.reindex(target_idx).ffill().dropna()
    s.index.freq = "MS"

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "fedfunds_monthly.parquet"
    s.to_frame().to_parquet(out_path)
    logger.info(f"  Processed saved: {out_path}")

    return s


def load_or_download_fedfunds() -> pd.Series:
    proc_path = PROC_DIR / "fedfunds_monthly.parquet"
    if proc_path.exists():
        logger.info(f"  Loading FEDFUNDS from cache: {proc_path}")
        s = pd.read_parquet(proc_path)[EXOG_COL]
        s.index.freq = "MS"
        return s
    raw = download_fedfunds()
    return prepare_fedfunds(raw)


def load_data():
    # Target series
    df_y = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df_y["cpi_global_rate"]
    y.index.freq = "MS"

    # Exogenous
    fedfunds = load_or_download_fedfunds()
    logger.info(f"  FEDFUNDS: {fedfunds.index.min().date()} -> {fedfunds.index.max().date()} "
                f"({len(fedfunds)} obs)  min={fedfunds.min():.2f}%  max={fedfunds.max():.2f}%")

    # Align to y index
    fedfunds = fedfunds.reindex(y.index).ffill()

    df = pd.DataFrame({"cpi_global_rate": y, EXOG_COL: fedfunds}).dropna()

    # Splits
    train_mask = df.index <= DATE_TRAIN_END
    val_mask   = (df.index > DATE_TRAIN_END) & (df.index <= DATE_VAL_END)

    y_train = df.loc[train_mask, "cpi_global_rate"]
    y_val   = df.loc[val_mask,   "cpi_global_rate"]
    X_train = df.loc[train_mask, [EXOG_COL]]
    X_val   = df.loc[val_mask,   [EXOG_COL]]

    return y_train, y_val, X_train, X_val


def fit_arimax(y_train: pd.Series, X_train: pd.DataFrame):
    """auto_arima with exogenous. Same ranges as script 01 (p/q max=4). d=1, D=0, seasonal=False."""
    model = pm.auto_arima(
        y_train,
        exogenous=X_train,
        start_p=0, max_p=4,
        start_q=0, max_q=4,
        d=1,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )
    return model


def diagnose_residuals(model, name="arimax_global"):
    resid = model.resid()
    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)

    logger.info(f"\n--- Residual diagnostics ({name}) ---")
    logger.info(f"  Mean:  {resid.mean():.6f}")
    logger.info(f"  Std:   {resid.std():.4f}")
    logger.info(f"  Min:   {resid.min():.4f}   Max: {resid.max():.4f}")
    logger.info(f"  Ljung-Box (H0: no autocorrelation):")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "RESIDUAL AUTOCORRELATION"
        logger.info(f"    Lag {lag:2d}: stat={row['lb_stat']:7.3f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return resid, lb


def forecast_and_evaluate(model, y_train, y_val, X_val):
    n_val = len(y_val)
    fc, ci = model.predict(
        n_periods=n_val,
        exogenous=X_val,
        return_conf_int=True,
        alpha=0.05,
    )
    fc_series = pd.Series(fc, index=y_val.index, name="forecast")
    metrics = {
        "MAE":  round(float(mae(y_val.values, fc)), 4),
        "RMSE": round(float(rmse(y_val.values, fc)), 4),
        "MASE": round(float(mase(y_val.values, fc, y_train.values, m=12)), 4),
    }
    return fc_series, ci, metrics


def save_results(model, metrics, resid, lb):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "arimax_global_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    logger.info(f"\nSummary saved: {summary_path}")

    lb_dict = {
        f"lag_{lag}": {"stat": round(row["lb_stat"], 4), "pvalue": round(row["lb_pvalue"], 4)}
        for lag, row in lb.iterrows()
    }
    out = {
        "model":         "arimax_global",
        "exog":          EXOG_COL,
        "order":         list(model.order),
        "aic":           round(float(model.aic()), 4),
        "bic":           round(float(model.bic()), 4),
        "n_train":       int(model.nobs_),
        "residuals": {
            "mean":      round(float(resid.mean()), 6),
            "std":       round(float(resid.std()),  4),
            "ljung_box": lb_dict,
        },
        "metrics_val":   metrics,
    }
    metrics_path = RESULTS_DIR / "arimax_global_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")
    return out


def load_prev_metrics():
    prev = {}
    for name in ["arima_global", "arima111_global"]:
        path = RESULTS_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                prev[name] = json.load(f)
    return prev


def main():
    logger.info("=" * 60)
    logger.info(f"ARIMAX GLOBAL - Baseline with exogenous: {EXOG_COL.upper()}")
    logger.info("=" * 60)

    logger.info("\nLoading data...")
    y_train, y_val, X_train, X_val = load_data()

    logger.info(f"\nTrain: {y_train.index.min().date()} -> {y_train.index.max().date()} ({len(y_train)} obs)")
    logger.info(f"Val:   {y_val.index.min().date()} -> {y_val.index.max().date()} ({len(y_val)} obs)")
    logger.info(f"FEDFUNDS train: min={X_train[EXOG_COL].min():.2f}%  max={X_train[EXOG_COL].max():.2f}%")
    logger.info(f"FEDFUNDS val:   min={X_val[EXOG_COL].min():.2f}%  max={X_val[EXOG_COL].max():.2f}%")

    logger.info("\n--- auto_arima search with exogenous FEDFUNDS ---")
    model = fit_arimax(y_train, X_train)

    order = model.order
    logger.info(f"\nSelected model: ARIMAX{order} + {EXOG_COL.upper()}")
    logger.info(f"AIC: {model.aic():.4f}  |  BIC: {model.bic():.4f}")
    logger.info(model.summary())

    resid, lb = diagnose_residuals(model, f"ARIMAX{order}")

    logger.info(f"\n--- Forecast on validation ({len(y_val)} months) ---")
    fc, ci, metrics = forecast_and_evaluate(model, y_train, y_val, X_val)

    logger.info(f"\nValidation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\n{'Date':>12} {'Actual':>10} {'Pred':>10} {'Error':>10} {'FedFunds':>10}")
    logger.info("-" * 58)
    for date, real, pred, ff in zip(y_val.index, y_val.values, fc.values, X_val[EXOG_COL].values):
        flag = " <--" if abs(real - pred) > 1.5 else ""
        logger.info(f"{str(date.date()):>12} {real:10.4f} {pred:10.4f} {real-pred:10.4f} {ff:10.2f}{flag}")

    result_dict = save_results(model, metrics, resid, lb)

    prev = load_prev_metrics()
    all_m = {**prev, "arimax_global": result_dict}

    if prev:
        logger.info(f"\n{'=' * 70}")
        logger.info("ARIMA(3,1,0) vs ARIMA(1,1,1) vs ARIMAX+FEDFUNDS comparison")
        logger.info(f"{'=' * 70}")
        names = [k for k in ["arima_global", "arima111_global", "arimax_global"] if k in all_m]
        labels = {"arima_global": "ARIMA(3,1,0)", "arima111_global": "ARIMA(1,1,1)",
                  "arimax_global": f"ARIMAX+{EXOG_COL.upper()}"}
        header = f"{'Metric':<8}" + "".join(f" {labels[n]:>16}" for n in names)
        logger.info(header)
        logger.info("-" * (8 + 17 * len(names)))
        for m_name in ["MAE", "RMSE", "MASE"]:
            row = f"{m_name:<8}"
            for n in names:
                row += f" {all_m[n]['metrics_val'][m_name]:>16.4f}"
            logger.info(row)
        logger.info(f"\n{'AIC':<8}" + "".join(f" {all_m[n]['aic']:>16.4f}" for n in names))
        logger.info(f"{'BIC':<8}" + "".join(f" {all_m[n]['bic']:>16.4f}" for n in names))

        if "arima_global" in all_m:
            base_mae = all_m["arima_global"]["metrics_val"]["MAE"]
            ax_mae   = metrics["MAE"]
            improvement = (base_mae - ax_mae) / base_mae * 100
            logger.info(f"FEDFUNDS benefit over ARIMA(3,1,0): {improvement:+.1f}% in val MAE")
            if improvement > 0:
                logger.info("=> Monetary exogenous improves forecast over the validation period.")
            else:
                logger.info("=> Exogenous does not improve forecast on static validation.")
                logger.info("   Note: FEDFUNDS was near 0% during 2021 (val period).")
                logger.info("   Its effect will be more visible in rolling backtesting (2022-2024).")

    logger.info(f"\n{'=' * 60}")
    logger.info("ARIMAX GLOBAL SUMMARY")
    logger.info(f"  Model:    ARIMAX{order} + {EXOG_COL.upper()}")
    logger.info(f"  AIC: {result_dict['aic']}  BIC: {result_dict['bic']}")
    logger.info(f"  MAE val: {metrics['MAE']}  RMSE: {metrics['RMSE']}  MASE: {metrics['MASE']}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
