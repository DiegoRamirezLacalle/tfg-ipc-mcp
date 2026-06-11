"""
27_timegpt_C1_inst_europe.py - TimeGPT C1_institutional HICP Eurozone

TimeGPT native with institutional covariates via X_df.
Future horizon (h steps): signals propagated with last known value.

Covariates: epu_europe_ma3, brent_ma3, esi_eurozone, eurusd_ma3,
            dfr, dfr_ma3, ttf_ma3, breakeven_5y_lag1

Cost control: --test-run (5 origins) / default full.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
MAX_H = 12
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "timegpt_C1_inst_europe"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SERIES_ID = "HICP_EUROPE"

XREG_COVS = ["epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3",
             "dfr", "dfr_ma3", "ttf_ma3", "breakeven_5y_lag1"]


def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError(
            "NIXTLA_API_KEY not configured. "
            "Edit the .env file at the monorepo root."
        )
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_europe.parquet")
    df.index = pd.to_datetime(df.index)
    df.index.freq = "MS"
    return df


def build_nixtla_dfs(df: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = df.loc[:origin]
    last_vals = hist[XREG_COVS].iloc[-1].fillna(0.0)

    # Historical target + exogenous (required by Nixtla API)
    hist_reset = hist[XREG_COVS].fillna(0.0).reset_index().rename(columns={"date": "ds"})
    df_tgt = pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": hist.index,
        "y": hist["hicp_index"].values,
    })
    for c in XREG_COVS:
        df_tgt[c] = hist_reset[c].values

    # X_df: future rows - forward-fill last known value
    future_idx = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
    )
    rows_fut = pd.DataFrame({"ds": future_idx, "unique_id": SERIES_ID})
    for c in XREG_COVS:
        rows_fut[c] = float(last_vals[c])

    X_df = rows_fut[["unique_id", "ds"] + XREG_COVS]
    return df_tgt, X_df


def run_rolling(
    df: pd.DataFrame,
    client,
    test_run: bool = False,
) -> tuple[pd.DataFrame, float]:
    y = df["hicp_index"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run:
        origins = origins[:5]
        logger.info(f"[test-run] Running with {len(origins)} origins")

    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []

    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
        df_tgt, X_df = build_nixtla_dfs(df, origin)
        try:
            fc = client.forecast(
                df=df_tgt, X_df=X_df, h=MAX_H, freq="MS",
                time_col="ds", target_col="y", id_col="unique_id",
            )
            fc = fc.sort_values("ds").reset_index(drop=True)
            pred_values = fc["TimeGPT"].values
        except Exception as e:
            logger.warning(f"\n[!] {origin.date()}: {e}")
            continue

        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS:
                continue
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            ya = y.reindex(fc_dates)
            if ya.isna().any():
                continue
            for i, (d, r, p) in enumerate(zip(fc_dates, ya.values, pred_values[:h]), 1):
                records.append({
                    "origin": origin, "fc_date": d, "step": i, "horizon": h,
                    "model": MODEL_NAME, "y_true": float(r), "y_pred": float(p),
                    "error": float(r - p), "abs_error": float(abs(r - p)),
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
            "MAE":     round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE":    round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE":    round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def main():
    parser = argparse.ArgumentParser(
        description="TimeGPT C1 institutional europe rolling backtesting"
    )
    parser.add_argument("--test-run", action="store_true",
                        help="Run only 5 origins to verify cost/functionality")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"BACKTESTING - {MODEL_NAME}")
    logger.info(f"Covariates: {XREG_COVS}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    client = get_client()
    logger.info("[timegpt] Nixtla client initialized")

    df_preds, mase_scale = run_rolling(df, client, test_run=args.test_run)
    if df_preds.empty:
        logger.warning("[!] No predictions generated.")
        return

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info(f"\n{'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    logger.info("-" * 45)
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]
            logger.info(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                        f"{m['MASE']:8.4f} {m['n_evals']:5d}")

    c0p = RESULTS_DIR / "timegpt_C0_europe_metrics.json"
    if c0p.exists():
        with open(c0p) as f:
            c0 = json.load(f).get("timegpt_C0_europe", {})
        logger.info(f"\n--- C0 vs {MODEL_NAME} ---")
        for h in HORIZONS:
            c0m = c0.get(f"h{h}", {}).get("MAE")
            c1m = metrics.get(f"h{h}", {}).get("MAE")
            if c0m and c1m:
                logger.info(f"  h={h}: C0={c0m:.4f}  C1={c1m:.4f}  "
                            f"delta={c1m - c0m:+.4f} ({(c1m - c0m) / c0m * 100:+.1f}%)")

    if args.test_run:
        logger.info("\n[test-run] Run without --test-run for full backtesting.")
        return

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    with open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"\nSaved: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
