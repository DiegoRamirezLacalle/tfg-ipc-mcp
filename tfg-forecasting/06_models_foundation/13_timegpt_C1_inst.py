"""
13_timegpt_C1_inst.py — TimeGPT C1 institutional: EPU Europe

Covariates (3): epu_europe_ma3 (0.737), epu_europe_log (0.701), epu_europe_lag1 (0.682)
Complete data from 2002, no NaN. Full IPC context 2002+.
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
MODEL_NAME = "timegpt_C1_inst"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SERIES_ID = "ipc_spain"

EXOG_COLS = ["epu_europe_ma3", "epu_europe_log", "epu_europe_lag1"]


def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError("NIXTLA_API_KEY not configured.")
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


def load_data() -> tuple[pd.Series, pd.DataFrame]:
    ipc_df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = ipc_df["indice_general"]
    y.index = pd.to_datetime(y.index)
    y.index.freq = "MS"
    c1 = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    c1["date"] = pd.to_datetime(c1["date"])
    c1 = c1.set_index("date")
    c1.index.freq = "MS"
    return y, c1


def build_nixtla_df(
    y: pd.Series,
    exog: pd.DataFrame,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_y = y.loc[:origin]
    hist_df = pd.DataFrame({
        "unique_id": SERIES_ID, "ds": context_y.index, "y": context_y.values,
    })
    for col in EXOG_COLS:
        col_vals = exog.loc[:origin, col].reindex(context_y.index)
        hist_df[col] = col_vals.values
    last_row = exog.loc[:origin, EXOG_COLS].iloc[-1]
    fc_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
    )
    future_df = pd.DataFrame({"unique_id": SERIES_ID, "ds": fc_dates})
    for col in EXOG_COLS:
        future_df[col] = float(last_row[col])
    return hist_df, future_df


def run_rolling(
    y: pd.Series,
    exog: pd.DataFrame,
    client,
    test_run: bool = False,
) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run:
        origins = origins[:5]
        logger.info(f"[test-run] {len(origins)} origins")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in tqdm(origins, desc=f"{MODEL_NAME}"):
        hist_df, future_df = build_nixtla_df(y, exog, origin)
        try:
            fc = client.forecast(
                df=hist_df, X_df=future_df, h=MAX_H, freq="MS",
                time_col="ds", target_col="y", id_col="unique_id",
            )
            pred_values = fc.sort_values("ds")["TimeGPT"].values
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
            "MAE": round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()
    n = 5 if args.test_run else 48
    logger.info("=" * 60)
    logger.info(f"BACKTESTING — {MODEL_NAME} ({'TEST' if args.test_run else 'FULL'})")
    logger.info(f"Covariates: {EXOG_COLS}")
    logger.info(f"Estimated cost: {n} API calls")
    logger.info("=" * 60)
    y, exog = load_data()
    client = get_client()
    df_preds, mase_scale = run_rolling(y, exog, client, test_run=args.test_run)
    if df_preds.empty:
        logger.warning("[!] No predictions generated.")
        return
    metrics = compute_metrics(df_preds, mase_scale)
    logger.info(f"\n{'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]
            logger.info(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                        f"{m['MASE']:8.4f} {m['n_evals']:5d}")
    c0p = RESULTS_DIR / "timegpt_C0_metrics.json"
    if c0p.exists():
        with open(c0p) as f:
            c0 = json.load(f).get("timegpt_C0", {})
        logger.info(f"\n--- C0 vs {MODEL_NAME} ---")
        for h in HORIZONS:
            c0m = c0.get(f"h{h}", {}).get("MAE")
            c1m = metrics.get(f"h{h}", {}).get("MAE")
            if c0m and c1m:
                logger.info(f"  h={h}: C0={c0m:.4f}  C1={c1m:.4f}  "
                            f"delta={c1m - c0m:+.4f} ({(c1m - c0m) / c0m * 100:+.1f}%)")
    if args.test_run:
        logger.info("\n[test-run] OK. Run without --test-run for full backtesting.")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    with open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"\nSaved: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
