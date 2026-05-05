"""
22_chronos2_C1_mcp_europe.py - Chronos-2 C1_mcp HICP Eurozone

Exclusively semantic BCE/GDELT covariates (pure MCP experiment):
  bce_shock_score   (corr C0 residuals = +0.36) ***
  bce_tone_numeric  (corr C0 residuals = +0.26) ***
  bce_cumstance     (corr C0 residuals = -0.58) ***
  gdelt_tone_ma6    (corr C0 residuals = +0.46) ***
  signal_available  (availability indicator)

NaN before 2015-01 filled with 0 (neutral).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
MODEL_NAME = "chronos2_C1_mcp_europe"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
Q_IDX = {"p10": 2, "p50": 10, "p90": 18}

EXOG_COLS = ["bce_shock_score", "bce_tone_numeric", "bce_cumstance",
             "gdelt_tone_ma6", "signal_available"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_europe.parquet")
    df.index = pd.to_datetime(df.index)
    df.index.freq = "MS"
    return df


def load_model():
    from chronos import Chronos2Pipeline
    logger.info("[chronos2] Loading amazon/chronos-2 ...")
    p = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    logger.info("[chronos2] Loaded")
    return p


def prepare_input(df: pd.DataFrame, origin: pd.Timestamp) -> dict:
    ctx = df.loc[:origin].copy()
    tgt = ctx["hicp_index"].values.astype(np.float64)
    past = {c: ctx[c].fillna(0.0).values.astype(np.float64) for c in EXOG_COLS if c in ctx.columns}
    # Future: signal_available=1, rest neutral (0)
    future = {}
    for c in EXOG_COLS:
        if c == "signal_available":
            future[c] = np.ones(MAX_H, dtype=np.float64)
        else:
            future[c] = np.zeros(MAX_H, dtype=np.float64)
    return {"target": tgt, "past_covariates": past, "future_covariates": future}


def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["hicp_index"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
        inp = prepare_input(df, origin)
        try:
            preds = model.predict([inp], prediction_length=MAX_H)
            q = preds[0].numpy()[0]
        except Exception as e:
            logger.warning(f"\n[!] {origin.date()}: {e}")
            continue
        p50 = q[Q_IDX["p50"]]
        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS:
                continue
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            ya = y.reindex(fc_dates)
            if ya.isna().any():
                continue
            for i, (d, r) in enumerate(zip(fc_dates, ya.values), 1):
                records.append({
                    "origin": origin, "fc_date": d, "step": i, "horizon": h,
                    "model": MODEL_NAME, "y_true": float(r), "y_pred": float(p50[i - 1]),
                    "error": float(r - p50[i - 1]), "abs_error": float(abs(r - p50[i - 1])),
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
    logger.info("=" * 60)
    logger.info(f"BACKTESTING - {MODEL_NAME}")
    logger.info(f"MCP covariates: {EXOG_COLS}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    model = load_model()

    df_preds, mase_scale = run_rolling(df, model)
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

    c0p = RESULTS_DIR / "chronos2_C0_europe_metrics.json"
    if c0p.exists():
        with open(c0p) as f:
            c0 = json.load(f).get("chronos2_C0_europe", {})
        logger.info(f"\n--- C0 vs {MODEL_NAME} ---")
        for h in HORIZONS:
            c0m = c0.get(f"h{h}", {}).get("MAE")
            c1m = metrics.get(f"h{h}", {}).get("MAE")
            if c0m and c1m:
                logger.info(f"  h={h}: C0={c0m:.4f}  C1_mcp={c1m:.4f}  "
                            f"delta={c1m - c0m:+.4f} ({(c1m - c0m) / c0m * 100:+.1f}%)")

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    with open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"\nSaved: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
