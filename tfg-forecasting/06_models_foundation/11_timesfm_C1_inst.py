"""
11_timesfm_C1_inst.py — TimesFM C1 institutional: Ridge XReg with EPU Europe

Architecture: TimesFM base (full IPC context 2002+) + Ridge correction.
Ridge fitted over full window 2002:origin (EPU Europe available since 1987).
Ridge covariates: epu_europe_ma3, epu_europe_log, epu_europe_lag1
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
MODEL_NAME = "timesfm_C1_inst"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
RIDGE_ALPHA = 1.0

# EPU Europe available from 2002 complete — no clipping needed
XREG_COVS = ["epu_europe_ma3", "epu_europe_log", "epu_europe_lag1"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    return df


def load_model():
    import timesfm
    logger.info("[timesfm] Loading google/timesfm-2.5-200m-pytorch ...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tfm.compile(forecast_config=timesfm.ForecastConfig(
        max_context=512, max_horizon=MAX_H, per_core_batch_size=1,
    ))
    logger.info("[timesfm] Loaded")
    return tfm


def compute_xreg_correction(df: pd.DataFrame, origin: pd.Timestamp) -> float:
    """Ridge on monthly IPC change, window 2002:origin with EPU Europe."""
    window = df.loc[:origin].copy()
    if len(window) < 13:
        return 0.0
    ipc_mom = window["indice_general"].diff(1)
    valid = ~ipc_mom.isna()
    X = window.loc[valid, XREG_COVS].fillna(0.0).values.astype(np.float64)
    y_diff = ipc_mom[valid].values.astype(np.float64)
    reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    reg.fit(X, y_diff)
    current = df.loc[origin:origin, XREG_COVS].fillna(0.0).values.astype(np.float64)
    neutral = np.zeros_like(current)
    return float(reg.predict(current)[0] - reg.predict(neutral)[0])


def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["indice_general"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
        # Full IPC context from 2002 (do not clip to 2015+)
        context = y.loc[:origin].values.astype(np.float32)
        try:
            point_out, _ = model.forecast(horizon=MAX_H, inputs=[context])
            base_pred = np.array(point_out[0])
        except Exception as e:
            logger.warning(f"\n[!] Base error {origin.date()}: {e}")
            continue
        correction = compute_xreg_correction(df, origin)
        full_pred = base_pred + correction
        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS:
                continue
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            ya = y.reindex(fc_dates)
            if ya.isna().any():
                continue
            for i, (d, r, p) in enumerate(zip(fc_dates, ya.values, full_pred[:h]), 1):
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
    logger.info("=" * 60)
    logger.info(f"BACKTESTING — {MODEL_NAME}")
    logger.info(f"XReg covs: {XREG_COVS}")
    logger.info("=" * 60)
    df = load_data()
    logger.info(f"Data: {len(df)} obs")
    model = load_model()
    df_preds, mase_scale = run_rolling(df, model)
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
    c0p = RESULTS_DIR / "timesfm_C0_metrics.json"
    if c0p.exists():
        with open(c0p) as f:
            c0 = json.load(f).get("timesfm_C0", {})
        logger.info(f"\n--- C0 vs {MODEL_NAME} ---")
        for h in HORIZONS:
            c0m = c0.get(f"h{h}", {}).get("MAE")
            c1m = metrics.get(f"h{h}", {}).get("MAE")
            if c0m and c1m:
                logger.info(f"  h={h}: C0={c0m:.4f}  C1={c1m:.4f}  "
                            f"delta={c1m - c0m:+.4f} ({(c1m - c0m) / c0m * 100:+.1f}%)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    with open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"\nSaved: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
