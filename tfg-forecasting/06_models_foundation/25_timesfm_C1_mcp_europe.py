"""
25_timesfm_C1_mcp_europe.py — TimesFM C1_mcp HICP Eurozone

Architecture: TimesFM base + Ridge correction with MCP/BCE signals only.
Pure MCP experiment: BCE Eurozone growth vs. HICP Europe.

MCP covariates (all with corr > 0.25 vs C0 residuals):
  bce_shock_score, bce_tone_numeric, bce_cumstance, gdelt_tone_ma6, signal_available
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
MODEL_NAME = "timesfm_C1_mcp_europe"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
RIDGE_ALPHA = 1.0

XREG_COVS = ["bce_shock_score", "bce_tone_numeric", "bce_cumstance",
             "gdelt_tone_ma6", "signal_available"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_europe.parquet")
    df.index = pd.to_datetime(df.index)
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
    window = df.loc[:origin].copy()
    if len(window) < 13:
        return 0.0
    rate_mom = window["hicp_index"].diff(1)
    valid = ~rate_mom.isna()
    X = window.loc[valid, XREG_COVS].fillna(0.0).values.astype(np.float64)
    y = rate_mom[valid].values.astype(np.float64)
    reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    reg.fit(X, y)
    current = df.loc[origin:origin, XREG_COVS].fillna(0.0).values.astype(np.float64)
    neutral = np.zeros_like(current)
    return float(reg.predict(current)[0] - reg.predict(neutral)[0])


def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["hicp_index"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
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
            "MAE":     round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE":    round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE":    round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def main():
    logger.info("=" * 60)
    logger.info(f"BACKTESTING — {MODEL_NAME}")
    logger.info(f"Ridge MCP covariates: {XREG_COVS}")
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

    for ref_name in ["timesfm_C0_europe", "timesfm_C1_inst_europe"]:
        p = RESULTS_DIR / f"{ref_name}_metrics.json"
        if p.exists():
            with open(p) as f:
                ref = json.load(f).get(ref_name, {})
            logger.info(f"\n--- {ref_name} vs {MODEL_NAME} ---")
            for h in HORIZONS:
                rm = ref.get(f"h{h}", {}).get("MAE")
                cm = metrics.get(f"h{h}", {}).get("MAE")
                if rm and cm:
                    logger.info(f"  h={h}: ref={rm:.4f}  mcp={cm:.4f}  "
                                f"delta={cm - rm:+.4f} ({(cm - rm) / rm * 100:+.1f}%)")

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    with open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"\nSaved: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
