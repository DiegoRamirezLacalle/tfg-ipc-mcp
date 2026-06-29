"""
34_chronos2_C1_fwd_spain.py - Chronos-2 C1 Spain (IPC), HONEST forward covariates

Forward-path generalization of 09_chronos2_C1_inst.py (Spain). Identical in every
respect (same target `indice_general`, same EPU-Europe covariates, same origins,
same MASE scale, same native Chronos-2 covariate mechanism) EXCEPT the future
covariate path uses the shared ExogPolicy.FORWARD_PATH (damped RW-with-drift,
data <= origin) instead of the flat last-value carry-forward.

The comparison vs script 09 (flat-hold `chronos2_C1_inst`) therefore isolates the
value of the forward path for the Spain index target. Does NOT overwrite the
flat-hold C1 result; writes a new, clearly-named variant `chronos2_C1_fwd_spain`.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.exog_policies import ExogPolicy, build_future_covariates
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
MAX_H = 12
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "chronos2_C1_fwd_spain"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
Q_IDX = {"p10": 2, "p50": 10, "p90": 18}

EXOG_COLS = ["epu_europe_ma3", "epu_europe_log", "epu_europe_lag1"]

# Forward-path hyperparameters (fixed a priori; not tuned on the test window).
DRIFT_WINDOW = 12
PHI = 0.85

# Flat-hold counterpart for the in-log delta print.
FLAT_HOLD_KEY = "chronos2_C1_inst"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    return df


def load_model():
    from chronos import Chronos2Pipeline
    logger.info("[chronos2] Loading amazon/chronos-2 ...")
    p = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    logger.info("[chronos2] Loaded")
    return p


def prepare_input(df: pd.DataFrame, origin: pd.Timestamp, h: int) -> dict:
    ctx = df.loc[:origin]
    target = ctx["indice_general"].values.astype(np.float64)
    past = {c: ctx[c].values.astype(np.float64) for c in EXOG_COLS if c in ctx.columns}
    future = build_future_covariates(
        df, EXOG_COLS, origin, h, ExogPolicy.FORWARD_PATH,
        drift_window=DRIFT_WINDOW, phi=PHI,
    )
    return {"target": target, "past_covariates": past, "future_covariates": future}


def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["indice_general"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
        inp = prepare_input(df, origin, MAX_H)
        try:
            preds = model.predict([inp], prediction_length=MAX_H)
            q = preds[0].numpy()[0]
        except Exception as e:
            logger.warning(f"\n[!] {origin.date()}: {e}")
            continue
        p50, p10, p90 = q[Q_IDX["p50"]], q[Q_IDX["p10"]], q[Q_IDX["p90"]]
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
                    "y_pred_p10": float(p10[i - 1]), "y_pred_p90": float(p90[i - 1]),
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
        p10, p90 = hd["y_pred_p10"].values, hd["y_pred_p90"].values
        cov = float(np.mean((yt >= p10) & (yt <= p90)))
        res[f"h{h}"] = {
            "MAE": round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "coverage_80": round(cov, 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def _delta_log(metrics: dict, mfile: str, mkey: str, tag: str) -> None:
    p = RESULTS_DIR / mfile
    if not p.exists():
        logger.warning("[!] %s metrics missing (%s) - skipping", tag, mfile)
        return
    ref = json.loads(p.read_text(encoding="utf-8")).get(mkey, {})
    logger.info(f"\n--- {tag} vs {MODEL_NAME} ---")
    for h in HORIZONS:
        rm = ref.get(f"h{h}", {}).get("MAE")
        cm = metrics.get(f"h{h}", {}).get("MAE")
        if rm and cm:
            logger.info(f"  h={h}: ref={rm:.4f}  fwd={cm:.4f}  "
                        f"delta={cm - rm:+.4f} ({(cm - rm) / rm * 100:+.1f}%)")


def main():
    logger.info("=" * 60)
    logger.info(f"BACKTESTING - {MODEL_NAME}")
    logger.info(f"Covariates: {EXOG_COLS}  (damped RW-drift forward, phi={PHI})")
    logger.info("=" * 60)
    df = load_data()
    logger.info(f"Data: {len(df)} obs")
    model = load_model()
    df_preds, mase_scale = run_rolling(df, model)
    if df_preds.empty:
        logger.warning("[!] No predictions generated.")
        return
    metrics = compute_metrics(df_preds, mase_scale)
    logger.info(f"\n{'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'Cov80':>6} {'N':>5}")
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]
            logger.info(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                        f"{m['MASE']:8.4f} {m['coverage_80']:6.2%} {m['n_evals']:5d}")

    _delta_log(metrics, "chronos2_C0_metrics.json", "chronos2_C0", "C0")
    _delta_log(metrics, f"{FLAT_HOLD_KEY}_metrics.json", FLAT_HOLD_KEY, "C1_inst (flat-hold)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    with open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info(f"\nSaved: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
