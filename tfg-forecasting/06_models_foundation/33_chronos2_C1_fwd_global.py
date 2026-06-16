"""
33_chronos2_C1_fwd_global.py - Chronos-2 C1 Global, HONEST forward covariates

Phase 2 fix for root cause #2 (future exogenous paths held FLAT). Identical to
15_chronos2_C1_inst_global.py (same target cpi_global_rate, same native
covariates, same origins, same MASE scale) EXCEPT the future covariate path is
a genuine forecast instead of the last value repeated.

Forward path (pre-registered, leakage-free -- uses only data <= origin):
  damped random-walk-with-drift. For covariate x with last value x_T and
  drift = mean monthly change over the trailing DRIFT_WINDOW months,
      x_{T+k} = x_T + drift * sum_{j=0..k-1} PHI^j
  so the per-step increment fades with PHI (=0.85), preventing the runaway
  extrapolation of an undamped RW-drift while still propagating recent momentum
  rather than freezing the series.

This is the same architecture as script 15; only `future_covariates` changes,
so the comparison vs 15 (flat-hold) isolates the value of the forward path.
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
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
MAX_H = 12
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "chronos2_C1_fwd_global"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
Q_IDX = {"p10": 2, "p50": 10, "p90": 18}

EXOG_COLS = ["imf_comm_ma3", "brent_log_ma3", "gscpi_ma3"]

# Forward-path hyperparameters (fixed a priori; not tuned on the test window).
DRIFT_WINDOW = 12
PHI = 0.85

SUBPERIODS = {
    "A_pre_crisis":    ("2021-01-01", "2022-06-01"),
    "B_shock":         ("2022-07-01", "2023-06-01"),
    "C_normalizacion": ("2023-07-01", "2024-12-01"),
}


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(
        ROOT / "data" / "processed" / "features_c1_global_institutional.parquet"
    )
    df.index = pd.to_datetime(df.index)
    df.index.freq = "MS"
    return df


def load_model():
    from chronos import Chronos2Pipeline
    logger.info("[chronos2] Loading amazon/chronos-2 ...")
    p = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    logger.info("[chronos2] Loaded")
    return p


def forecast_covariate(values: np.ndarray, h: int) -> np.ndarray:
    """Damped RW-with-drift forecast for one covariate (data <= origin only)."""
    last = float(values[-1])
    tail = values[-(DRIFT_WINDOW + 1):]
    diffs = np.diff(tail)
    drift = float(np.nanmean(diffs)) if diffs.size else 0.0
    out = np.empty(h, dtype=np.float64)
    cum = 0.0
    weight = 1.0
    for k in range(h):
        cum += weight * drift
        out[k] = last + cum
        weight *= PHI
    return out


def prepare_input(df: pd.DataFrame, origin: pd.Timestamp, h: int) -> dict:
    ctx = df.loc[:origin]
    target = ctx["cpi_global_rate"].values.astype(np.float64)
    past = {c: ctx[c].values.astype(np.float64) for c in EXOG_COLS if c in ctx.columns}
    future = {c: forecast_covariate(ctx[c].values.astype(np.float64), h) for c in EXOG_COLS}
    return {"target": target, "past_covariates": past, "future_covariates": future}


def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["cpi_global_rate"]
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
        res[f"h{h}"] = {
            "MAE":     round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE":    round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE":    round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def compute_subperiod_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    res = {}
    for sp, (s, e) in SUBPERIODS.items():
        mask = (df_preds["fc_date"] >= s) & (df_preds["fc_date"] <= e)
        res[sp] = {}
        for h in HORIZONS:
            hd = df_preds[mask & (df_preds["horizon"] == h)]
            if hd.empty:
                continue
            yt, yp = hd["y_true"].values, hd["y_pred"].values
            res[sp][f"h{h}"] = {
                "MAE":  round(float(np.mean(np.abs(yt - yp))), 4),
                "MASE": round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
                "n":    int(len(hd["origin"].unique())),
            }
    return res


def main():
    logger.info("=" * 60)
    logger.info(f"BACKTESTING - {MODEL_NAME}")
    logger.info(f"Covariates: {EXOG_COLS}  (damped RW-drift forward, phi={PHI})")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    model = load_model()

    df_preds, mase_scale = run_rolling(df, model)
    if df_preds.empty:
        logger.warning("[!] No predictions generated.")
        return

    metrics = compute_metrics(df_preds, mase_scale)
    sp_metrics = compute_subperiod_metrics(df_preds, mase_scale)

    logger.info(f"\n{'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]
            logger.info(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                        f"{m['MASE']:8.4f} {m['n_evals']:5d}")

    # Comparison vs GLOBAL C0 (script 30) and vs flat-hold C1 (script 15).
    for tag, mfile, mkey in [
        ("C0_global", "chronos2_C0_global_metrics.json", "chronos2_C0_global"),
        ("C1_inst_global (flat-hold)", "chronos2_C1_inst_global_metrics.json", "chronos2_C1_inst_global"),
    ]:
        p = RESULTS_DIR / mfile
        if not p.exists():
            logger.warning("[!] %s metrics missing (%s) - skipping comparison", tag, mfile)
            continue
        ref = json.loads(p.read_text(encoding="utf-8")).get(mkey, {})
        logger.info(f"\n--- {tag} vs {MODEL_NAME} ---")
        for h in HORIZONS:
            rm = ref.get(f"h{h}", {}).get("MAE")
            cm = metrics.get(f"h{h}", {}).get("MAE")
            if rm and cm:
                logger.info(f"  h={h}: ref={rm:.4f}  fwd={cm:.4f}  "
                            f"delta={cm - rm:+.4f} ({(cm - rm) / rm * 100:+.1f}%)")

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    with open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w") as f:
        json.dump({MODEL_NAME: metrics, "subperiods": sp_metrics}, f, indent=2)
    logger.info(f"\nSaved: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
