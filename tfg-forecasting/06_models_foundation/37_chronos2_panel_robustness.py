"""
37_chronos2_panel_robustness.py - Phase B robustness / placebo (Chronos-2 panel)
--------------------------------------------------------------------------------
Bulletproofs the Phase B forward-path result against two attacks:

  1. "You tuned phi/window."  -> run extra *informed* forward-path settings and
     show the forward-vs-flat win survives:
        fwd_phi100  : undamped RW-with-drift (phi=1.0, window=12)
        fwd_w24     : longer drift window (phi=0.85, window=24)
     (the canonical phi=0.85/window=12 already exists in the panel run, script 36)

  2. "Any non-flat path would help, not specifically the informed drift."
     -> a PLACEBO forward path with the SAME damped-drift magnitude but a
     RANDOM SIGN per (country, origin, covariate), seeded and reproducible:
        placebo_randsign
     If the benefit is the *informed direction* of recent momentum, randomizing
     the sign should destroy it (placebo ~ flat-hold, not better).

Everything else is identical to script 36 (same countries, covariates, origins,
MASE scale). Only future covariates change. No retraining.

Output (appended as new model labels; C0/C1_inst/C1_fwd untouched):
  08_results/chronos2_panel_robustness_predictions.parquet
  08_results/chronos2_panel_robustness_metrics.json
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
from shared.exog_policies import ExogPolicy, build_future_covariates, damped_rw_drift_path
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
PROCESSED = ROOT / "data" / "processed"
HORIZONS = [1, 3, 6, 12]
MAX_H = 12
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
Q_P50 = 10
EXOG_COLS = ["epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3"]
PLACEBO_SEED = 20260629

# Informed forward-path sensitivity variants: (label, drift_window, phi).
INFORMED_VARIANTS = [
    ("fwd_phi100", 12, 1.00),
    ("fwd_w24",    24, 0.85),
]
PLACEBO_LABEL = "placebo_randsign"
ALL_LABELS = [v[0] for v in INFORMED_VARIANTS] + [PLACEBO_LABEL]


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED / "hicp_panel_europe.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p


def load_covariates() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "features_c1_europe.parquet")
    df.index = pd.to_datetime(df.index)
    return df[EXOG_COLS].copy()


def load_model():
    from chronos import Chronos2Pipeline
    logger.info("[chronos2] Loading amazon/chronos-2 ...")
    m = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    logger.info("[chronos2] Loaded")
    return m


def build_country_frame(panel, cov, country) -> pd.DataFrame:
    s = panel[panel["country"] == country].set_index("date")["hicp_index"].sort_index()
    df = pd.DataFrame({"hicp_index": s}).join(cov, how="left")
    df.index.freq = "MS"
    return df


def placebo_path(values: np.ndarray, h: int, rng: np.random.Generator,
                 drift_window: int = 12, phi: float = 0.85) -> np.ndarray:
    """Damped-drift path with the same |drift| magnitude but a RANDOM SIGN.

    Non-flat with the same per-step magnitude as the informed forward path, but
    the direction is randomized -> isolates 'informed momentum' from 'any wiggle'.
    """
    values = np.asarray(values, dtype=np.float64)
    last = float(values[-1])
    tail = values[-(drift_window + 1):]
    diffs = np.diff(tail)
    drift = float(np.nanmean(diffs)) if diffs.size else 0.0
    sign = 1.0 if rng.random() < 0.5 else -1.0
    drift = sign * abs(drift)
    out = np.empty(h, dtype=np.float64)
    cum, w = 0.0, 1.0
    for k in range(h):
        cum += w * drift
        out[k] = last + cum
        w *= phi
    return out


def run_country(df: pd.DataFrame, country: str, model, rng: np.random.Generator):
    y = df["hicp_index"]
    origins = pd.date_range(ORIGINS_START, ORIGINS_END, freq="MS")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in origins:
        ctx = df.loc[:origin]
        tgt = ctx["hicp_index"].values.astype(np.float64)
        past = {c: ctx[c].fillna(0.0).values.astype(np.float64) for c in EXOG_COLS}

        variant_future = {}
        for label, w, phi in INFORMED_VARIANTS:
            variant_future[label] = build_future_covariates(
                df, EXOG_COLS, origin, MAX_H, ExogPolicy.FORWARD_PATH,
                fillna=0.0, drift_window=w, phi=phi)
        # placebo: random sign per covariate (draw in fixed EXOG order for repro)
        series_ffill = ctx[EXOG_COLS].fillna(0.0)
        variant_future[PLACEBO_LABEL] = {
            c: placebo_path(series_ffill[c].values, MAX_H, rng) for c in EXOG_COLS
        }

        preds = {}
        for label in ALL_LABELS:
            try:
                out = model.predict(
                    [{"target": tgt, "past_covariates": past,
                      "future_covariates": variant_future[label]}],
                    prediction_length=MAX_H)
                preds[label] = out[0].numpy()[0][Q_P50]
            except Exception as e:
                logger.warning("[!] %s %s %s: %s", country, label, origin.date(), e)

        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS:
                continue
            fc_dates = pd.date_range(origin + pd.DateOffset(months=1), periods=h, freq="MS")
            ya = y.reindex(fc_dates)
            if ya.isna().any():
                continue
            for label in ALL_LABELS:
                if label not in preds:
                    continue
                p50 = preds[label]
                for i, (d, r) in enumerate(zip(fc_dates, ya.values), 1):
                    yp = float(p50[i - 1])
                    records.append({
                        "country": country, "model": label, "origin": origin,
                        "fc_date": d, "step": i, "horizon": h,
                        "y_true": float(r), "y_pred": yp,
                        "error": float(r - yp), "abs_error": float(abs(r - yp)),
                    })
    return records, mase_scale


def metrics_for(cdf: pd.DataFrame, scale: float) -> dict:
    res = {}
    for label in ALL_LABELS:
        md = cdf[cdf["model"] == label]
        res[label] = {}
        for h in HORIZONS:
            hd = md[md["horizon"] == h]
            if hd.empty:
                continue
            yt, yp = hd["y_true"].values, hd["y_pred"].values
            res[label][f"h{h}"] = {
                "MAE": round(float(np.mean(np.abs(yt - yp))), 4),
                "MASE": round(float(np.mean(np.abs(yt - yp)) / scale), 4),
                "n_evals": int(len(hd["origin"].unique())),
            }
    return res


def main():
    logger.info("=" * 60)
    logger.info("PANEL ROBUSTNESS - Chronos-2 (informed phi/window + placebo)")
    logger.info("Variants: %s  | placebo seed=%d", ", ".join(ALL_LABELS), PLACEBO_SEED)
    logger.info("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    cov = load_covariates()
    countries = sorted(panel["country"].unique())
    model = load_model()

    all_records, all_metrics = [], {}
    for ci, country in enumerate(tqdm(countries, desc="countries")):
        rng = np.random.default_rng(PLACEBO_SEED + ci)  # reproducible per country
        df = build_country_frame(panel, cov, country)
        recs, scale = run_country(df, country, model, rng)
        all_records.extend(recs)
        m = metrics_for(pd.DataFrame(recs), scale)
        m["mase_scale"] = round(scale, 6)
        all_metrics[country] = m

    df_all = pd.DataFrame(all_records)
    df_all.to_parquet(RESULTS_DIR / "chronos2_panel_robustness_predictions.parquet", index=False)
    with open(RESULTS_DIR / "chronos2_panel_robustness_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("\nSaved: chronos2_panel_robustness_predictions.parquet (%d rows)", len(df_all))
    logger.info("Saved: chronos2_panel_robustness_metrics.json (%d countries)", len(all_metrics))


if __name__ == "__main__":
    main()
