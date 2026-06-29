"""
36_chronos2_panel_europe.py - Chronos-2 country panel (Phase B)
---------------------------------------------------------------
Runs the SAME zero-shot Chronos-2 forecasting protocol used for the single
Europe series, but for every euro-area country in hicp_panel_europe.parquet, to
expand the pooled-test cross-section from 3 to ~19 units.

Per country, three conditions on the same rolling origins (2021-01..2024-12,
h={1,3,6,12}, MASE scale on 2002-2020):
  C0        - univariate (target only)
  C1_inst   - shared euro-area covariates, FLAT carry-forward future
  C1_fwd    - shared euro-area covariates, FORWARD-PATH future (damped RW-drift)

Covariates are the shared euro-area set from features_c1_europe.parquet
(area-wide drivers common to every member), so the only thing that differs from
the single-series Europe runs is the target country. No retraining.

Output:
  08_results/chronos2_panel_europe_predictions.parquet   (long; cols incl. country, model)
  08_results/chronos2_panel_europe_metrics.json
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
PROCESSED = ROOT / "data" / "processed"
HORIZONS = [1, 3, 6, 12]
MAX_H = 12
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
Q_P50 = 10  # median quantile index (same as scripts 21/35)

EXOG_COLS = ["epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3"]
DRIFT_WINDOW, PHI = 12, 0.85
CONDITIONS = ["C0", "C1_inst", "C1_fwd"]


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


def build_country_frame(panel: pd.DataFrame, cov: pd.DataFrame, country: str) -> pd.DataFrame:
    s = (panel[panel["country"] == country]
         .set_index("date")["hicp_index"].sort_index())
    df = pd.DataFrame({"hicp_index": s}).join(cov, how="left")
    df.index.freq = "MS"
    return df


def _inputs(df: pd.DataFrame, origin: pd.Timestamp):
    """Return {condition: model-input dict} for one origin (data <= origin only)."""
    ctx = df.loc[:origin]
    tgt = ctx["hicp_index"].values.astype(np.float64)
    past = {c: ctx[c].fillna(0.0).values.astype(np.float64) for c in EXOG_COLS}
    flat = build_future_covariates(df, EXOG_COLS, origin, MAX_H,
                                   ExogPolicy.CARRY_FORWARD, fillna=0.0)
    fwd = build_future_covariates(df, EXOG_COLS, origin, MAX_H,
                                  ExogPolicy.FORWARD_PATH, fillna=0.0,
                                  drift_window=DRIFT_WINDOW, phi=PHI)
    return {
        "C0":      {"target": tgt},
        "C1_inst": {"target": tgt, "past_covariates": past, "future_covariates": flat},
        "C1_fwd":  {"target": tgt, "past_covariates": past, "future_covariates": fwd},
    }


def run_country(df: pd.DataFrame, country: str, model) -> tuple[list[dict], float]:
    y = df["hicp_index"]
    origins = pd.date_range(ORIGINS_START, ORIGINS_END, freq="MS")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in origins:
        inp = _inputs(df, origin)
        preds = {}
        for cond in CONDITIONS:
            try:
                out = model.predict([inp[cond]], prediction_length=MAX_H)
                preds[cond] = out[0].numpy()[0][Q_P50]
            except Exception as e:
                logger.warning("[!] %s %s %s: %s", country, cond, origin.date(), e)
        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS:
                continue
            fc_dates = pd.date_range(origin + pd.DateOffset(months=1), periods=h, freq="MS")
            ya = y.reindex(fc_dates)
            if ya.isna().any():
                continue
            for cond in CONDITIONS:
                if cond not in preds:
                    continue
                p50 = preds[cond]
                for i, (d, r) in enumerate(zip(fc_dates, ya.values), 1):
                    yp = float(p50[i - 1])
                    records.append({
                        "country": country, "model": cond, "origin": origin,
                        "fc_date": d, "step": i, "horizon": h,
                        "y_true": float(r), "y_pred": yp,
                        "error": float(r - yp), "abs_error": float(abs(r - yp)),
                    })
    return records, mase_scale


def metrics_for(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    res = {}
    for cond in CONDITIONS:
        cd = df_preds[df_preds["model"] == cond]
        res[cond] = {}
        for h in HORIZONS:
            hd = cd[cd["horizon"] == h]
            if hd.empty:
                continue
            yt, yp = hd["y_true"].values, hd["y_pred"].values
            res[cond][f"h{h}"] = {
                "MAE": round(float(np.mean(np.abs(yt - yp))), 4),
                "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
                "MASE": round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
                "n_evals": int(len(hd["origin"].unique())),
            }
    return res


def main():
    logger.info("=" * 60)
    logger.info("PANEL BACKTEST - Chronos-2 euro-area HICP (C0 / C1_inst / C1_fwd)")
    logger.info("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    cov = load_covariates()
    countries = sorted(panel["country"].unique())
    logger.info("Countries (%d): %s", len(countries), ", ".join(countries))

    model = load_model()

    all_records, all_metrics = [], {}
    for country in tqdm(countries, desc="countries"):
        df = build_country_frame(panel, cov, country)
        recs, scale = run_country(df, country, model)
        all_records.extend(recs)
        cdf = pd.DataFrame(recs)
        m = metrics_for(cdf, scale)
        m["mase_scale"] = round(scale, 6)
        all_metrics[country] = m
        d12 = {c: m[c].get("h12", {}).get("MAE") for c in CONDITIONS}
        logger.info("  %s h12 MAE  C0=%.3f  C1_inst=%.3f  C1_fwd=%.3f",
                    country, d12["C0"], d12["C1_inst"], d12["C1_fwd"])

    df_all = pd.DataFrame(all_records)
    df_all.to_parquet(RESULTS_DIR / "chronos2_panel_europe_predictions.parquet", index=False)
    with open(RESULTS_DIR / "chronos2_panel_europe_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("\nSaved: chronos2_panel_europe_predictions.parquet  (%d rows)", len(df_all))
    logger.info("Saved: chronos2_panel_europe_metrics.json  (%d countries)", len(all_metrics))


if __name__ == "__main__":
    main()
