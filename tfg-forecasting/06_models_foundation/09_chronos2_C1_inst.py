"""
09_chronos2_C1_inst.py — Chronos-2 C1 solo EPU Europe (institutional)

Covariables (3): epu_europe_ma3 (corr 0.737), epu_europe_log (0.701), epu_europe_lag1 (0.682)
Datos completos desde 2002, sin NaN. Contexto IPC completo.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))
from shared.constants import DATE_TRAIN_END, DATE_TEST_END

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]; MAX_H = 12
ORIGINS_START = "2021-01-01"; ORIGINS_END = DATE_TEST_END
MODEL_NAME = "chronos2_C1_inst"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
Q_IDX = {"p10": 2, "p50": 10, "p90": 18}

EXOG_COLS = ["epu_europe_ma3", "epu_europe_log", "epu_europe_lag1"]

SUBPERIODS = {
    "A_2021": ("2021-01-01", "2021-12-01"),
    "B_2022_shock": ("2022-01-01", "2022-12-01"),
    "C_2023_2024": ("2023-01-01", "2024-12-01"),
}

def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"]); df = df.set_index("date"); df.index.freq = "MS"
    return df

def load_model():
    from chronos import Chronos2Pipeline
    print(f"[chronos2] Cargando amazon/chronos-2 ...");
    p = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    print("[chronos2] Cargado"); return p

def prepare_input(df, origin, h):
    ctx = df.loc[:origin]
    target = ctx["indice_general"].values.astype(np.float64)
    past = {c: ctx[c].values.astype(np.float64) for c in EXOG_COLS if c in ctx.columns}
    last = df.loc[:origin, EXOG_COLS].iloc[-1]
    future = {c: np.full(h, float(last[c]), dtype=np.float64) for c in EXOG_COLS}
    return {"target": target, "past_covariates": past, "future_covariates": future}

def run_rolling(df, model):
    y = df["indice_general"]; origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
        inp = prepare_input(df, origin, MAX_H)
        try:
            preds = model.predict([inp], prediction_length=MAX_H); q = preds[0].numpy()[0]
        except Exception as e:
            print(f"\n[!] {origin.date()}: {e}"); continue
        p50, p10, p90 = q[Q_IDX["p50"]], q[Q_IDX["p10"]], q[Q_IDX["p90"]]
        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS: continue
            fc_dates = pd.date_range(start=origin+pd.DateOffset(months=1), periods=h, freq="MS")
            ya = y.reindex(fc_dates)
            if ya.isna().any(): continue
            for i, (d, r) in enumerate(zip(fc_dates, ya.values), 1):
                records.append({"origin":origin,"fc_date":d,"step":i,"horizon":h,"model":MODEL_NAME,
                    "y_true":float(r),"y_pred":float(p50[i-1]),"y_pred_p10":float(p10[i-1]),
                    "y_pred_p90":float(p90[i-1]),"error":float(r-p50[i-1]),"abs_error":float(abs(r-p50[i-1]))})
    return pd.DataFrame(records), mase_scale

def compute_metrics(df_preds, mase_scale):
    res = {}
    for h in HORIZONS:
        hd = df_preds[df_preds["horizon"]==h]
        if hd.empty: continue
        yt, yp = hd["y_true"].values, hd["y_pred"].values
        p10, p90 = hd["y_pred_p10"].values, hd["y_pred_p90"].values
        cov = float(np.mean((yt>=p10)&(yt<=p90)))
        res[f"h{h}"] = {"MAE":round(float(np.mean(np.abs(yt-yp))),4),
            "RMSE":round(float(np.sqrt(np.mean((yt-yp)**2))),4),
            "MASE":round(float(np.mean(np.abs(yt-yp))/mase_scale),4),
            "coverage_80":round(cov,4),"n_evals":int(len(hd["origin"].unique()))}
    return res

def main():
    print(f"{'='*60}\nBACKTESTING — {MODEL_NAME}\nCovariables: {EXOG_COLS}\n{'='*60}")
    df = load_data(); print(f"Datos: {len(df)} obs"); model = load_model()
    df_preds, mase_scale = run_rolling(df, model)
    if df_preds.empty: print("[!] Sin predicciones"); return
    metrics = compute_metrics(df_preds, mase_scale)
    print(f"\n{'Horizonte':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'Cov80':>6} {'N':>5}")
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]; print(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} {m['MASE']:8.4f} {m['coverage_80']:6.2%} {m['n_evals']:5d}")
    c0p = RESULTS_DIR/"chronos2_C0_metrics.json"
    if c0p.exists():
        c0 = json.load(open(c0p)).get("chronos2_C0",{})
        print(f"\n--- C0 vs {MODEL_NAME} ---")
        for h in HORIZONS:
            c0m = c0.get(f"h{h}",{}).get("MAE"); c1m = metrics.get(f"h{h}",{}).get("MAE")
            if c0m and c1m: print(f"  h={h}: C0={c0m:.4f}  {MODEL_NAME}={c1m:.4f}  delta={c1m-c0m:+.4f} ({(c1m-c0m)/c0m*100:+.1f}%)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR/f"{MODEL_NAME}_predictions.parquet", index=False)
    json.dump({MODEL_NAME: metrics}, open(RESULTS_DIR/f"{MODEL_NAME}_metrics.json","w"), indent=2)
    print(f"\nGuardado: {MODEL_NAME}_metrics.json")

if __name__ == "__main__":
    main()
