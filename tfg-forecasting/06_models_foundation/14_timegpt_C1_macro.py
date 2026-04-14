"""
14_timegpt_C1_macro.py — TimeGPT C1 macro: brent + ttf + EPU Europe

Covariables (3, best-of-domain):
  brent_ma3       # corr 0.715 energia
  ttf_ma3         # corr 0.541 gas
  epu_europe_ma3  # corr 0.737 incertidumbre institucional

Datos completos desde 2002, sin NaN. IPC contexto completo 2002+.
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))
from shared.constants import DATE_TRAIN_END, DATE_TEST_END

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]; MAX_H = 12
ORIGINS_START = "2021-01-01"; ORIGINS_END = DATE_TEST_END
MODEL_NAME = "timegpt_C1_macro"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SERIES_ID = "ipc_spain"

EXOG_COLS = ["brent_ma3", "ttf_ma3", "epu_europe_ma3"]

def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui": raise ValueError("NIXTLA_API_KEY no configurada.")
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)

def load_data():
    ipc_df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = ipc_df["indice_general"]; y.index = pd.to_datetime(y.index); y.index.freq = "MS"
    c1 = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    c1["date"] = pd.to_datetime(c1["date"]); c1 = c1.set_index("date"); c1.index.freq = "MS"
    return y, c1

def build_nixtla_df(y, exog, origin):
    context_y = y.loc[:origin]
    hist_df = pd.DataFrame({"unique_id": SERIES_ID, "ds": context_y.index, "y": context_y.values})
    for col in EXOG_COLS:
        col_vals = exog.loc[:origin, col].reindex(context_y.index)
        hist_df[col] = col_vals.values
    last_row = exog.loc[:origin, EXOG_COLS].iloc[-1]
    fc_dates = pd.date_range(start=origin+pd.DateOffset(months=1), periods=MAX_H, freq="MS")
    future_df = pd.DataFrame({"unique_id": SERIES_ID, "ds": fc_dates})
    for col in EXOG_COLS:
        future_df[col] = float(last_row[col])
    return hist_df, future_df

def run_rolling(y, exog, client, test_run=False):
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run: origins = origins[:5]; print(f"[test-run] {len(origins)} origenes")
    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []
    for origin in tqdm(origins, desc=f"{MODEL_NAME}"):
        hist_df, future_df = build_nixtla_df(y, exog, origin)
        try:
            fc = client.forecast(df=hist_df, X_df=future_df, h=MAX_H, freq="MS",
                time_col="ds", target_col="y", id_col="unique_id")
            pred_values = fc.sort_values("ds")["TimeGPT"].values
        except Exception as e:
            print(f"\n[!] {origin.date()}: {e}"); continue
        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS: continue
            fc_dates = pd.date_range(start=origin+pd.DateOffset(months=1), periods=h, freq="MS")
            ya = y.reindex(fc_dates)
            if ya.isna().any(): continue
            for i, (d, r, p) in enumerate(zip(fc_dates, ya.values, pred_values[:h]), 1):
                records.append({"origin":origin,"fc_date":d,"step":i,"horizon":h,"model":MODEL_NAME,
                    "y_true":float(r),"y_pred":float(p),"error":float(r-p),"abs_error":float(abs(r-p))})
    return pd.DataFrame(records), mase_scale

def compute_metrics(df_preds, mase_scale):
    res = {}
    for h in HORIZONS:
        hd = df_preds[df_preds["horizon"]==h]
        if hd.empty: continue
        yt, yp = hd["y_true"].values, hd["y_pred"].values
        res[f"h{h}"] = {"MAE":round(float(np.mean(np.abs(yt-yp))),4),
            "RMSE":round(float(np.sqrt(np.mean((yt-yp)**2))),4),
            "MASE":round(float(np.mean(np.abs(yt-yp))/mase_scale),4),
            "n_evals":int(len(hd["origin"].unique()))}
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()
    n = 5 if args.test_run else 48
    print(f"{'='*60}\nBACKTESTING — {MODEL_NAME} ({'TEST' if args.test_run else 'FULL'})\n"
          f"Covariables: {EXOG_COLS}\nCoste estimado: {n} llamadas API\n{'='*60}")
    y, exog = load_data(); client = get_client()
    df_preds, mase_scale = run_rolling(y, exog, client, test_run=args.test_run)
    if df_preds.empty: print("[!] Sin predicciones"); return
    metrics = compute_metrics(df_preds, mase_scale)
    print(f"\n{'Horizonte':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]; print(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} {m['MASE']:8.4f} {m['n_evals']:5d}")
    c0p = RESULTS_DIR/"timegpt_C0_metrics.json"
    if c0p.exists():
        c0 = json.load(open(c0p)).get("timegpt_C0",{})
        print(f"\n--- C0 vs {MODEL_NAME} ---")
        for h in HORIZONS:
            c0m = c0.get(f"h{h}",{}).get("MAE"); c1m = metrics.get(f"h{h}",{}).get("MAE")
            if c0m and c1m: print(f"  h={h}: C0={c0m:.4f}  C1={c1m:.4f}  delta={c1m-c0m:+.4f} ({(c1m-c0m)/c0m*100:+.1f}%)")
    if args.test_run:
        print("\n[test-run] OK. Lanzar sin --test-run para backtesting completo."); return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR/f"{MODEL_NAME}_predictions.parquet", index=False)
    json.dump({MODEL_NAME: metrics}, open(RESULTS_DIR/f"{MODEL_NAME}_metrics.json","w"), indent=2)
    print(f"\nGuardado: {MODEL_NAME}_metrics.json")

if __name__ == "__main__":
    main()
