"""
24_timesfm_C1_inst_europe.py — TimesFM C1_institutional HICP Eurozona

Arquitectura: TimesFM base + Ridge correction con seniales institucionales.
Ridge entrenado sobre ventana expandiente 2002:origin.
Covariables: epu_europe_ma3, brent_ma3, esi_eurozone, eurusd_ma3,
             dfr, dfr_ma3, ttf_ma3, breakeven_5y_lag1
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))
from shared.constants import DATE_TRAIN_END, DATE_TEST_END

RESULTS_DIR   = ROOT / "08_results"
HORIZONS      = [1, 3, 6, 12]; MAX_H = 12
ORIGINS_START = "2021-01-01"; ORIGINS_END = DATE_TEST_END
MODEL_NAME    = "timesfm_C1_inst_europe"
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)
RIDGE_ALPHA   = 1.0

XREG_COVS = ["epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3",
             "dfr", "dfr_ma3", "ttf_ma3", "breakeven_5y_lag1"]


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_europe.parquet")
    df.index = pd.to_datetime(df.index); df.index.freq = "MS"
    return df


def load_model():
    import timesfm
    print("[timesfm] Cargando google/timesfm-2.5-200m-pytorch ...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tfm.compile(forecast_config=timesfm.ForecastConfig(
        max_context=512, max_horizon=MAX_H, per_core_batch_size=1))
    print("[timesfm] Cargado"); return tfm


def compute_xreg_correction(df, origin):
    window = df.loc[:origin].copy()
    if len(window) < 13: return 0.0
    rate_mom = window["hicp_index"].diff(1)
    valid = ~rate_mom.isna()
    X_raw = window.loc[valid, XREG_COVS].fillna(0.0).values.astype(np.float64)
    y = rate_mom[valid].values.astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    reg.fit(X, y)
    current_raw = df.loc[origin:origin, XREG_COVS].fillna(0.0).values.astype(np.float64)
    current = scaler.transform(current_raw)
    neutral = np.zeros_like(current)  # scaled-space 0 = media histórica de cada feature
    return float(reg.predict(current)[0] - reg.predict(neutral)[0])


def run_rolling(df, model):
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
            print(f"\n[!] Base error {origin.date()}: {e}"); continue
        correction = compute_xreg_correction(df, origin)
        full_pred  = base_pred + correction
        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS: continue
            fc_dates = pd.date_range(start=origin + pd.DateOffset(months=1), periods=h, freq="MS")
            ya = y.reindex(fc_dates)
            if ya.isna().any(): continue
            for i, (d, r, p) in enumerate(zip(fc_dates, ya.values, full_pred[:h]), 1):
                records.append({"origin": origin, "fc_date": d, "step": i, "horizon": h,
                    "model": MODEL_NAME, "y_true": float(r), "y_pred": float(p),
                    "error": float(r - p), "abs_error": float(abs(r - p))})
    return pd.DataFrame(records), mase_scale


def compute_metrics(df_preds, mase_scale):
    res = {}
    for h in HORIZONS:
        hd = df_preds[df_preds["horizon"] == h]
        if hd.empty: continue
        yt, yp = hd["y_true"].values, hd["y_pred"].values
        res[f"h{h}"] = {
            "MAE":     round(float(np.mean(np.abs(yt - yp))), 4),
            "RMSE":    round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "MASE":    round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def main():
    print("=" * 60)
    print(f"BACKTESTING — {MODEL_NAME}")
    print(f"Ridge covariables: {XREG_COVS}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df    = load_data()
    model = load_model()

    df_preds, mase_scale = run_rolling(df, model)
    if df_preds.empty:
        print("[!] Sin predicciones"); return

    metrics = compute_metrics(df_preds, mase_scale)

    print(f"\n{'Horizonte':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    print("-" * 45)
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]
            print(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} {m['MASE']:8.4f} {m['n_evals']:5d}")

    c0p = RESULTS_DIR / "timesfm_C0_europe_metrics.json"
    if c0p.exists():
        c0 = json.load(open(c0p)).get("timesfm_C0_europe", {})
        print(f"\n--- C0 vs {MODEL_NAME} ---")
        for h in HORIZONS:
            c0m = c0.get(f"h{h}", {}).get("MAE")
            c1m = metrics.get(f"h{h}", {}).get("MAE")
            if c0m and c1m:
                print(f"  h={h}: C0={c0m:.4f}  C1={c1m:.4f}  delta={c1m-c0m:+.4f} ({(c1m-c0m)/c0m*100:+.1f}%)")

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    json.dump({MODEL_NAME: metrics}, open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w"), indent=2)
    print(f"\nGuardado: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
