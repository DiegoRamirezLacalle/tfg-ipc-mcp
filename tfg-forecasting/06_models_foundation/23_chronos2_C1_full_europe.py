"""
23_chronos2_C1_full_europe.py — Chronos-2 C1_full HICP Eurozona

Covariables: institucionales + MCP (todas las seniales disponibles).
C1_FULL = C1_INST + C1_MCP
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

RESULTS_DIR   = ROOT / "08_results"
HORIZONS      = [1, 3, 6, 12]; MAX_H = 12
ORIGINS_START = "2021-01-01"; ORIGINS_END = DATE_TEST_END
MODEL_NAME    = "chronos2_C1_full_europe"
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)
Q_IDX         = {"p10": 2, "p50": 10, "p90": 18}

C1_INST = ["epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3",
           "dfr", "dfr_ma3", "ttf_ma3", "breakeven_5y_lag1"]
C1_MCP  = ["bce_shock_score", "bce_tone_numeric", "bce_cumstance",
           "gdelt_tone_ma6", "signal_available"]
EXOG_COLS = C1_INST + C1_MCP

# Columnas MCP que en el futuro deben ser neutralizadas (=0)
MCP_NEUTRAL_COLS = ["bce_shock_score", "bce_tone_numeric", "bce_cumstance", "gdelt_tone_ma6"]


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_europe.parquet")
    df.index = pd.to_datetime(df.index); df.index.freq = "MS"
    return df


def load_model():
    from chronos import Chronos2Pipeline
    print("[chronos2] Cargando amazon/chronos-2 ...")
    p = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    print("[chronos2] Cargado"); return p


def prepare_input(df, origin):
    ctx  = df.loc[:origin].copy()
    tgt  = ctx["hicp_index"].values.astype(np.float64)
    past = {c: ctx[c].fillna(0.0).values.astype(np.float64) for c in EXOG_COLS if c in ctx.columns}
    last = ctx[EXOG_COLS].iloc[-1].fillna(0.0)
    future = {}
    for c in EXOG_COLS:
        if c in MCP_NEUTRAL_COLS:
            future[c] = np.zeros(MAX_H, dtype=np.float64)
        elif c == "signal_available":
            future[c] = np.ones(MAX_H, dtype=np.float64)
        else:
            future[c] = np.full(MAX_H, float(last[c]), dtype=np.float64)
    return {"target": tgt, "past_covariates": past, "future_covariates": future}


def run_rolling(df, model):
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
            print(f"\n[!] {origin.date()}: {e}"); continue
        p50 = q[Q_IDX["p50"]]
        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS: continue
            fc_dates = pd.date_range(start=origin + pd.DateOffset(months=1), periods=h, freq="MS")
            ya = y.reindex(fc_dates)
            if ya.isna().any(): continue
            for i, (d, r) in enumerate(zip(fc_dates, ya.values), 1):
                records.append({"origin": origin, "fc_date": d, "step": i, "horizon": h,
                    "model": MODEL_NAME, "y_true": float(r), "y_pred": float(p50[i-1]),
                    "error": float(r - p50[i-1]), "abs_error": float(abs(r - p50[i-1]))})
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
    print(f"Inst: {C1_INST}")
    print(f"MCP:  {C1_MCP}")
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

    # Comparativa vs C0 e inst
    for c0name, key in [("chronos2_C0_europe", "chronos2_C0_europe"),
                        ("chronos2_C1_inst_europe", "chronos2_C1_inst_europe"),
                        ("chronos2_C1_mcp_europe", "chronos2_C1_mcp_europe")]:
        p = RESULTS_DIR / f"{c0name}_metrics.json"
        if p.exists():
            ref = json.load(open(p)).get(key, {})
            print(f"\n--- {key} vs {MODEL_NAME} ---")
            for h in HORIZONS:
                rm = ref.get(f"h{h}", {}).get("MAE")
                cm = metrics.get(f"h{h}", {}).get("MAE")
                if rm and cm:
                    print(f"  h={h}: ref={rm:.4f}  full={cm:.4f}  delta={cm-rm:+.4f} ({(cm-rm)/rm*100:+.1f}%)")

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    json.dump({MODEL_NAME: metrics}, open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w"), indent=2)
    print(f"\nGuardado: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
