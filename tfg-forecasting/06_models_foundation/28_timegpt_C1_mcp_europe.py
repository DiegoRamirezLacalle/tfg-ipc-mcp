"""
28_timegpt_C1_mcp_europe.py — TimeGPT C1_mcp HICP Eurozona

TimeGPT nativo con solo seniales MCP/BCE como covariables.
Para el futuro (h pasos), todas las seniales MCP se ponen a 0 (neutral),
signal_available=1.

Control costes: --test-run (5 origenes) / default full.
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

RESULTS_DIR   = ROOT / "08_results"
HORIZONS      = [1, 3, 6, 12]; MAX_H = 12
ORIGINS_START = "2021-01-01"; ORIGINS_END = DATE_TEST_END
MODEL_NAME    = "timegpt_C1_mcp_europe"
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)
SERIES_ID     = "HICP_EUROPE"

XREG_COVS       = ["bce_shock_score", "bce_tone_numeric", "bce_cumstance",
                   "gdelt_tone_ma6", "signal_available"]
MCP_NEUTRAL_COLS = ["bce_shock_score", "bce_tone_numeric", "bce_cumstance", "gdelt_tone_ma6"]


def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError("NIXTLA_API_KEY no configurada.")
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1_europe.parquet")
    df.index = pd.to_datetime(df.index); df.index.freq = "MS"
    return df


def build_nixtla_dfs(df, origin):
    hist = df.loc[:origin]
    last_vals = hist[XREG_COVS].iloc[-1].fillna(0.0)

    # df: historical target + exogenous (required by nixtla API)
    hist_reset = hist[XREG_COVS].fillna(0.0).reset_index().rename(columns={"date": "ds"})
    df_tgt = pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": hist.index,
        "y": hist["hicp_index"].values,
    })
    for c in XREG_COVS:
        df_tgt[c] = hist_reset[c].values

    # X_df: future rows — carry-forward last known MCP values to avoid
    # level discontinuity (bce_cumstance is unbounded cumulative, not in [-1,1])
    future_idx = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
    )
    rows_fut = pd.DataFrame({"ds": future_idx, "unique_id": SERIES_ID})
    for c in XREG_COVS:
        rows_fut[c] = 1.0 if c == "signal_available" else float(last_vals[c])

    X_df = rows_fut[["unique_id", "ds"] + XREG_COVS]
    return df_tgt, X_df


def run_rolling(df, client, test_run=False):
    y = df["hicp_index"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run:
        origins = origins[:5]
        print(f"[test-run] {len(origins)} origenes")

    yt = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(yt.values[12:] - yt.values[:-12])))
    records = []

    for origin in tqdm(origins, desc=f"{MODEL_NAME} rolling"):
        df_tgt, X_df = build_nixtla_dfs(df, origin)
        try:
            fc = client.forecast(
                df=df_tgt, X_df=X_df, h=MAX_H, freq="MS",
                time_col="ds", target_col="y", id_col="unique_id",
            )
            fc = fc.sort_values("ds").reset_index(drop=True)
            pred_values = fc["TimeGPT"].values
        except Exception as e:
            print(f"\n[!] {origin.date()}: {e}"); continue

        for h in HORIZONS:
            if origin + pd.DateOffset(months=h) > TEST_END_TS: continue
            fc_dates = pd.date_range(start=origin + pd.DateOffset(months=1), periods=h, freq="MS")
            ya = y.reindex(fc_dates)
            if ya.isna().any(): continue
            for i, (d, r, p) in enumerate(zip(fc_dates, ya.values, pred_values[:h]), 1):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"BACKTESTING — {MODEL_NAME}")
    print(f"Covariables MCP: {XREG_COVS}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df     = load_data()
    client = get_client()

    df_preds, mase_scale = run_rolling(df, client, test_run=args.test_run)
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

    for ref_name in ["timegpt_C0_europe", "timegpt_C1_inst_europe"]:
        p = RESULTS_DIR / f"{ref_name}_metrics.json"
        if p.exists():
            ref = json.load(open(p)).get(ref_name, {})
            print(f"\n--- {ref_name} vs {MODEL_NAME} ---")
            for h in HORIZONS:
                rm = ref.get(f"h{h}", {}).get("MAE")
                cm = metrics.get(f"h{h}", {}).get("MAE")
                if rm and cm:
                    print(f"  h={h}: ref={rm:.4f}  mcp={cm:.4f}  delta={cm-rm:+.4f} ({(cm-rm)/rm*100:+.1f}%)")

    if args.test_run:
        print("\n[test-run] Lanzar sin --test-run para completo."); return

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    json.dump({MODEL_NAME: metrics}, open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w"), indent=2)
    print(f"\nGuardado: {MODEL_NAME}_metrics.json")


if __name__ == "__main__":
    main()
