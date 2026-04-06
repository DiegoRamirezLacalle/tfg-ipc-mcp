"""
04_chronos2_C1.py — Chronos-2 condicion C1 (historico + senales MCP)

Rolling-origin backtesting con covariables del pipeline MCP.

Chronos-2 soporta covariables nativamente via dict inputs:
  - past_covariates: valores historicos de todas las covariables
  - future_covariates: valores futuros de covariables conocidas (DFR/MRR)
  Para senales MCP (no conocidas a futuro): solo past_covariates

Modelo: amazon/chronos-2
Senales MCP ya con shift +1 aplicado (sin leakage temporal).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
MAX_H = max(HORIZONS)
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "chronos2_C1"
CHRONOS_MODEL_ID = "amazon/chronos-2"

# Cuantiles: 21 levels [0.01..0.99]
Q_IDX = {"p10": 2, "p50": 10, "p90": 18}

# Covariables numericas — separadas en "conocidas a futuro" y "solo pasado"
# DFR/MRR son publicas en tiempo real (BCE decisions), se pueden pasar como future
KNOWN_FUTURE_COVS = ["dfr", "mrr"]
PAST_ONLY_COVS = [
    "dfr_diff", "dfr_lag3", "dfr_lag6", "dfr_lag12",
    "gdelt_avg_tone", "gdelt_goldstein", "gdelt_n_articles",
    "bce_shock_score", "bce_uncertainty", "ine_surprise_score",
]
# Categoricas como covariables — Chronos-2 soporta categoricas en numpy
CAT_COVS = ["bce_tone", "dominant_topic"]


# ── Datos ────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"

    for col in KNOWN_FUTURE_COVS + PAST_ONLY_COVS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    for col in CAT_COVS:
        if col in df.columns:
            df[col] = df[col].fillna("neutral")
    return df


# ── Modelo ───────────────────────────────────────────────────────

def load_model():
    from chronos import Chronos2Pipeline

    print(f"[chronos2] Cargando {CHRONOS_MODEL_ID} ...")
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        device_map="cpu",
    )
    print("[chronos2] Modelo cargado (21 cuantiles, covariables nativas)")
    return pipeline


# ── Preparar inputs con covariables ─────────────────────────────

def prepare_input(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    h: int,
) -> dict:
    """
    Prepara un dict input para Chronos-2 con covariables.

    Estructura:
      - target: 1D array de IPC historico
      - past_covariates: dict de arrays (len = context)
      - future_covariates: dict de arrays (len = h), solo para DFR/MRR
    """
    context_df = df.loc[:origin]
    target = context_df["indice_general"].values.astype(np.float64)

    # Past covariates: todas (numericas + categoricas)
    past_covs = {}
    for col in KNOWN_FUTURE_COVS + PAST_ONLY_COVS:
        if col in context_df.columns:
            past_covs[col] = context_df[col].values.astype(np.float64)
    for col in CAT_COVS:
        if col in context_df.columns:
            past_covs[col] = context_df[col].values.astype(str)

    # Future covariates: solo DFR/MRR (conocidas, publicas en tiempo real)
    fc_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
    )
    future_covs = {}
    for col in KNOWN_FUTURE_COVS:
        if col in df.columns:
            future_vals = df[col].reindex(fc_dates)
            # Si no hay valores futuros reales, forward-fill
            if future_vals.isna().any():
                last_val = float(context_df[col].iloc[-1])
                future_vals = future_vals.fillna(last_val)
            future_covs[col] = future_vals.values.astype(np.float64)

    return {
        "target": target,
        "past_covariates": past_covs,
        "future_covariates": future_covs if future_covs else None,
    }


# ── Rolling backtesting ─────────────────────────────────────────

def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["indice_general"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="Chronos2 C1 rolling"):
        # Preparar input con covariables para h=12
        inp = prepare_input(df, origin, MAX_H)

        # Filtrar future_covariates None
        input_dict = {"target": inp["target"], "past_covariates": inp["past_covariates"]}
        if inp["future_covariates"]:
            input_dict["future_covariates"] = inp["future_covariates"]

        try:
            preds = model.predict([input_dict], prediction_length=MAX_H)
            # Shape: (n_variates, n_quantiles, pred_len) = (1, 21, 12)
            quantiles = preds[0].numpy()
            q = quantiles[0]  # (21, 12)
        except Exception as e:
            print(f"\n[!] Error en {origin.date()}: {e}")
            continue

        p50 = q[Q_IDX["p50"]]
        p10 = q[Q_IDX["p10"]]
        p90 = q[Q_IDX["p90"]]

        for h in HORIZONS:
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue

            y_true = y_actual.values

            for i, (date, real) in enumerate(zip(fc_dates, y_true), start=1):
                records.append({
                    "origin": origin,
                    "fc_date": date,
                    "step": i,
                    "horizon": h,
                    "model": MODEL_NAME,
                    "y_true": float(real),
                    "y_pred": float(p50[i - 1]),
                    "y_pred_p10": float(p10[i - 1]),
                    "y_pred_p90": float(p90[i - 1]),
                    "error": float(real - p50[i - 1]),
                    "abs_error": float(abs(real - p50[i - 1])),
                })

    return pd.DataFrame(records), mase_scale


# ── Metricas ─────────────────────────────────────────────────────

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for h in HORIZONS:
        h_df = df_preds[df_preds["horizon"] == h]
        if h_df.empty:
            continue
        y_true = h_df["y_true"].values
        y_pred = h_df["y_pred"].values
        p10 = h_df["y_pred_p10"].values
        p90 = h_df["y_pred_p90"].values
        coverage = float(np.mean((y_true >= p10) & (y_true <= p90)))

        results[f"h{h}"] = {
            "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
            "coverage_80": round(coverage, 4),
            "n_evals": int(len(h_df["origin"].unique())),
        }
    return results


def print_table(metrics: dict) -> None:
    print(f"\n{'Horizonte':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'Cov80':>6} {'N':>5}")
    print("-" * 52)
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics:
            m = metrics[key]
            print(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                  f"{m['MASE']:8.4f} {m['coverage_80']:6.2%} {m['n_evals']:5d}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"BACKTESTING ROLLING — {MODEL_NAME}")
    print(f"Modelo: {CHRONOS_MODEL_ID}")
    print(f"Origenes: {ORIGINS_START} - {ORIGINS_END}")
    print(f"Horizontes: {HORIZONS}")
    print(f"Covariables (known future): {KNOWN_FUTURE_COVS}")
    print(f"Covariables (past only): {PAST_ONLY_COVS}")
    print(f"Covariables (categoricas): {CAT_COVS}")
    print("=" * 60)

    df = load_data()
    print(f"Datos: {df.index.min().date()} - {df.index.max().date()} ({len(df)} obs)")

    model = load_model()

    df_preds, mase_scale = run_rolling(df, model)
    print(f"\nPredicciones generadas: {len(df_preds)}")

    if df_preds.empty:
        print("[!] No se generaron predicciones. Revisar errores arriba.")
        return

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print(f"RESULTADOS {MODEL_NAME}")
    print("=" * 60)
    print_table(metrics)

    # Comparativa C0 vs C1
    c0_path = RESULTS_DIR / "chronos2_C0_metrics.json"
    if c0_path.exists():
        with open(c0_path) as f:
            c0_metrics = json.load(f).get("chronos2_C0", {})
        print("\n--- C0 vs C1 (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            c0_mae = c0_metrics.get(key, {}).get("MAE", "N/A")
            c1_mae = metrics.get(key, {}).get("MAE", "N/A")
            if isinstance(c0_mae, float) and isinstance(c1_mae, float):
                delta = c1_mae - c0_mae
                pct = (delta / c0_mae) * 100
                print(f"  h={h}: C0={c0_mae:.4f}  C1={c1_mae:.4f}  "
                      f"delta={delta:+.4f} ({pct:+.1f}%)")

    # Guardar
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    print(f"\nPredicciones: {preds_path}")

    metrics_path = RESULTS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    print(f"Metricas:     {metrics_path}")


if __name__ == "__main__":
    main()
