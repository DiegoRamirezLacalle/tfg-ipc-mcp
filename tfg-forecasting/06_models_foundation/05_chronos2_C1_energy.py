"""
05_chronos2_C1_energy.py — Chronos-2 C1 con variables energeticas + MCP seleccionadas

Covariables (8 total, subconjunto optimo):
  Energia (datos reales desde 2002, sin NaN):
    brent_ma3       # corr 0.715 con IPC(t+1) en 2015+
    ttf_ma3         # corr 0.541
    brent_ret       # captura shocks energeticos rapidos

  MCP (NaN pre-2015, datos reales post-2015):
    bce_shock_score, bce_tone_numeric, bce_cumstance,
    gdelt_tone_ma6, signal_available

Contexto: IPC completo desde 2002 + separacion energia/MCP para NaN.
Chronos-2 soporta covariables nativamente via dict inputs.
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
MODEL_NAME = "chronos2_C1_energy"
CHRONOS_MODEL_ID = "amazon/chronos-2"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SIGNAL_START = "2015-01-01"

# Subperiodos para analisis granular
SUBPERIODS = {
    "A_2021": ("2021-01-01", "2021-12-01"),
    "B_2022_shock": ("2022-01-01", "2022-12-01"),
    "C_2023_2024": ("2023-01-01", "2024-12-01"),
}

# Cuantiles: 21 levels [0.01..0.99]
Q_IDX = {"p10": 2, "p50": 10, "p90": 18}

# Energia: datos reales desde 2002 (Brent via proxy WTI), sin NaN pre-2015
ENERGY_COLS = ["brent_ma3", "brent_ret", "ttf_ma3"]

# MCP: NaN/cero pre-2015, datos reales post-2015
MCP_COLS = [
    "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
    "gdelt_tone_ma6", "signal_available",
]

EXOG_COLS = ENERGY_COLS + MCP_COLS


# ── Datos ────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"

    for col in EXOG_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
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

    Tratamiento diferenciado Fix v2:
      - IPC: contexto COMPLETO desde 2002 (tendencia largo plazo)
      - ENERGY_COLS: datos reales desde 2002, sin NaN
      - MCP_COLS: cero pre-2015, datos reales post-2015
    """
    # Contexto completo desde 2002 (no recortar a 2015+)
    context_df = df.loc[:origin]
    target = context_df["indice_general"].values.astype(np.float64)

    # Past covariates
    past_covs = {}

    # Energia: datos reales completos
    for col in ENERGY_COLS:
        if col in context_df.columns:
            past_covs[col] = context_df[col].values.astype(np.float64)

    # MCP: cero pre-2015, datos reales post-2015
    # (Chronos-2 recibe arrays numericos, NaN no soportado bien -> usar 0.0)
    for col in MCP_COLS:
        if col in context_df.columns:
            vals = context_df[col].copy()
            vals.loc[vals.index < SIGNAL_START] = 0.0
            past_covs[col] = vals.values.astype(np.float64)

    # Future covariates: forward-fill del ultimo valor conocido
    fc_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
    )
    future_covs = {}
    last_row = df.loc[:origin, EXOG_COLS].iloc[-1]

    for col in EXOG_COLS:
        if col == "signal_available":
            future_covs[col] = np.ones(h, dtype=np.float64)
        else:
            future_covs[col] = np.full(h, float(last_row[col]), dtype=np.float64)

    return {
        "target": target,
        "past_covariates": past_covs,
        "future_covariates": future_covs,
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

    for origin in tqdm(origins, desc="Chronos2 C1 energy rolling"):
        inp = prepare_input(df, origin, MAX_H)

        input_dict = {
            "target": inp["target"],
            "past_covariates": inp["past_covariates"],
            "future_covariates": inp["future_covariates"],
        }

        try:
            preds = model.predict([input_dict], prediction_length=MAX_H)
            quantiles = preds[0].numpy()
            q = quantiles[0]  # (21, 12)
        except Exception as e:
            print(f"\n[!] Error en {origin.date()}: {e}")
            continue

        p50 = q[Q_IDX["p50"]]
        p10 = q[Q_IDX["p10"]]
        p90 = q[Q_IDX["p90"]]

        for h in HORIZONS:
            horizon_end = origin + pd.DateOffset(months=h)
            if horizon_end > TEST_END_TS:
                continue

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


def compute_subperiod_metrics(
    df_preds: pd.DataFrame, mase_scale: float
) -> dict:
    results = {}
    for period_name, (start, end) in SUBPERIODS.items():
        mask = (
            (df_preds["origin"] >= pd.Timestamp(start))
            & (df_preds["origin"] <= pd.Timestamp(end))
        )
        period_df = df_preds[mask]
        if period_df.empty:
            continue
        results[period_name] = {}
        for h in HORIZONS:
            h_df = period_df[period_df["horizon"] == h]
            if h_df.empty:
                continue
            y_true = h_df["y_true"].values
            y_pred = h_df["y_pred"].values
            p10 = h_df["y_pred_p10"].values
            p90 = h_df["y_pred_p90"].values
            coverage = float(np.mean((y_true >= p10) & (y_true <= p90)))
            results[period_name][f"h{h}"] = {
                "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
                "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
                "MASE": round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
                "coverage_80": round(coverage, 4),
                "n_origins": int(len(h_df["origin"].unique())),
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


def print_subperiod_table(sub_metrics: dict) -> None:
    print(f"\n{'Periodo':<18} {'h':>3} {'MAE':>8} {'MASE':>8} {'Cov80':>6} {'N':>4}")
    print("-" * 52)
    for period_name, hdict in sub_metrics.items():
        for h in HORIZONS:
            key = f"h{h}"
            if key in hdict:
                m = hdict[key]
                print(f"{period_name:<18} {h:>3} {m['MAE']:8.4f} {m['MASE']:8.4f} "
                      f"{m['coverage_80']:6.2%} {m['n_origins']:4d}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"BACKTESTING ROLLING — {MODEL_NAME}")
    print(f"Modelo: {CHRONOS_MODEL_ID}")
    print(f"Origenes: {ORIGINS_START} - {ORIGINS_END}")
    print(f"Horizontes: {HORIZONS}")
    print(f"Energy cols: {ENERGY_COLS}")
    print(f"MCP cols: {MCP_COLS}")
    print("=" * 60)

    df = load_data()
    print(f"Datos: {df.index.min().date()} - {df.index.max().date()} ({len(df)} obs)")
    print(f"Exogenas disponibles: {[c for c in EXOG_COLS if c in df.columns]}")

    model = load_model()

    df_preds, mase_scale = run_rolling(df, model)
    print(f"\nPredicciones generadas: {len(df_preds)}")

    if df_preds.empty:
        print("[!] No se generaron predicciones. Revisar errores arriba.")
        return

    metrics = compute_metrics(df_preds, mase_scale)
    sub_metrics = compute_subperiod_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print(f"RESULTADOS GLOBALES — {MODEL_NAME}")
    print("=" * 60)
    print_table(metrics)

    print("\n" + "=" * 60)
    print(f"RESULTADOS POR SUBPERIODO — {MODEL_NAME}")
    print("=" * 60)
    print_subperiod_table(sub_metrics)

    # Comparativa C0 vs C1_energy
    c0_path = RESULTS_DIR / "chronos2_C0_metrics.json"
    if c0_path.exists():
        with open(c0_path) as f:
            c0_metrics = json.load(f).get("chronos2_C0", {})
        print("\n--- C0 vs C1_energy (MAE global) ---")
        for h in HORIZONS:
            key = f"h{h}"
            c0_mae = c0_metrics.get(key, {}).get("MAE", "N/A")
            c1_mae = metrics.get(key, {}).get("MAE", "N/A")
            if isinstance(c0_mae, float) and isinstance(c1_mae, float):
                delta = c1_mae - c0_mae
                pct = (delta / c0_mae) * 100
                print(f"  h={h}: C0={c0_mae:.4f}  C1_energy={c1_mae:.4f}  "
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

    sub_path = RESULTS_DIR / f"{MODEL_NAME}_subperiod_metrics.json"
    with open(sub_path, "w") as f:
        json.dump({MODEL_NAME: sub_metrics}, f, indent=2)
    print(f"Subperiodos:  {sub_path}")


if __name__ == "__main__":
    main()
