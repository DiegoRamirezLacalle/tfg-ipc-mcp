"""
08_chronos2_C1_energy_only.py — Chronos-2 C1 solo con variables energeticas

Experimento de ablacion: aislar efecto predictivo de energia sin senales MCP.

Covariables (3 total, solo energia):
    brent_ma3       # corr 0.715 con IPC(t+1) en 2015+
    brent_ret       # captura shocks energeticos rapidos
    ttf_ma3         # corr 0.541 con IPC(t+1) en 2015+

Todas con datos reales desde 2002 (proxy backfill). Sin NaN.
Contexto completo desde 2002 (no recortar a 2015+).
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
MODEL_NAME = "chronos2_C1_energy_only"
CHRONOS_MODEL_ID = "amazon/chronos-2"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

EXOG_COLS = ["brent_ma3", "brent_ret", "ttf_ma3"]

SUBPERIODS = {
    "A_2021": ("2021-01-01", "2021-12-01"),
    "B_2022_shock": ("2022-01-01", "2022-12-01"),
    "C_2023_2024": ("2023-01-01", "2024-12-01"),
}

Q_IDX = {"p10": 2, "p50": 10, "p90": 18}


# ── Datos ────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    return df


# ── Modelo ───────────────────────────────────────────────────────

def load_model():
    from chronos import Chronos2Pipeline
    print(f"[chronos2] Cargando {CHRONOS_MODEL_ID} ...")
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID, device_map="cpu",
    )
    print("[chronos2] Modelo cargado")
    return pipeline


# ── Preparar inputs ──────────────────────────────────────────────

def prepare_input(df: pd.DataFrame, origin: pd.Timestamp, h: int) -> dict:
    """Contexto completo desde 2002 + 3 covariables energia."""
    context_df = df.loc[:origin]
    target = context_df["indice_general"].values.astype(np.float64)

    past_covs = {}
    for col in EXOG_COLS:
        if col in context_df.columns:
            past_covs[col] = context_df[col].values.astype(np.float64)

    # Future: forward-fill del ultimo valor
    last_row = df.loc[:origin, EXOG_COLS].iloc[-1]
    future_covs = {}
    for col in EXOG_COLS:
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

    for origin in tqdm(origins, desc="Chronos2 C1 energy_only"):
        inp = prepare_input(df, origin, MAX_H)

        try:
            preds = model.predict([inp], prediction_length=MAX_H)
            q = preds[0].numpy()[0]  # (21, 12)
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


def compute_subperiod_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
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
    print(f"BACKTESTING — {MODEL_NAME}")
    print(f"Modelo: {CHRONOS_MODEL_ID}")
    print(f"Covariables: {EXOG_COLS}")
    print("=" * 60)

    df = load_data()
    print(f"Datos: {df.index.min().date()} - {df.index.max().date()} ({len(df)} obs)")

    model = load_model()

    df_preds, mase_scale = run_rolling(df, model)
    print(f"\nPredicciones: {len(df_preds)}")

    if df_preds.empty:
        print("[!] No se generaron predicciones.")
        return

    metrics = compute_metrics(df_preds, mase_scale)
    sub_metrics = compute_subperiod_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print(f"RESULTADOS GLOBALES — {MODEL_NAME}")
    print("=" * 60)
    print_table(metrics)

    print("\n" + "=" * 60)
    print(f"SUBPERIODOS — {MODEL_NAME}")
    print("=" * 60)
    print_subperiod_table(sub_metrics)

    # Comparativa C0
    c0_path = RESULTS_DIR / "chronos2_C0_metrics.json"
    if c0_path.exists():
        with open(c0_path) as f:
            c0_m = json.load(f).get("chronos2_C0", {})
        print("\n--- C0 vs C1_energy_only (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            c0_mae = c0_m.get(key, {}).get("MAE")
            c1_mae = metrics.get(key, {}).get("MAE")
            if c0_mae and c1_mae:
                delta = c1_mae - c0_mae
                print(f"  h={h}: C0={c0_mae:.4f}  energy_only={c1_mae:.4f}  "
                      f"delta={delta:+.4f} ({delta/c0_mae*100:+.1f}%)")

    # Guardar
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    print(f"\nPredicciones: {preds_path}")

    metrics_path = RESULTS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    print(f"Metricas: {metrics_path}")

    sub_path = RESULTS_DIR / f"{MODEL_NAME}_subperiod_metrics.json"
    with open(sub_path, "w") as f:
        json.dump({MODEL_NAME: sub_metrics}, f, indent=2)
    print(f"Subperiodos: {sub_path}")


if __name__ == "__main__":
    main()
