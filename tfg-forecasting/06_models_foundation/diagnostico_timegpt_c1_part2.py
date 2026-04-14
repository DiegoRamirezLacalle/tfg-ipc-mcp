"""
diagnostico_timegpt_c1_part2.py — D2 y D3 (continuacion del diagnostico)

D1 ya confirmado: Variante B (2015+) es la mejor, MAE = 0.4038 vs C0 = 0.5646
Ahora ejecutamos D2 (subconjuntos) y D3 (fill strategy) usando Variante B.

Para ahorrar llamadas, usamos 1 origen representativo + Variante B.
Presupuesto: 12 llamadas (5 subconjuntos + 3 fill + 1 C0 ref + 3 margen).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

SERIES_ID = "ipc_spain"
MAX_H = 12

# Usar 3 origenes: uno por subperiodo
TEST_ORIGINS = pd.to_datetime([
    "2021-06-01",   # periodo tranquilo
    "2022-09-01",   # pleno shock BCE
    "2023-09-01",   # post-shock
])

ALL_EXOG = [
    "gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6",
    "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
    "ine_surprise_score", "ine_inflacion",
    "signal_available",
]

api_calls = 0


def load_ipc() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df["indice_general"]
    y.index = pd.to_datetime(y.index)
    y.index.freq = "MS"
    return y


def load_features() -> pd.DataFrame:
    c1 = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    c1["date"] = pd.to_datetime(c1["date"])
    c1 = c1.set_index("date")
    c1.index.freq = "MS"
    for col in ALL_EXOG:
        if col in c1.columns:
            c1[col] = c1[col].fillna(0.0)
    return c1


def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


def forecast_one(
    client, y, origin, exog=None, exog_cols=None,
    start_date="2015-01-01", fill_strategy="forward",
):
    """Una llamada API. Devuelve prediccion h=1."""
    global api_calls

    context_y = y.loc[start_date:origin] if start_date else y.loc[:origin]

    hist = pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": context_y.index,
        "y": context_y.values,
    })

    future = None
    if exog is not None and exog_cols:
        # Historico
        ctx_exog = exog.loc[context_y.index[0]:origin, exog_cols].reindex(context_y.index)
        for col in exog_cols:
            hist[col] = ctx_exog[col].values if col in ctx_exog.columns else 0.0

        # Futuro
        fc_dates = pd.date_range(
            start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
        )
        future = pd.DataFrame({"unique_id": SERIES_ID, "ds": fc_dates})
        last_row = exog.loc[:origin, exog_cols].iloc[-1]

        if fill_strategy == "forward":
            for col in exog_cols:
                future[col] = 1.0 if col == "signal_available" else float(last_row[col])
        elif fill_strategy == "zero":
            for col in exog_cols:
                future[col] = 0.0
        elif fill_strategy == "mean3":
            last3 = exog.loc[:origin, exog_cols].tail(3).mean()
            for col in exog_cols:
                future[col] = 1.0 if col == "signal_available" else float(last3[col])

    kwargs = dict(
        df=hist, h=MAX_H, freq="MS",
        time_col="ds", target_col="y", id_col="unique_id",
    )
    if future is not None:
        kwargs["X_df"] = future

    try:
        fc = client.forecast(**kwargs)
        api_calls += 1
        fc = fc.sort_values("ds").reset_index(drop=True)
        return float(fc["TimeGPT"].iloc[0])
    except Exception as e:
        api_calls += 1
        print(f"    [!] Error: {e}")
        return None


def main():
    global api_calls

    print("=" * 60)
    print("DIAGNOSTICO TIMEGPT C1 — Parte 2 (D2 + D3)")
    print(f"Origenes: {[o.strftime('%Y-%m') for o in TEST_ORIGINS]}")
    print("Todas las variantes usan contexto desde 2015 (Variante B)")
    print("=" * 60)

    y = load_ipc()
    exog = load_features()
    client = get_client()

    # Actuals para h=1
    actuals = {}
    for origin in TEST_ORIGINS:
        target = origin + pd.DateOffset(months=1)
        actuals[origin] = float(y.loc[target])

    # ── C0 baseline (sin exogenas, contexto completo) ──
    print("\n--- C0 baseline (contexto completo, sin exogenas) ---")
    c0_preds = {}
    for origin in TEST_ORIGINS:
        p = forecast_one(client, y, origin, start_date=None)
        c0_preds[origin] = p
        ae = abs(p - actuals[origin]) if p else None
        print(f"  {origin.strftime('%Y-%m')}: pred={p:.2f}  actual={actuals[origin]:.2f}  AE={ae:.4f}" if p else
              f"  {origin.strftime('%Y-%m')}: ERROR")

    c0_mae = np.mean([abs(c0_preds[o] - actuals[o]) for o in TEST_ORIGINS if c0_preds[o]])
    print(f"  C0 MAE h=1: {c0_mae:.4f}")

    # ── D2: Subconjuntos de senales (Variante B: desde 2015) ──
    print("\n" + "=" * 60)
    print("D2 — Subconjuntos de senales (contexto desde 2015)")
    print("=" * 60)

    subsets = {
        "GDELT_3": ["gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6"],
        "BCE_3": ["bce_shock_score", "bce_tone_numeric", "bce_cumstance"],
        "INE_2": ["ine_surprise_score", "ine_inflacion"],
        "STANCE_2": ["bce_cumstance", "signal_available"],
        "ALL_9": ALL_EXOG,
    }

    d2_results = {}
    for name, cols in subsets.items():
        print(f"\n  [{name}] {cols}")
        preds = {}
        for origin in TEST_ORIGINS:
            p = forecast_one(client, y, origin, exog, cols, start_date="2015-01-01")
            preds[origin] = p
        mae = np.mean([abs(preds[o] - actuals[o]) for o in TEST_ORIGINS if preds[o]])
        d2_results[name] = mae
        delta = (mae - c0_mae) / c0_mae * 100
        print(f"    MAE h=1: {mae:.4f}  (vs C0: {delta:+.1f}%)")

    # ── D3: Estrategia fill (con ALL_9, Variante B) ──
    print("\n" + "=" * 60)
    print("D3 — Estrategia fill horizonte futuro (ALL_9, desde 2015)")
    print("=" * 60)

    d3_results = {}
    for strat in ["forward", "zero", "mean3"]:
        print(f"\n  [{strat}]")
        preds = {}
        for origin in TEST_ORIGINS:
            p = forecast_one(client, y, origin, exog, ALL_EXOG,
                             start_date="2015-01-01", fill_strategy=strat)
            preds[origin] = p
        mae = np.mean([abs(preds[o] - actuals[o]) for o in TEST_ORIGINS if preds[o]])
        d3_results[strat] = mae
        delta = (mae - c0_mae) / c0_mae * 100
        print(f"    MAE h=1: {mae:.4f}  (vs C0: {delta:+.1f}%)")

    # ── RESUMEN ──
    print("\n" + "=" * 60)
    print("RESUMEN DIAGNOSTICO COMPLETO")
    print("=" * 60)

    print(f"\nC0 baseline MAE h=1: {c0_mae:.4f}")

    print("\nD1 (de part1): Hipotesis regimen ceros CONFIRMADA")
    print("  A_completo:    MAE = 0.4714  (vs C0: -16.5%)")
    print("  B_desde2015:   MAE = 0.4038  (vs C0: -28.5%)  <-- MEJOR")
    print("  C_nan_pre2015: MAE = 0.4294  (vs C0: -23.9%)")

    print("\nD2: Subconjuntos de senales (Variante B):")
    d2_sorted = sorted(d2_results.items(), key=lambda x: x[1])
    for name, mae in d2_sorted:
        delta = (mae - c0_mae) / c0_mae * 100
        marker = " <-- MEJOR" if mae == d2_sorted[0][1] else ""
        print(f"  {name:15s}: MAE = {mae:.4f}  (vs C0: {delta:+.1f}%){marker}")

    neutrals = [k for k, v in d2_sorted if v <= c0_mae * 1.05]
    harmful = [k for k, v in d2_sorted if v > c0_mae * 1.20]
    better = [k for k, v in d2_sorted if v < c0_mae * 0.95]
    print(f"\n  Mejoran C0 (>5%): {better or ['ninguno']}")
    print(f"  Neutras (<=5% degradacion): {neutrals or ['ninguna']}")
    print(f"  Daninas (>20% degradacion): {harmful or ['ninguna']}")

    print("\nD3: Estrategia fill horizonte:")
    d3_sorted = sorted(d3_results.items(), key=lambda x: x[1])
    for strat, mae in d3_sorted:
        delta = (mae - c0_mae) / c0_mae * 100
        marker = " <-- MEJOR" if mae == d3_sorted[0][1] else ""
        print(f"  {strat:15s}: MAE = {mae:.4f}  (vs C0: {delta:+.1f}%){marker}")

    best_subset = d2_sorted[0][0]
    best_subset_mae = d2_sorted[0][1]
    best_fill = d3_sorted[0][0]
    best_fill_mae = d3_sorted[0][1]

    print("\n" + "=" * 60)
    print("CONCLUSION RECOMENDADA")
    print("=" * 60)
    print(f"  1. Recortar contexto a 2015+ (Variante B)")
    print(f"  2. Mejor subconjunto de senales: {best_subset} (MAE={best_subset_mae:.4f})")
    print(f"  3. Mejor estrategia fill: {best_fill} (MAE={best_fill_mae:.4f})")
    print(f"  4. C0 baseline: MAE={c0_mae:.4f}")
    delta_best = (best_subset_mae - c0_mae) / c0_mae * 100
    print(f"  5. Mejora esperada vs C0: {delta_best:+.1f}%")
    print(f"\n  Total llamadas API: {api_calls}")


if __name__ == "__main__":
    main()
