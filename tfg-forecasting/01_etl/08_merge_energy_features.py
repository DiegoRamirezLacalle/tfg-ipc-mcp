"""
08_merge_energy_features.py — Merge energy_prices_monthly into features_c1

Hace merge por fecha, forward-fill de NaN (gaps en Yahoo Finance),
y calcula correlaciones de las 8 nuevas variables con IPC(t+1).

Entrada: data/processed/features_c1.parquet (23 cols)
         data/processed/energy_prices_monthly.parquet (8 cols)
Salida:  data/processed/features_c1.parquet (31 cols, sobreescrito)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

ENERGY_COLS = [
    "brent_log", "brent_ret", "brent_ma3", "brent_lag1",
    "ttf_log", "ttf_ret", "ttf_ma3", "ttf_lag1",
]


def main():
    print("=" * 60)
    print("MERGE ENERGY PRICES INTO features_c1")
    print("=" * 60)

    # ── Cargar ────────────────────────────────────────────────────
    c1 = pd.read_parquet(PROCESSED_DIR / "features_c1.parquet")
    energy = pd.read_parquet(PROCESSED_DIR / "energy_prices_monthly.parquet")

    print(f"features_c1: {c1.shape[0]} filas, {c1.shape[1]} cols")
    print(f"energy:      {energy.shape[0]} filas, {energy.shape[1]} cols")

    # Asegurar que date es columna y tipo datetime
    if "date" not in c1.columns:
        c1 = c1.reset_index().rename(columns={c1.index.name or "index": "date"})
    c1["date"] = pd.to_datetime(c1["date"])

    energy = energy.reset_index()
    energy.rename(columns={energy.columns[0]: "date"}, inplace=True)
    energy["date"] = pd.to_datetime(energy["date"])

    # Verificar que no hay columnas de energia ya en c1
    existing = [c for c in ENERGY_COLS if c in c1.columns]
    if existing:
        print(f"\n[!] Columnas de energia ya existen en features_c1: {existing}")
        print("    Sobreescribiendo...")
        c1 = c1.drop(columns=existing)

    # ── Merge ─────────────────────────────────────────────────────
    c1 = c1.merge(energy[["date"] + ENERGY_COLS], on="date", how="left")

    # Forward-fill NaN de gaps en Yahoo Finance (precios continuos)
    for col in ENERGY_COLS:
        c1[col] = c1[col].ffill()
        # Los primeros meses sin dato (si brent/ttf empieza despues de 2002)
        c1[col] = c1[col].bfill()

    nan_after = c1[ENERGY_COLS].isna().sum()
    print(f"\nNaN tras forward/back-fill: {nan_after.sum()}")

    # ── Guardar ───────────────────────────────────────────────────
    out = PROCESSED_DIR / "features_c1.parquet"
    c1.to_parquet(out, index=False)
    print(f"\nfeatures_c1 actualizado: {c1.shape[0]} filas, {c1.shape[1]} cols")
    print(f"Columnas: {list(c1.columns)}")

    # ── Correlaciones con IPC(t+1) ────────────────────────────────
    print("\n" + "=" * 60)
    print("CORRELACION ENERGY VARS con IPC(t+1)")
    print("=" * 60)

    c1_indexed = c1.set_index("date")
    ipc_lead = c1_indexed["indice_general"].shift(-1)  # IPC del mes siguiente

    print(f"\n{'Variable':<18} {'Corr IPC(t+1)':>14} {'Corr |abs|':>12} {'Periodo':>18} {'N':>5}")
    print("-" * 72)

    corr_results = {}
    for col in ENERGY_COLS:
        series = c1_indexed[col]
        valid = series.notna() & ipc_lead.notna()
        if valid.sum() < 20:
            print(f"{col:<18} {'N/A':>14} {'N/A':>12} {'insuficiente':>18} {valid.sum():>5}")
            continue

        corr = float(series[valid].corr(ipc_lead[valid]))
        # Correlacion con IPC(t+1) en primera diferencia (mas informativa)
        ipc_diff = c1_indexed["indice_general"].diff()
        ipc_diff_lead = ipc_diff.shift(-1)
        valid_diff = series.notna() & ipc_diff_lead.notna()
        corr_diff = float(series[valid_diff].corr(ipc_diff_lead[valid_diff])) if valid_diff.sum() > 20 else 0.0

        start = series.dropna().index.min().strftime("%Y-%m")
        end = series.dropna().index.max().strftime("%Y-%m")

        corr_results[col] = {"corr_level": corr, "corr_diff": corr_diff, "n": int(valid.sum())}
        print(f"{col:<18} {corr:>+14.3f} {corr_diff:>+12.3f} {start+' - '+end:>18} {valid.sum():>5}")

    # ── Correlaciones solo periodo 2015+ (con senales MCP) ────────
    print(f"\n{'Variable':<18} {'Corr 2015+':>14} {'Diff 2015+':>12}")
    print("-" * 48)
    mask_2015 = c1_indexed.index >= "2015-01-01"
    for col in ENERGY_COLS:
        series_15 = c1_indexed.loc[mask_2015, col]
        ipc_lead_15 = ipc_lead[mask_2015]
        valid = series_15.notna() & ipc_lead_15.notna()
        if valid.sum() < 10:
            continue
        corr = float(series_15[valid].corr(ipc_lead_15[valid]))

        ipc_diff_lead_15 = c1_indexed["indice_general"].diff().shift(-1)[mask_2015]
        valid_d = series_15.notna() & ipc_diff_lead_15.notna()
        corr_d = float(series_15[valid_d].corr(ipc_diff_lead_15[valid_d])) if valid_d.sum() > 10 else 0.0

        print(f"{col:<18} {corr:>+14.3f} {corr_d:>+12.3f}")

    # ── Recomendacion ─────────────────────────────────────────────
    strong = [c for c, v in corr_results.items() if abs(v["corr_level"]) > 0.5]
    medium = [c for c, v in corr_results.items()
              if 0.3 < abs(v["corr_level"]) <= 0.5]
    weak = [c for c, v in corr_results.items() if abs(v["corr_level"]) <= 0.3]

    print(f"\n{'='*60}")
    print("RECOMENDACION")
    print(f"{'='*60}")
    print(f"  Correlacion fuerte (>0.5): {strong or 'ninguna'}")
    print(f"  Correlacion media (0.3-0.5): {medium or 'ninguna'}")
    print(f"  Correlacion debil (<0.3): {weak or 'ninguna'}")

    if strong:
        print(f"\n  >>> LANZAR modelos C1 con energia: {strong}")
        print(f"      Correlacion justifica el coste de la API.")
    elif medium:
        print(f"\n  >>> CONSIDERAR modelos C1 con energia: {medium}")
        print(f"      Correlacion moderada, puede ayudar en combinacion con MCP.")
    else:
        print(f"\n  >>> NO lanzar modelos con energia.")
        print(f"      Correlaciones demasiado debiles para justificar coste API.")


if __name__ == "__main__":
    main()
