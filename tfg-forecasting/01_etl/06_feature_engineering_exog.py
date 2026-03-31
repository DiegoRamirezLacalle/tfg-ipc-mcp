"""
06_feature_engineering_exog.py — Construccion del dataset exogeno

Une el IPC del INE con los tipos de interes BCE y genera features derivadas.
El resultado es un DataFrame alineado mensualmente listo para SARIMAX.

Features generadas:
  dfr          — Deposit Facility Rate nivel (variable principal)
  mrr          — Main Refinancing Rate nivel (variable secundaria)
  dfr_diff     — Cambio mensual del DFR (accion de politica monetaria)
  dfr_lag3     — DFR con retardo 3 meses
  dfr_lag6     — DFR con retardo 6 meses (transmision tipica: 6-18m)
  dfr_lag12    — DFR con retardo 12 meses

Entrada:  data/processed/ipc_spain_index.parquet
          data/processed/ecb_rates_monthly.parquet
Salida:   data/processed/features_exog.parquet
"""

from pathlib import Path

import pandas as pd

ROOT          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    # ── Cargar fuentes ────────────────────────────────────────────
    ipc = pd.read_parquet(PROCESSED_DIR / "ipc_spain_index.parquet")[["indice_general"]]
    rates = pd.read_parquet(PROCESSED_DIR / "ecb_rates_monthly.parquet")

    # ── Alinear al rango comun ────────────────────────────────────
    df = ipc.join(rates, how="inner")
    df.index.freq = "MS"

    # ── Features derivadas ────────────────────────────────────────
    df["dfr_diff"]  = df["dfr"].diff()
    df["dfr_lag3"]  = df["dfr"].shift(3)
    df["dfr_lag6"]  = df["dfr"].shift(6)
    df["dfr_lag12"] = df["dfr"].shift(12)

    # Los lags introducen NaN al inicio — los documentamos pero no los eliminamos
    # para no perder observaciones del train set; cada modelo recorta lo que necesite
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("NaN por columna (esperados en lags iniciales):")
        print(nan_counts[nan_counts > 0].to_string())

    print(f"\nRango: {df.index.min().date()} - {df.index.max().date()}")
    print(f"Observaciones: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nPrimeras filas:\n{df.head()}")
    print(f"\nCorrelaciones con indice_general:")
    print(df.corr()["indice_general"].drop("indice_general").round(3).to_string())

    out = PROCESSED_DIR / "features_exog.parquet"
    df.to_parquet(out)
    print(f"\nExportado: {out}")


if __name__ == "__main__":
    main()
