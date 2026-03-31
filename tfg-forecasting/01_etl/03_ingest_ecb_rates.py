"""
03_ingest_ecb_rates.py — Ingestión de tipos de interés BCE

Lee los CSV de cambios diarios del BCE (DFR y MRR), resamplea a frecuencia
mensual con last().ffill() (correcto para series escalonadas de política
monetaria) y exporta a data/processed/ecb_rates_monthly.parquet.

Entrada:  data/raw/DFR.csv  — Deposit Facility Rate (tipo de depósito)
          data/raw/MRR.csv  — Main Refinancing Rate (tipo de refinanciación)
Salida:   data/processed/ecb_rates_monthly.parquet
"""

from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]   # tfg-forecasting/
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Constantes ────────────────────────────────────────────────────────────
DATE_START = "2002-01-01"
DATE_END   = "2025-06-01"


def load_rate(path: Path, col_name: str) -> pd.Series:
    """Lee un CSV de cambios diarios BCE y devuelve serie mensual."""
    df = pd.read_csv(path, usecols=["TIME_PERIOD", "OBS_VALUE"])
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    df = df.sort_values("TIME_PERIOD").set_index("TIME_PERIOD")

    # Resampleo a Month Start: último valor del mes, luego forward-fill
    # ffill propaga el tipo vigente a los meses sin cambio (comportamiento correcto)
    monthly = (
        df["OBS_VALUE"]
        .resample("MS")
        .last()
        .ffill()
    )
    monthly.name = col_name
    return monthly


def main() -> None:
    dfr = load_rate(RAW_DIR / "DFR.csv", "dfr")
    mrr = load_rate(RAW_DIR / "MRR.csv", "mrr")

    rates = pd.concat([dfr, mrr], axis=1)

    # Recortar al rango del experimento
    rates = rates.loc[DATE_START:DATE_END]

    # Validación básica
    gaps = rates.isna().sum()
    if gaps.any():
        print(f"[!] NaN tras ffill:\n{gaps[gaps > 0]}")
    else:
        print("Sin gaps tras ffill — OK")

    print(f"\nRango: {rates.index.min().date()} - {rates.index.max().date()}")
    print(f"Observaciones: {len(rates)}")
    print(f"\nPrimeras filas:\n{rates.head()}")
    print(f"\nÚltimas filas:\n{rates.tail()}")
    print(f"\nEstadísticas:\n{rates.describe().round(3).to_string()}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "ecb_rates_monthly.parquet"
    rates.to_parquet(out)
    print(f"\nExportado: {out}")


if __name__ == "__main__":
    main()
