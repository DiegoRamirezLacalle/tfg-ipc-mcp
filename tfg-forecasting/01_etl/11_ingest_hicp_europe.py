"""
11_ingest_hicp_europe.py
------------------------
Descarga el HICP Eurozone (índice en nivel, base 2015=100) del BCE vía SDMX.

Serie: ICP.M.U2.N.000000.4.INX
  M       = mensual
  U2      = Zona Euro (composición variable)
  N       = no ajustado estacionalmente
  000000  = Todos los artículos (índice general)
  4/INX   = índice en nivel (no tasa de variación)

Rango objetivo: 2002-01 a 2024-12.

Salidas:
  data/raw/hicp_europe_raw.csv
  data/processed/hicp_europe_index.parquet   (columnas: date, hicp_index)
  data/snapshots/hicp_europe_v1_<YYYYMM>.parquet

Verificación inline:
  - Sin NaN ni gaps mensuales
  - Valor ~100 en 2015 (base del índice)
  - Pico visible en 2022-2023
  - Plot guardado en data/processed/hicp_europe_check.png
"""

from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
RAW_CSV   = ROOT / "data" / "raw" / "hicp_europe_raw.csv"
PROC_PQ   = ROOT / "data" / "processed" / "hicp_europe_index.parquet"
SNAP_DIR  = ROOT / "data" / "snapshots"
PLOT_OUT  = ROOT / "data" / "processed" / "hicp_europe_check.png"

ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/ICP/"
    "M.U2.N.000000.4.INX?format=csvdata"
)
START = "2002-01"
END   = "2024-12"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TFG-Academic-Research/1.0)",
    "Accept": "text/csv",
}


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------

def download_raw() -> str:
    print(f"Descargando HICP Eurozone desde BCE SDMX...")
    r = httpx.get(ECB_URL, headers=HEADERS, timeout=60, follow_redirects=True)
    r.raise_for_status()
    print(f"  HTTP {r.status_code} — {len(r.text):,} caracteres recibidos")
    return r.text


# ---------------------------------------------------------------------------
# Parseo
# ---------------------------------------------------------------------------

def parse_series(csv_text: str) -> pd.DataFrame:
    df_raw = pd.read_csv(StringIO(csv_text))

    df = (
        df_raw[["TIME_PERIOD", "OBS_VALUE"]]
        .copy()
        .rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "hicp_index"})
    )

    df["date"] = pd.to_datetime(df["date"].str.strip() + "-01")
    df["hicp_index"] = pd.to_numeric(df["hicp_index"], errors="coerce")
    df = df.dropna(subset=["hicp_index"]).sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Filtrado al rango objetivo
# ---------------------------------------------------------------------------

def filter_range(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(START + "-01")) & (
        df["date"] <= pd.Timestamp(END + "-01")
    )
    df_f = df[mask].copy().reset_index(drop=True)
    print(f"  Filas en rango {START} – {END}: {len(df_f)}")
    return df_f


# ---------------------------------------------------------------------------
# Verificación de calidad
# ---------------------------------------------------------------------------

def verify(df: pd.DataFrame) -> None:
    print("\n--- Verificación de calidad ---")

    # 1. Sin NaN
    nans = df["hicp_index"].isna().sum()
    print(f"  NaN: {nans}  {'OK' if nans == 0 else 'ADVERTENCIA'}")

    # 2. Cobertura temporal esperada: 276 meses (2002-01 a 2024-12)
    expected_months = (
        pd.date_range(start=START + "-01", end=END + "-01", freq="MS")
    )
    actual_months = set(df["date"])
    missing = [d for d in expected_months if d not in actual_months]
    print(f"  Meses esperados: {len(expected_months)}  |  Encontrados: {len(df)}")
    if missing:
        print(f"  GAPS detectados ({len(missing)}): {missing[:5]}...")
    else:
        print("  Sin gaps mensuales: OK")

    # 3. Valor base 2015
    df_2015 = df[df["date"].dt.year == 2015]["hicp_index"]
    mean_2015 = df_2015.mean()
    print(f"  Media 2015: {mean_2015:.2f}  (esperado ~100.0)")

    # 4. Pico 2022-2023
    peak_row = df.loc[df["hicp_index"].idxmax()]
    print(f"  Pico máximo: {peak_row['hicp_index']:.2f}  en {peak_row['date'].strftime('%Y-%m')}")

    # 5. Estadísticas básicas
    print("\n--- Estadísticas básicas ---")
    print(df["hicp_index"].describe().round(3).to_string())
    print(f"\n  Primer valor: {df.iloc[0]['date'].strftime('%Y-%m')} = {df.iloc[0]['hicp_index']:.2f}")
    print(f"  Último valor: {df.iloc[-1]['date'].strftime('%Y-%m')} = {df.iloc[-1]['hicp_index']:.2f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_series(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(df["date"], df["hicp_index"], color="#1565c0", linewidth=1.5, label="HICP Eurozone")
    ax.axhline(100, color="#888", linewidth=0.8, linestyle="--", label="Base 2015=100")

    # Marcar pico
    idx_max = df["hicp_index"].idxmax()
    ax.annotate(
        f"Pico: {df.loc[idx_max,'hicp_index']:.1f}\n{df.loc[idx_max,'date'].strftime('%Y-%m')}",
        xy=(df.loc[idx_max, "date"], df.loc[idx_max, "hicp_index"]),
        xytext=(30, -18), textcoords="offset points",
        fontsize=8, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8),
    )

    ax.set_title("HICP Eurozone — Índice en nivel (base 2015=100)", fontsize=12)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Índice HICP")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=130)
    plt.close(fig)
    print(f"\n  Plot guardado: {PLOT_OUT}")


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame) -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    PROC_PQ.parent.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    # Raw CSV
    df.to_csv(RAW_CSV, index=False)
    print(f"\n  Raw CSV:   {RAW_CSV}")

    # Processed parquet (date como columna, no como índice — consistente con ipc_spain)
    df.to_parquet(PROC_PQ, index=False)
    print(f"  Processed: {PROC_PQ}")

    # Snapshot versionado
    yyyymm = date.today().strftime("%Y%m")
    snap_path = SNAP_DIR / f"hicp_europe_v1_{yyyymm}.parquet"
    df.to_parquet(snap_path, index=False)
    print(f"  Snapshot:  {snap_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("ETL — HICP Eurozone (BCE SDMX)")
    print("=" * 55)

    csv_text = download_raw()

    df_all = parse_series(csv_text)
    print(f"  Serie completa: {df_all['date'].min().strftime('%Y-%m')} "
          f"a {df_all['date'].max().strftime('%Y-%m')}  ({len(df_all)} obs)")

    # Guardar raw completo antes de filtrar
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(RAW_CSV, index=False)

    df = filter_range(df_all)

    verify(df)
    plot_series(df)
    save(df)

    print("\n" + "=" * 55)
    print("ETL completado.")
    print("=" * 55)


if __name__ == "__main__":
    main()
