"""
01_ingest_cpi_global.py — Descarga e ingestion de inflacion mensual global (World Bank)

Fuente:
  World Bank Global Inflation Dataset
  Hoja: hcpi_m (Headline CPI Index, mensual, 186 paises)
  Rango temporal: 2002-01 a 2024-12

Nota sobre el formato:
  La hoja hcpi_m almacena el INDICE de precios al consumo (no la tasa).
  No existe una fila "World" en esta hoja — solo paises individuales.
  La tasa mensual global se calcula como:
    (1) tasa interanual por pais: pct_change(12) sobre el indice
    (2) mediana transversal de todas las tasas disponibles en cada mes
  Esto replica la metodologia del World Bank HCPI_GLOBAL_MED (Aggregate sheet, anual).

  cpi_global_rate es la tasa de variacion interanual (YoY, %) en mediana global.

Salidas:
  data/raw/cpi_global_raw.xlsx                    — Excel original descargado
  data/processed/cpi_global_monthly.parquet       — Serie limpia (date, cpi_global_rate)
  data/snapshots/cpi_global_v1_YYYYMM.parquet     — Snapshot versionado
  data/processed/cpi_global_monthly_check.png     — Plot de verificacion
"""

from __future__ import annotations

import io
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids charmap encode errors in prints)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"

SOURCE_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "1ad246272dbbc437c74323719506aa0c-0350012021/original/Inflation-data.xlsx"
)
SHEET      = "hcpi_m"
DATE_START = "2002-01-01"
DATE_END   = "2024-12-31"   # inclusive


# ── Descarga ─────────────────────────────────────────────────────────────────

def download_excel(url: str, dest: Path) -> bytes:
    print(f"  Descargando {url} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    print(f"  Guardado: {dest}  ({len(r.content)/1024:.0f} KB)")
    return r.content


# ── Parseo ────────────────────────────────────────────────────────────────────

def load_hcpi_m(content: bytes) -> pd.DataFrame:
    """
    Carga la hoja hcpi_m y devuelve un DataFrame wide:
      filas = paises (Country Code como indice)
      columnas = pd.Timestamp (primer dia del mes)
    """
    df = pd.read_excel(io.BytesIO(content), sheet_name=SHEET, engine="openpyxl")

    # Columnas de fecha: enteros/floats con formato YYYYMM (e.g. 197001)
    date_cols = {}
    for col in df.columns:
        if isinstance(col, (int, float)) and not np.isnan(float(col)):
            ival = int(col)
            year, month = divmod(ival, 100)
            if 1950 <= year <= 2030 and 1 <= month <= 12:
                date_cols[col] = pd.Timestamp(year, month, 1)

    # Filtrar solo columnas de fecha
    meta_cols = ["Country Code", "IMF Country Code", "Country",
                 "Indicator Type", "Series Name"]
    date_orig  = list(date_cols.keys())

    # Excluir ultima fila (nota al pie)
    df = df[df["Country Code"].notna() &
            ~df["Country Code"].astype(str).str.lower().str.startswith("note")]

    df_wide = df.set_index("Country Code")[date_orig].copy()
    df_wide.columns = [date_cols[c] for c in date_orig]
    df_wide = df_wide.astype(float)

    print(f"  Paises: {len(df_wide)}  |  "
          f"Fechas: {df_wide.columns.min().date()} a {df_wide.columns.max().date()}")
    return df_wide


def compute_global_rate(df_wide: pd.DataFrame,
                        date_start: str, date_end: str) -> pd.Series:
    """
    Para cada pais: tasa interanual = pct_change(12) sobre el indice mensual.
    Tasa global = mediana transversal de todos los paises con dato disponible.
    Filtra al rango [date_start, date_end].
    """
    # YoY para cada pais: (index_t / index_t-12 - 1) * 100
    yoy = df_wide.pct_change(periods=12, axis=1) * 100  # filas=paises, cols=fechas

    # Trasponer a (fecha, pais)
    yoy_T = yoy.T
    yoy_T.index = pd.DatetimeIndex(yoy_T.index)
    yoy_T.index.freq = "MS"

    # Mediana transversal
    global_med = yoy_T.median(axis=1, skipna=True)
    global_med.name = "cpi_global_rate"

    # Filtrar rango
    mask = (global_med.index >= date_start) & (global_med.index <= date_end)
    series = global_med[mask].dropna()

    # Covertura: cuantos paises tienen dato en cada mes
    n_countries = yoy_T[mask].notna().sum(axis=1)
    print(f"  Cobertura media por mes: {n_countries.mean():.0f} paises")
    print(f"  Cobertura min/max: {n_countries.min()}/{n_countries.max()} paises")

    return series


# ── Estadisticas ──────────────────────────────────────────────────────────────

def print_stats(series: pd.Series) -> None:
    print("\n" + "─" * 52)
    print("ESTADISTICAS — cpi_global_rate (YoY mediana, %)")
    print("─" * 52)
    print(f"  Rango fechas : {series.index.min().date()} → {series.index.max().date()}")
    print(f"  Observaciones: {len(series)}")
    print(f"  NaN count    : {series.isna().sum()}")
    print(f"  Media        : {series.mean():.4f} %")
    print(f"  Mediana      : {series.median():.4f} %")
    print(f"  Min          : {series.min():.4f} %  ({series.idxmin().date()})")
    print(f"  Max          : {series.max():.4f} %  ({series.idxmax().date()})")
    print(f"  Std          : {series.std():.4f} %")
    print("─" * 52)
    # Gaps
    expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    missing = expected.difference(series.index)
    print(f"  Gaps: {'ninguno' if len(missing) == 0 else missing.tolist()}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_series(series: pd.Series, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        "Global Monthly Inflation Rate — World Bank hcpi_m\n"
        "(mediana interanual de 186 paises, replica HCPI_GLOBAL_MED)",
        fontsize=12, fontweight="bold", y=0.99)

    SHADING = [
        ("Crisis financiera", "2008-09-01", "2009-06-30", "#fff3cd", 0.55),
        ("Covid-19",          "2020-01-01", "2020-12-31", "#e8e8e8", 0.50),
        ("Shock inflac.",     "2021-01-01", "2022-12-31", "#f8d7d7", 0.55),
        ("Normalizacion",     "2023-01-01", "2024-12-31", "#d7e8f8", 0.40),
    ]

    # ── Panel superior: serie ────────────────────────────────────────────────
    ax = axes[0]
    for label, s, e, color, alpha in SHADING:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=color, alpha=alpha, zorder=0, label=label)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.plot(series.index, series.values,
            color="#2166ac", linewidth=1.8, zorder=3, label="Inflacion global (YoY, %)")
    ax.scatter(series.idxmax(), series.max(), color="#d62728", s=70, zorder=5,
               label=f"Max: {series.max():.2f}% ({series.idxmax().strftime('%Y-%m')})")
    ax.scatter(series.idxmin(), series.min(), color="#1f77b4", s=70,
               marker="v", zorder=5,
               label=f"Min: {series.min():.2f}% ({series.idxmin().strftime('%Y-%m')})")
    ax.set_ylabel("Tasa de inflacion YoY (%)", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_xlim(series.index.min(), series.index.max())
    ax.grid(axis="y", alpha=0.3)

    # ── Panel inferior: volatilidad rolling ──────────────────────────────────
    ax2 = axes[1]
    roll_std = series.rolling(12, min_periods=6).std()
    ax2.fill_between(roll_std.index, roll_std.values, color="#9ecae1", alpha=0.7)
    ax2.plot(roll_std.index, roll_std.values, color="#2166ac", linewidth=1.0)
    ax2.set_ylabel("Std rolling 12m", fontsize=9)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_xlim(series.index.min(), series.index.max())
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot guardado: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("INGESTA CPI GLOBAL — World Bank hcpi_m (186 paises)")
    print("=" * 60)

    # 1. Descarga
    raw_path = RAW_DIR / "cpi_global_raw.xlsx"
    content  = download_excel(SOURCE_URL, raw_path)

    # 2. Cargar hoja como matriz paises x fechas
    print("\nCargando hoja hcpi_m...")
    df_wide = load_hcpi_m(content)

    # 3. Calcular tasa global (mediana YoY)
    print("\nCalculando tasa interanual global (mediana transversal)...")
    series = compute_global_rate(df_wide, DATE_START, DATE_END)

    # 4. Estadisticas
    print_stats(series)

    # 5. DataFrame de salida
    df_out = series.rename("cpi_global_rate").to_frame()
    df_out.index.name = "date"
    print(f"\n  Primeras filas:\n{df_out.head(4).to_string()}")
    print(f"  ...\n  Ultimas filas:\n{df_out.tail(4).to_string()}")

    # 6. Guardar processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    proc_path = PROCESSED_DIR / "cpi_global_monthly.parquet"
    df_out.to_parquet(proc_path)
    print(f"\n  Guardado processed: {proc_path}")

    # 7. Snapshot versionado
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime("%Y%m")
    snap_path = SNAPSHOTS_DIR / f"cpi_global_v1_{version}.parquet"
    df_out.to_parquet(snap_path)
    print(f"  Guardado snapshot:  {snap_path}")

    # 8. Plot
    plot_series(series, PROCESSED_DIR / "cpi_global_monthly_check.png")

    print("\n" + "=" * 60)
    print("ETL completado.")
    print("=" * 60)


if __name__ == "__main__":
    main()
