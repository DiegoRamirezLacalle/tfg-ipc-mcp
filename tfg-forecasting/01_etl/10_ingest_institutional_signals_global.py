"""
10_ingest_institutional_signals_global.py — Señales institucionales C1 Global

Descarga 10 señales para el experimento C1_institutional del pipeline CPI Global.
Equivalente a 09_ingest_institutional_signals.py pero con cobertura global.

Señales:
  FRED (pandas_datareader):
    GEPUCURRENT       — Global Economic Policy Uncertainty Index (Davis et al.)
    PALLFNFINDEXM     — IMF All Commodity Price Index (2016=100)
    DTWEXBGS          — Broad Real Dollar Index (USD vs. major currencies)
    VIXCLS            — CBOE VIX (agregado mensual por media)
    DGS10             — 10Y US Treasury yield (condiciones financieras globales)
    FEDFUNDS          — Federal Funds Rate

  Descargas directas:
    GSCPI             — NY Fed Global Supply Chain Pressure Index
    GPR               — Geopolitical Risk Index (Caldara-Iacoviello, Fed)

  Reutilizados de parquets existentes:
    brent_log         — Precio Brent (log), ya en energy_prices_monthly.parquet
    dfr               — ECB Deposit Facility Rate, ya en ecb_rates_monthly.parquet

Transformaciones por señal base: _ma3, _lag1, _diff
Shift +1 en todas las series para evitar leakage temporal.
Rango final: 2002-01-01 a 2024-12-01.

Salida: data/processed/features_c1_global_institutional.parquet
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END

OUTPUT_PATH = ROOT / "data" / "processed" / "features_c1_global_institutional.parquet"
CPI_PATH    = ROOT / "data" / "processed" / "cpi_global_monthly.parquet"

DATE_START = "2001-01-01"  # un mes antes para que shift+1 no pierda 2002-01
DATE_END   = "2024-12-31"
IDX        = pd.date_range("2002-01-01", "2024-12-01", freq="MS")


# ── Utilidades ─────────────────────────────────────────────────────────────

def _monthly_index(s: pd.Series) -> pd.Series:
    """Normaliza índice a freq=MS y ordena."""
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.index = s.index + pd.offsets.MonthBegin(0)  # normalizar a día 1
    return s.sort_index()


def _derive(s: pd.Series, name: str) -> pd.DataFrame:
    """Genera _ma3, _lag1, _diff a partir de una serie base con shift+1."""
    s = s.shift(1)  # evitar leakage: el valor del mes t entra en t+1
    out = pd.DataFrame(index=s.index)
    out[f"{name}_ma3"]  = s.rolling(3).mean()
    out[f"{name}_lag1"] = s.shift(1)
    out[f"{name}_diff"] = s.diff(1)
    return out


# ── Descarga FRED ──────────────────────────────────────────────────────────

def _fred(series_id: str) -> pd.Series:
    """Descarga serie de FRED via URL directa y agrega a mensual (media)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"  FRED {series_id} ...", end=" ", flush=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    val_col  = [c for c in df.columns if c != date_col][0]
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.Series(
        df[val_col].replace(".", np.nan).astype(float).values,
        index=df[date_col]
    ).dropna()
    # Agregar a mensual por media (cubre tanto series diarias como mensuales)
    s = s.resample("MS").mean()
    s = s.reindex(pd.date_range(DATE_START, DATE_END, freq="MS")).ffill()
    print(f"{s.notna().sum()} obs  ({s.first_valid_index().date()} - {s.last_valid_index().date()})")
    return s


def download_fred_signals() -> dict[str, pd.Series]:
    fred_ids = {
        "gepu":        "GEPUCURRENT",
        "imf_comm":    "PALLFNFINDEXM",
        "dxy":         "DTWEXBGS",
        "vix":         "VIXCLS",
        "usg10y":      "DGS10",
        "fedfunds":    "FEDFUNDS",
    }
    return {name: _fred(sid) for name, sid in fred_ids.items()}


# ── GSCPI — NY Fed ────────────────────────────────────────────────────────

def download_gscpi() -> pd.Series:
    url = ("https://www.newyorkfed.org/medialibrary/research/interactives/"
           "gscpi/downloads/gscpi_data.xlsx")
    print(f"  GSCPI (NY Fed) ...", end=" ", flush=True)
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    xls = pd.read_excel(io.BytesIO(r.content), sheet_name=None)
    # Hoja de datos mensuales (la que tiene columna 'Date' y 'GSCPI')
    monthly_sheets = [k for k in xls if "monthly" in k.lower() or "data" in k.lower()]
    df = xls[monthly_sheets[0]] if monthly_sheets else list(xls.values())[-1]

    # Buscar columna de fecha y columna de valor GSCPI
    date_col = [c for c in df.columns if "date" in str(c).lower()]
    val_col  = [c for c in df.columns if "gscpi" in str(c).lower()]

    if not date_col:
        date_col = [df.columns[0]]
    if not val_col:
        val_col = [df.columns[1]]

    s = pd.Series(
        df[val_col[0]].values,
        index=pd.to_datetime(df[date_col[0]])
    ).dropna()
    s = _monthly_index(s)
    print(f"{s.notna().sum()} obs  ({s.index.min().date()} - {s.index.max().date()})")
    return s


# ── GPR — Caldara-Iacoviello ──────────────────────────────────────────────

def download_gpr() -> pd.Series:
    """Intenta varias URLs conocidas del GPR dataset."""
    urls = [
        "https://www.matteoiacoviello.com/gpr_files/gpr_data.xls",
        "https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.xls",
        "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
    ]
    print(f"  GPR (Caldara-Iacoviello) ...", end=" ", flush=True)
    content = None
    for url in urls:
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200:
                content = r.content
                break
        except Exception:
            continue

    if content is None:
        print("FALLBACK: usando GPR=0 (no disponible)")
        return pd.Series(0.0, index=pd.date_range(DATE_START, DATE_END, freq="MS"), name="gpr")

    xls = pd.read_excel(io.BytesIO(content), sheet_name=None)
    df  = list(xls.values())[0]

    # Construir fecha de Year+Month si existe, o buscar columna date-like
    if "Year" in df.columns and "Month" in df.columns:
        df["date"] = pd.to_datetime(
            df["Year"].astype(int).astype(str) + "-" +
            df["Month"].astype(int).astype(str).str.zfill(2) + "-01"
        )
        val_col = [c for c in df.columns if "gpr" in str(c).lower() and "threat" not in str(c).lower()
                   and c not in ("Year","Month","date")]
        s = pd.Series(df[val_col[0]].values, index=df["date"]).dropna()
    else:
        date_col = df.columns[0]
        val_col  = [c for c in df.columns[1:] if "gpr" in str(c).lower()]
        if not val_col:
            val_col = [df.columns[1]]
        s = pd.Series(df[val_col[0]].values, index=pd.to_datetime(df[date_col])).dropna()

    s = _monthly_index(s)
    print(f"{s.notna().sum()} obs  ({s.index.min().date()} - {s.index.max().date()})")
    return s


# ── Reutilizar parquets existentes ────────────────────────────────────────

def load_existing() -> dict[str, pd.Series]:
    energy = pd.read_parquet(ROOT / "data" / "processed" / "energy_prices_monthly.parquet")
    ecb    = pd.read_parquet(ROOT / "data" / "processed" / "ecb_rates_monthly.parquet")
    return {
        "brent_log": energy["brent_log"],
        "dfr":       ecb["dfr"],
    }


# ── Correlaciones ──────────────────────────────────────────────────────────

def print_correlations(features: pd.DataFrame, cpi: pd.Series) -> list[str]:
    """Correlación de cada columna con cpi_global_rate(t+1), tres periodos."""
    target = cpi.shift(-1)  # t+1

    periods = {
        "completo":    (None, None),
        "2015+":       ("2015-01-01", None),
        "shock_2021+": ("2021-01-01", "2023-12-01"),
    }

    sep = "-" * 72
    print(f"\n{sep}")
    print("CORRELACIONES con cpi_global_rate(t+1)")
    print(f"{'Señal':<30} {'Completo':>10} {'2015+':>10} {'Shock':>10}")
    print(sep)

    selected = []
    for col in sorted(features.columns):
        corrs = {}
        for pname, (start, end) in periods.items():
            mask = pd.Series(True, index=target.index)
            if start:
                mask &= target.index >= start
            if end:
                mask &= target.index <= end
            c = features.loc[mask, col].corr(target[mask])
            corrs[pname] = c
        mark = " *" if abs(corrs["completo"]) >= 0.2 else "  "
        print(f"  {col:<28} {corrs['completo']:>10.3f} {corrs['2015+']:>10.3f} "
              f"{corrs['shock_2021+']:>10.3f}{mark}")
        if abs(corrs["completo"]) >= 0.2:
            selected.append(col)

    print(sep)
    print(f"  {len(selected)}/{len(features.columns)} señales con |corr| >= 0.2 (completo)\n")
    return selected


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ETL C1_institutional — CPI Global")
    print("=" * 60)

    # 1. Descarga
    print("\n1. Descargando señales FRED...")
    fred = download_fred_signals()

    print("\n2. Descargando GSCPI (NY Fed)...")
    gscpi = download_gscpi()

    print("\n3. Descargando GPR (Caldara-Iacoviello)...")
    gpr = download_gpr()

    print("\n4. Cargando parquets existentes...")
    existing = load_existing()
    for k, s in existing.items():
        print(f"  {k}: {s.notna().sum()} obs")

    # 2. Construir base dict de series crudas (sin derivar aún)
    raw = {**fred, "gscpi": gscpi, "gpr": gpr, **existing}

    # 3. Alinear todas a IDX mensual 2002-2024
    df_raw = pd.DataFrame(index=IDX)
    for name, s in raw.items():
        aligned = s.reindex(IDX)
        # Fill forward gaps cortos (máx 3 meses) para TTF/GSCPI
        aligned = aligned.ffill(limit=3)
        df_raw[name] = aligned

    print(f"\n5. Series alineadas a {IDX[0].date()} - {IDX[-1].date()}:")
    for col in df_raw.columns:
        missing = df_raw[col].isna().sum()
        print(f"  {col:<15} NaN={missing:>3}  "
              f"({df_raw[col].first_valid_index().date() if df_raw[col].notna().any() else 'todo NaN'})")

    # 4. Derivar: _ma3, _lag1, _diff (con shift+1 incluido en _derive)
    print("\n6. Generando derivadas (_ma3, _lag1, _diff con shift+1)...")
    parts = []
    for name in df_raw.columns:
        parts.append(_derive(df_raw[name], name))

    features = pd.concat(parts, axis=1).reindex(IDX)

    # Rellenar NaN residuales solo en el extremo izquierdo (warmup de ma3/lag)
    features = features.bfill(limit=3).ffill(limit=3)

    print(f"  Total columnas generadas: {len(features.columns)}")
    print(f"  NaN totales: {features.isna().sum().sum()}")

    # 5. Correlaciones + selección
    cpi = pd.read_parquet(CPI_PATH)["cpi_global_rate"].reindex(IDX)
    selected_cols = print_correlations(features, cpi)

    # 6. Añadir la serie objetivo al parquet para comodidad de los modelos
    features["cpi_global_rate"] = cpi

    # 7. Guardar parquet completo (con todas las derivadas, no solo las seleccionadas)
    features.index.name = "date"
    features.to_parquet(OUTPUT_PATH)
    print(f"\nGuardado: {OUTPUT_PATH}")
    print(f"Shape: {features.shape}")
    print(f"\nColumnas seleccionadas para C1_institutional (|corr|>=0.2):")
    for c in selected_cols:
        print(f"  {c}")

    # 8. Guardar también la lista de columnas seleccionadas
    import json
    sel_path = ROOT / "data" / "processed" / "c1_global_inst_selected_cols.json"
    with open(sel_path, "w") as f:
        json.dump(selected_cols, f, indent=2)
    print(f"\nColumnas seleccionadas guardadas en: {sel_path}")

    print("\n" + "=" * 60)
    print("ETL C1_institutional COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
