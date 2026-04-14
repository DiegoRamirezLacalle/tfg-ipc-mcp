"""
09_ingest_institutional_signals.py — Descarga EPU Spain, ESI Spain, EPU Europe

Fuentes:
  EPU Spain   — policyuncertainty.com (Spain_News_Index, Baker et al.)
  EPU Europe  — policyuncertainty.com (European_News_Index)
  ESI Spain   — Eurostat ei_bssi_m_r2 (EC Economic Sentiment Indicator, SA)

Transformaciones por serie:
  _log   — log(x) para normalizar escala
  _diff  — diferencia mensual (momentum)
  _ma3   — media movil 3 meses (suavizada)
  _lag1  — lag 1 mes

Todas con shift +1 para evitar leakage temporal.
Datos disponibles desde 2002 sin sparsity.
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

OUTPUT_PATH = ROOT / "data" / "processed" / "institutional_signals_monthly.parquet"
IPC_PATH = ROOT / "data" / "processed" / "ipc_spain_index.parquet"


# ── Descarga ─────────────────────────────────────────────────────

def download_epu() -> pd.DataFrame:
    """Descarga EPU Spain y EPU Europe de policyuncertainty.com."""
    url = "https://policyuncertainty.com/media/Europe_Policy_Uncertainty_Data.xlsx"
    print(f"  Descargando EPU Europe (incluye Spain_News_Index)...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = pd.read_excel(io.BytesIO(r.content))

    # Limpiar fila de source al final
    df = df[df["Year"].apply(lambda x: str(x).replace(".", "").isdigit())]
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.set_index("date").sort_index()
    df.index.freq = "MS"

    result = pd.DataFrame(index=df.index)
    result["epu_spain"] = df["Spain_News_Index"].astype(float)
    result["epu_europe"] = df["European_News_Index"].astype(float)

    print(f"    EPU Spain:  {result['epu_spain'].first_valid_index().date()} - "
          f"{result['epu_spain'].last_valid_index().date()}, "
          f"{result['epu_spain'].notna().sum()} valid")
    print(f"    EPU Europe: {result['epu_europe'].first_valid_index().date()} - "
          f"{result['epu_europe'].last_valid_index().date()}, "
          f"{result['epu_europe'].notna().sum()} valid")

    return result


def download_esi() -> pd.Series:
    """Descarga ESI Spain de Eurostat (ei_bssi_m_r2, seasonally adjusted)."""
    url = ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/"
           "data/ei_bssi_m_r2?format=SDMX-CSV&sinceTimePeriod=1985M01")
    print(f"  Descargando ESI Spain de Eurostat...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))

    # Filtrar: ESI, Spain, seasonally adjusted
    esi = df[
        (df["indic"] == "BS-ESI-I") &
        (df["geo"] == "ES") &
        (df["s_adj"] == "SA")
    ].copy()

    esi["date"] = pd.to_datetime(esi["TIME_PERIOD"])
    esi = esi.set_index("date").sort_index()

    s = esi["OBS_VALUE"].astype(float)
    s.name = "esi_spain"
    s.index.freq = "MS"

    print(f"    ESI Spain:  {s.first_valid_index().date()} - "
          f"{s.last_valid_index().date()}, {s.notna().sum()} valid")

    return s


# ── Transformaciones ─────────────────────────────────────────────

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Genera derivadas y aplica shift +1."""
    result = pd.DataFrame(index=df.index)

    for col in ["epu_spain", "epu_europe", "esi_spain"]:
        if col not in df.columns:
            continue
        s = df[col].copy()
        s = s.ffill(limit=2)  # Forward-fill gaps < 3 meses

        # Log
        result[f"{col}_log"] = np.log(s.clip(lower=0.01))

        # Diferencia mensual
        result[f"{col}_diff"] = s.diff()

        # Media movil 3 meses
        result[f"{col}_ma3"] = s.rolling(3, min_periods=2).mean()

        # Lag 1
        result[f"{col}_lag1"] = s.shift(1)

    # Shift +1 global para evitar leakage
    result = result.shift(1)

    return result


# ── Correlaciones ────────────────────────────────────────────────

def correlate_with_ipc(df: pd.DataFrame) -> pd.DataFrame:
    ipc = pd.read_parquet(IPC_PATH)
    y = ipc["indice_general"]
    y.index = pd.to_datetime(y.index)
    y.index.freq = "MS"
    y_lead = y.shift(-1).dropna()

    corrs = {}
    for col in df.columns:
        common = df[col].dropna().index.intersection(y_lead.index)
        row = {}

        if len(common) > 24:
            row["corr_full"] = round(float(df[col].reindex(common).corr(y_lead.reindex(common))), 4)

        common_2015 = common[common >= "2015-01-01"]
        if len(common_2015) > 12:
            row["corr_2015"] = round(
                float(df[col].reindex(common_2015).corr(y_lead.reindex(common_2015))), 4
            )

        common_2002 = common[common >= "2002-01-01"]
        if len(common_2002) > 24:
            row["corr_2002"] = round(
                float(df[col].reindex(common_2002).corr(y_lead.reindex(common_2002))), 4
            )

        if row:
            corrs[col] = row

    return pd.DataFrame(corrs).T


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DESCARGA SENALES INSTITUCIONALES")
    print("=" * 60)

    # 1. Descargar
    epu = download_epu()
    esi = download_esi()

    # Combinar
    raw = epu.copy()
    raw["esi_spain"] = esi.reindex(raw.index)
    print(f"\nSeries brutas combinadas: {raw.shape}")

    # 2. Transformar
    df = transform(raw)
    print(f"Transformadas (con shift+1): {df.shape}")
    print(f"Rango: {df.index.min().date()} - {df.index.max().date()}")
    print(f"\nNaN por columna:")
    for col in df.columns:
        n_valid = df[col].notna().sum()
        first_valid = df[col].first_valid_index()
        print(f"  {col}: {n_valid} valid, first={first_valid.date() if first_valid else 'N/A'}")

    # 3. Correlaciones
    print("\n" + "=" * 60)
    print("CORRELACIONES CON IPC(t+1)")
    print("=" * 60)
    corr_df = correlate_with_ipc(df)
    print(corr_df.to_string())

    # Gate check
    max_abs_2015 = corr_df["corr_2015"].abs().max() if "corr_2015" in corr_df.columns else 0
    max_abs_full = corr_df["corr_full"].abs().max() if "corr_full" in corr_df.columns else 0
    print(f"\nMax |corr| 2015+: {max_abs_2015:.4f}")
    print(f"Max |corr| full:  {max_abs_full:.4f}")

    if max_abs_2015 < 0.3 and max_abs_full < 0.3:
        print("\n[!] TODAS las correlaciones < 0.3. Consultar antes de lanzar modelos.")
    else:
        print("\n[OK] Correlaciones suficientes para continuar.")

    # Top 5 por |corr_2015|
    if "corr_2015" in corr_df.columns:
        top5 = corr_df["corr_2015"].abs().sort_values(ascending=False).head(5)
        print("\nTop 5 por |corr_2015|:")
        for col, val in top5.items():
            sign = corr_df.loc[col, "corr_2015"]
            print(f"  {col}: {sign:+.4f}")

    # 4. Guardar (recortar a 2002+)
    df_save = df.loc["2002-01-01":].copy()
    df_save = df_save.reset_index()
    if df_save.columns[0] != "date":
        df_save = df_save.rename(columns={df_save.columns[0]: "date"})

    ROOT.joinpath("data", "processed").mkdir(parents=True, exist_ok=True)
    df_save.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nGuardado: {OUTPUT_PATH}")
    print(f"Shape: {df_save.shape}")


if __name__ == "__main__":
    main()
