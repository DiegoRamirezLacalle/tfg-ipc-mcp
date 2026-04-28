"""
12_ingest_europe_signals.py — Seniales macro especificas Europa para C1

Descarga tres series de FRED y genera derivadas:

  ESIEZM    — ESI Eurozone Economic Sentiment Indicator (Comision Europea)
              Fallback: BSCICP03EZM665S (OECD BCI Euro Area)
  T5YIE     — 5-Year Breakeven Inflation Rate (expectativas de mercado)
  DEXUSEU   — EUR/USD tipo de cambio diario -> media mensual

Derivadas por serie: raw, ma3, lag1, diff.
Shift +1 en todas para evitar leakage (valor disponible en t usado en t+1).

Rango: 2002-01 a 2024-12 (276 meses, alineado con HICP Europa).

Salida:
  data/raw/europe_signals_raw.csv
  data/processed/europe_signals_monthly.parquet
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

DATE_START = "2002-01-01"
DATE_END   = "2024-12-31"
FREQ       = "MS"
OUT_RAW    = ROOT / "data" / "raw" / "europe_signals_raw.csv"
OUT_PROC   = ROOT / "data" / "processed" / "europe_signals_monthly.parquet"


def fetch_fred(series_id: str, fallback_id: str | None = None) -> pd.Series | None:
    """Descarga serie de FRED via pandas_datareader. Intenta fallback si falla."""
    try:
        import pandas_datareader as pdr
        s = pdr.get_data_fred(series_id, start=DATE_START, end=DATE_END)
        s = s.squeeze()
        print(f"  [FRED] {series_id}: {len(s)} obs ({s.index.min().date()} - {s.index.max().date()})")
        return s
    except Exception as e:
        print(f"  [FRED] {series_id} fallo: {e}")
        if fallback_id:
            try:
                import pandas_datareader as pdr
                s = pdr.get_data_fred(fallback_id, start=DATE_START, end=DATE_END)
                s = s.squeeze()
                print(f"  [FRED] fallback {fallback_id}: {len(s)} obs")
                return s
            except Exception as e2:
                print(f"  [FRED] fallback {fallback_id} tambien fallo: {e2}")
        return None


def to_monthly(s: pd.Series, method: str = "last") -> pd.Series:
    """Convierte serie a frecuencia mensual (inicio de mes)."""
    if method == "mean":
        m = s.resample("MS").mean()
    else:
        m = s.resample("MS").last()
    return m


def make_features(s: pd.Series, name: str) -> pd.DataFrame:
    """Genera raw, ma3, lag1, diff con shift +1 para evitar leakage."""
    df = pd.DataFrame(index=s.index)
    raw = s.copy()
    df[name]          = raw.shift(1)                     # shift evita leakage
    df[f"{name}_ma3"] = raw.rolling(3).mean().shift(1)
    df[f"{name}_lag1"] = raw.shift(2)                    # lag1 sobre shifted = lag2 del raw
    df[f"{name}_diff"] = raw.diff(1).shift(1)
    return df


def main():
    print("=" * 60)
    print("ETL EUROPE SIGNALS — ESI, Breakeven 5Y, EUR/USD")
    print("=" * 60)

    target_idx = pd.date_range(start=DATE_START, end="2024-12-01", freq=FREQ)
    frames = []
    raw_dict = {}

    # ── 1. ESI Eurozone ─────────────────────────────────────────────
    print("\n[1] ESI Eurozone (ESIEZM / fallback BSCICP03EZM665S)")
    esi = fetch_fred("ESIEZM", fallback_id="BSCICP03EZM665S")
    if esi is not None:
        esi_m = to_monthly(esi, method="last").reindex(target_idx)
        esi_m = esi_m.ffill().bfill()
        raw_dict["esi_eurozone_raw"] = esi_m
        feats = make_features(esi_m, "esi_eurozone")
        frames.append(feats)
        print(f"  ESI Eurozone: {esi_m.dropna().shape[0]} obs validos, "
              f"rango {esi_m.min():.1f}-{esi_m.max():.1f}")
    else:
        print("  [!] ESI Eurozone no disponible — columnas con NaN")
        null = pd.DataFrame(np.nan, index=target_idx,
                            columns=["esi_eurozone", "esi_eurozone_ma3",
                                     "esi_eurozone_lag1", "esi_eurozone_diff"])
        frames.append(null)

    # ── 2. 5Y Breakeven Inflation ────────────────────────────────────
    print("\n[2] 5Y Breakeven Inflation Rate (T5YIE)")
    brk = fetch_fred("T5YIE")
    if brk is not None:
        brk_m = to_monthly(brk, method="mean").reindex(target_idx)
        brk_m = brk_m.ffill().bfill()
        raw_dict["breakeven_5y_raw"] = brk_m
        feats = make_features(brk_m, "breakeven_5y")
        frames.append(feats)
        print(f"  Breakeven 5Y: {brk_m.dropna().shape[0]} obs validos, "
              f"rango {brk_m.min():.2f}%-{brk_m.max():.2f}%")
    else:
        print("  [!] Breakeven 5Y no disponible — columnas con NaN")
        null = pd.DataFrame(np.nan, index=target_idx,
                            columns=["breakeven_5y", "breakeven_5y_ma3",
                                     "breakeven_5y_lag1", "breakeven_5y_diff"])
        frames.append(null)

    # ── 3. EUR/USD ───────────────────────────────────────────────────
    print("\n[3] EUR/USD tipo de cambio diario -> media mensual (DEXUSEU)")
    eur = fetch_fred("DEXUSEU")
    if eur is not None:
        eur_m = to_monthly(eur, method="mean").reindex(target_idx)
        eur_m = eur_m.ffill().bfill()
        raw_dict["eurusd_raw"] = eur_m
        feats = make_features(eur_m, "eurusd")
        frames.append(feats)
        print(f"  EUR/USD: {eur_m.dropna().shape[0]} obs validos, "
              f"rango {eur_m.min():.4f}-{eur_m.max():.4f}")
    else:
        print("  [!] EUR/USD no disponible — columnas con NaN")
        null = pd.DataFrame(np.nan, index=target_idx,
                            columns=["eurusd", "eurusd_ma3",
                                     "eurusd_lag1", "eurusd_diff"])
        frames.append(null)

    # ── Combinar ─────────────────────────────────────────────────────
    df_out = pd.concat(frames, axis=1)
    df_out.index.name = "date"
    df_out.index.freq = FREQ

    n_nan = df_out.isna().sum().sum()
    print(f"\nDataFrame final: {df_out.shape}  |  NaN totales: {n_nan}")
    print(df_out.head(3).to_string())

    # ── Guardar ──────────────────────────────────────────────────────
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # Raw CSV con las series originales mensuales
    if raw_dict:
        pd.DataFrame(raw_dict).to_csv(OUT_RAW)
        print(f"\nRaw guardado: {OUT_RAW}")

    df_out.to_parquet(OUT_PROC)
    print(f"Procesado guardado: {OUT_PROC}")

    print("\nColumnas generadas:")
    for c in df_out.columns:
        n_valid = df_out[c].notna().sum()
        print(f"  {c:<30} {n_valid:>4}/{len(df_out)} validos")

    print("\n" + "=" * 60)
    print("LISTO — europe_signals_monthly.parquet generado")
    print("=" * 60)


if __name__ == "__main__":
    main()
