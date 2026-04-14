"""
07_ingest_energy_prices.py — Descarga y procesamiento de precios energeticos

Series:
  1. Brent Crude Oil (USD/barril) — BZ=F desde 2007-08, proxy WTI pre-2007
  2. Gas Natural TTF europeo (EUR/MWh) — TTF=F desde 2017-10, proxy HH pre-2017

Transformaciones por serie (4 cada una, total 8 columnas):
  - _log:  log del precio (estabilizar varianza)
  - _ret:  primera diferencia del log (retorno mensual)
  - _ma3:  media movil 3 meses del log (suavizar ruido)
  - _lag1: lag 1 mes del log (precio conocido antes de IPC)

Todas con shift +1 para evitar leakage: el precio del mes t entra en t+1.

Salida: data/processed/energy_prices_monthly.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

DATE_START = "2001-01-01"  # Descargar desde 2001 para que tras shift+1 haya datos en 2002-01
DATE_END = "2025-07-01"


def download_series(ticker: str, name: str) -> pd.Series:
    """Descarga cierre mensual de Yahoo Finance."""
    print(f"  Descargando {name} ({ticker})...")
    df = yf.download(ticker, start=DATE_START, end=DATE_END,
                     interval="1mo", progress=False)
    if df.empty:
        raise RuntimeError(f"No hay datos para {ticker}")

    # yfinance MultiIndex: (Price, Ticker)
    close = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    close.index = pd.to_datetime(close.index)
    close.name = name
    return close.dropna()


def build_proxy(primary: pd.Series, proxy: pd.Series, name: str) -> pd.Series:
    """
    Extiende primary hacia atras usando proxy escalado.
    En el periodo de overlap, calcula ratio medio primary/proxy,
    y aplica ese ratio al proxy para el periodo pre-primary.
    """
    overlap = primary.index.intersection(proxy.index)
    if len(overlap) < 12:
        print(f"  [!] Overlap {name}: solo {len(overlap)} meses, usando proxy directo")
        ratio = 1.0
    else:
        ratio = float((primary.loc[overlap] / proxy.loc[overlap]).median())
        print(f"  Ratio {name}: {ratio:.3f} (mediana sobre {len(overlap)} meses overlap)")

    # Periodo anterior al inicio de primary: proxy * ratio
    pre_dates = proxy.index[proxy.index < primary.index.min()]
    pre_vals = proxy.loc[pre_dates] * ratio

    # Combinar: proxy escalado + primary real
    combined = pd.concat([pre_vals, primary]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.name = name
    return combined


def transform_series(s: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Aplica las 4 transformaciones a una serie de precios.
    """
    log_s = np.log(s.clip(lower=0.01))  # clip para evitar log(0)

    df = pd.DataFrame(index=s.index)
    df[f"{prefix}_log"] = log_s
    df[f"{prefix}_ret"] = log_s.diff()
    df[f"{prefix}_ma3"] = log_s.rolling(3, min_periods=1).mean()
    df[f"{prefix}_lag1"] = log_s.shift(1)

    return df


def main():
    print("=" * 60)
    print("INGESTA PRECIOS ENERGETICOS")
    print("=" * 60)

    # ── Descargar series ──────────────────────────────────────────
    brent = download_series("BZ=F", "brent")
    wti = download_series("CL=F", "wti")
    ttf = download_series("TTF=F", "ttf")
    hh = download_series("NG=F", "henry_hub")

    print(f"\n  Brent: {brent.index.min().date()} - {brent.index.max().date()} ({len(brent)} obs)")
    print(f"  WTI:   {wti.index.min().date()} - {wti.index.max().date()} ({len(wti)} obs)")
    print(f"  TTF:   {ttf.index.min().date()} - {ttf.index.max().date()} ({len(ttf)} obs)")
    print(f"  HH:    {hh.index.min().date()} - {hh.index.max().date()} ({len(hh)} obs)")

    # ── Construir series completas con proxy ──────────────────────
    print("\nConstruyendo Brent completo (proxy: WTI pre-2007)...")
    brent_full = build_proxy(brent, wti, "brent")

    print("Construyendo TTF completo (proxy: Henry Hub pre-2017)...")
    ttf_full = build_proxy(ttf, hh, "ttf")

    print(f"\n  Brent completo: {brent_full.index.min().date()} - {brent_full.index.max().date()} ({len(brent_full)})")
    print(f"  TTF completo:   {ttf_full.index.min().date()} - {ttf_full.index.max().date()} ({len(ttf_full)})")

    # ── Transformaciones ──────────────────────────────────────────
    print("\nAplicando transformaciones (log, ret, ma3, lag1)...")
    df_brent = transform_series(brent_full, "brent")
    df_ttf = transform_series(ttf_full, "ttf")

    # Merge
    df = df_brent.join(df_ttf, how="outer")

    # ── Shift +1 (leakage guard): precio mes t -> exogena mes t+1 ─
    df = df.shift(1)

    # Alinear al rango del proyecto: 2002-01 a 2025-06
    target_idx = pd.date_range("2002-01-01", "2025-06-01", freq="MS")
    df = df.reindex(target_idx)
    df.index.name = "date"
    df.index.freq = "MS"

    # ── Info ──────────────────────────────────────────────────────
    print(f"\nDataset final:")
    print(f"  Rango: {df.index.min().date()} - {df.index.max().date()}")
    print(f"  Shape: {df.shape}")
    print(f"  Columnas: {list(df.columns)}")
    print(f"\n  NaN por columna:")
    print(df.isna().sum().to_string())
    print(f"\n  Primeras filas (post-shift):\n{df.head(5)}")
    print(f"\n  Ultimas filas:\n{df.tail(3)}")

    # ── Guardar ──────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "energy_prices_monthly.parquet"
    df.to_parquet(out)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
