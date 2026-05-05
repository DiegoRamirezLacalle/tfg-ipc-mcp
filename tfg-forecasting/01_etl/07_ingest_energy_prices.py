"""Download and process monthly energy price series.

Series:
  1. Brent Crude Oil (USD/barrel) - BZ=F from 2007-08, WTI proxy pre-2007
  2. European Natural Gas TTF (EUR/MWh) - TTF=F from 2017-10, Henry Hub proxy pre-2017

Transformations per series (4 each, 8 columns total):
  _log:  log price (variance stabilisation)
  _ret:  log return (monthly first difference)
  _ma3:  3-month rolling mean of log (noise smoothing)
  _lag1: 1-month lag of log (price known before CPI release)

All features shifted +1 to prevent leakage: month-t price enters at t+1.

Output: data/processed/energy_prices_monthly.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed"

DATE_START = "2001-01-01"  # one month before target so shift+1 covers 2002-01
DATE_END   = "2025-07-01"


def download_series(ticker: str, name: str) -> pd.Series:
    """Download monthly close prices from Yahoo Finance."""
    logger.info(f"  Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=DATE_START, end=DATE_END, interval="1mo", progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")

    close = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    close.index = pd.to_datetime(close.index)
    close.name = name
    return close.dropna()


def build_proxy(primary: pd.Series, proxy: pd.Series, name: str) -> pd.Series:
    """Extend primary backwards using a scaled proxy series.

    Computes the median ratio over the overlap period and applies it to
    the proxy for dates before primary starts.
    """
    overlap = primary.index.intersection(proxy.index)
    if len(overlap) < 12:
        logger.warning(f"  {name} overlap: only {len(overlap)} months, using proxy directly")
        ratio = 1.0
    else:
        ratio = float((primary.loc[overlap] / proxy.loc[overlap]).median())
        logger.info(f"  Ratio {name}: {ratio:.3f} (median over {len(overlap)} months overlap)")

    pre_dates = proxy.index[proxy.index < primary.index.min()]
    pre_vals  = proxy.loc[pre_dates] * ratio

    combined = pd.concat([pre_vals, primary]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.name = name
    return combined


def transform_series(s: pd.Series, prefix: str) -> pd.DataFrame:
    """Apply log, return, 3m MA, and lag-1 transformations to a price series."""
    log_s = np.log(s.clip(lower=0.01))  # clip to avoid log(0)

    df = pd.DataFrame(index=s.index)
    df[f"{prefix}_log"]  = log_s
    df[f"{prefix}_ret"]  = log_s.diff()
    df[f"{prefix}_ma3"]  = log_s.rolling(3, min_periods=1).mean()
    df[f"{prefix}_lag1"] = log_s.shift(1)
    return df


def main() -> None:
    logger.info("=" * 60)
    logger.info("INGEST ENERGY PRICES")
    logger.info("=" * 60)

    # Download raw series
    brent = download_series("BZ=F", "brent")
    wti   = download_series("CL=F", "wti")
    ttf   = download_series("TTF=F", "ttf")
    hh    = download_series("NG=F", "henry_hub")

    logger.info(f"  Brent: {brent.index.min().date()} - {brent.index.max().date()} ({len(brent)} obs)")
    logger.info(f"  WTI:   {wti.index.min().date()} - {wti.index.max().date()} ({len(wti)} obs)")
    logger.info(f"  TTF:   {ttf.index.min().date()} - {ttf.index.max().date()} ({len(ttf)} obs)")
    logger.info(f"  HH:    {hh.index.min().date()} - {hh.index.max().date()} ({len(hh)} obs)")

    # Build full series with proxy backfill
    logger.info("Building full Brent series (proxy: WTI pre-2007)...")
    brent_full = build_proxy(brent, wti, "brent")

    logger.info("Building full TTF series (proxy: Henry Hub pre-2017)...")
    ttf_full = build_proxy(ttf, hh, "ttf")

    logger.info(f"  Brent full: {brent_full.index.min().date()} - {brent_full.index.max().date()} ({len(brent_full)})")
    logger.info(f"  TTF full:   {ttf_full.index.min().date()} - {ttf_full.index.max().date()} ({len(ttf_full)})")

    # Apply transformations
    logger.info("Applying transformations (log, ret, ma3, lag1)...")
    df = transform_series(brent_full, "brent").join(transform_series(ttf_full, "ttf"), how="outer")

    # Shift +1 (anti-leakage): month-t price → exogenous variable at t+1
    df = df.shift(1)

    # Align to project range
    target_idx = pd.date_range("2002-01-01", "2025-06-01", freq="MS")
    df = df.reindex(target_idx)
    df.index.name = "date"
    df.index.freq = "MS"

    logger.info(f"Final dataset:")
    logger.info(f"  Range: {df.index.min().date()} - {df.index.max().date()}")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  NaN per column:\n{df.isna().sum().to_string()}")
    logger.info(f"  First rows (post-shift):\n{df.head(5)}")
    logger.info(f"  Last rows:\n{df.tail(3)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "energy_prices_monthly.parquet"
    df.to_parquet(out)
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
