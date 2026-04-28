"""Download Europe-specific macro signals for C1 experiments.

Downloads three FRED series and generates derived features:

  ESIEZM    — ESI Eurozone Economic Sentiment Indicator (European Commission)
              Fallback: BSCICP03EZM665S (OECD BCI Euro Area)
  T5YIE     — 5-Year Breakeven Inflation Rate (market inflation expectations)
  DEXUSEU   — EUR/USD exchange rate (daily average → monthly mean)

Derived features per series: raw, ma3, lag1, diff.
All shifted +1 to prevent leakage (month-t value used at t+1).

Range: 2002-01 to 2024-12 (276 months, aligned with HICP Europe).

Outputs:
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

from shared.logger import get_logger

logger = get_logger(__name__)

DATE_START = "2002-01-01"
DATE_END   = "2024-12-31"
FREQ       = "MS"
OUT_RAW    = ROOT / "data" / "raw" / "europe_signals_raw.csv"
OUT_PROC   = ROOT / "data" / "processed" / "europe_signals_monthly.parquet"


def fetch_fred(series_id: str, fallback_id: str | None = None) -> pd.Series | None:
    """Download a FRED series via pandas_datareader, with optional fallback."""
    try:
        import pandas_datareader as pdr
        s = pdr.get_data_fred(series_id, start=DATE_START, end=DATE_END).squeeze()
        logger.info(f"  [FRED] {series_id}: {len(s)} obs ({s.index.min().date()} - {s.index.max().date()})")
        return s
    except Exception as e:
        logger.warning(f"  [FRED] {series_id} failed: {e}")
        if fallback_id:
            try:
                import pandas_datareader as pdr
                s = pdr.get_data_fred(fallback_id, start=DATE_START, end=DATE_END).squeeze()
                logger.info(f"  [FRED] fallback {fallback_id}: {len(s)} obs")
                return s
            except Exception as e2:
                logger.warning(f"  [FRED] fallback {fallback_id} also failed: {e2}")
        return None


def to_monthly(s: pd.Series, method: str = "last") -> pd.Series:
    """Resample series to monthly frequency (month start)."""
    if method == "mean":
        return s.resample("MS").mean()
    return s.resample("MS").last()


def make_features(s: pd.Series, name: str) -> pd.DataFrame:
    """Generate raw, ma3, lag1, diff with shift +1 to prevent leakage.

    NOTE: lag1 is computed as shift(2) on the raw series, because the base
    series is already shifted by 1. This is lag-2 of the original — do not change.
    """
    df  = pd.DataFrame(index=s.index)
    raw = s.copy()
    df[name]             = raw.shift(1)                  # anti-leakage shift
    df[f"{name}_ma3"]    = raw.rolling(3).mean().shift(1)
    df[f"{name}_lag1"]   = raw.shift(2)                  # lag-2 of raw (lag-1 of shifted)
    df[f"{name}_diff"]   = raw.diff(1).shift(1)
    return df


def main() -> None:
    logger.info("=" * 60)
    logger.info("ETL EUROPE SIGNALS — ESI, Breakeven 5Y, EUR/USD")
    logger.info("=" * 60)

    target_idx = pd.date_range(start=DATE_START, end="2024-12-01", freq=FREQ)
    frames   = []
    raw_dict = {}

    # 1. ESI Eurozone
    logger.info("\n[1] ESI Eurozone (ESIEZM / fallback BSCICP03EZM665S)")
    esi = fetch_fred("ESIEZM", fallback_id="BSCICP03EZM665S")
    if esi is not None:
        esi_m = to_monthly(esi, method="last").reindex(target_idx).ffill().bfill()
        raw_dict["esi_eurozone_raw"] = esi_m
        frames.append(make_features(esi_m, "esi_eurozone"))
        logger.info(f"  ESI Eurozone: {esi_m.dropna().shape[0]} valid obs, "
                    f"range {esi_m.min():.1f}-{esi_m.max():.1f}")
    else:
        logger.warning("  ESI Eurozone not available — columns set to NaN")
        frames.append(pd.DataFrame(np.nan, index=target_idx,
                                   columns=["esi_eurozone", "esi_eurozone_ma3",
                                            "esi_eurozone_lag1", "esi_eurozone_diff"]))

    # 2. 5Y Breakeven Inflation Rate
    logger.info("\n[2] 5Y Breakeven Inflation Rate (T5YIE)")
    brk = fetch_fred("T5YIE")
    if brk is not None:
        brk_m = to_monthly(brk, method="mean").reindex(target_idx).ffill().bfill()
        raw_dict["breakeven_5y_raw"] = brk_m
        frames.append(make_features(brk_m, "breakeven_5y"))
        logger.info(f"  Breakeven 5Y: {brk_m.dropna().shape[0]} valid obs, "
                    f"range {brk_m.min():.2f}%-{brk_m.max():.2f}%")
    else:
        logger.warning("  Breakeven 5Y not available — columns set to NaN")
        frames.append(pd.DataFrame(np.nan, index=target_idx,
                                   columns=["breakeven_5y", "breakeven_5y_ma3",
                                            "breakeven_5y_lag1", "breakeven_5y_diff"]))

    # 3. EUR/USD exchange rate
    logger.info("\n[3] EUR/USD daily rate → monthly mean (DEXUSEU)")
    eur = fetch_fred("DEXUSEU")
    if eur is not None:
        eur_m = to_monthly(eur, method="mean").reindex(target_idx).ffill().bfill()
        raw_dict["eurusd_raw"] = eur_m
        frames.append(make_features(eur_m, "eurusd"))
        logger.info(f"  EUR/USD: {eur_m.dropna().shape[0]} valid obs, "
                    f"range {eur_m.min():.4f}-{eur_m.max():.4f}")
    else:
        logger.warning("  EUR/USD not available — columns set to NaN")
        frames.append(pd.DataFrame(np.nan, index=target_idx,
                                   columns=["eurusd", "eurusd_ma3",
                                            "eurusd_lag1", "eurusd_diff"]))

    # Combine and save
    df_out = pd.concat(frames, axis=1)
    df_out.index.name = "date"
    df_out.index.freq = FREQ

    logger.info(f"\nFinal DataFrame: {df_out.shape}  |  Total NaN: {df_out.isna().sum().sum()}")
    logger.info(f"\n{df_out.head(3).to_string()}")

    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    if raw_dict:
        pd.DataFrame(raw_dict).to_csv(OUT_RAW)
        logger.info(f"\nRaw saved: {OUT_RAW}")

    df_out.to_parquet(OUT_PROC)
    logger.info(f"Processed saved: {OUT_PROC}")

    logger.info("\nColumns generated:")
    for c in df_out.columns:
        logger.info(f"  {c:<30} {df_out[c].notna().sum():>4}/{len(df_out)} valid")

    logger.info("\n" + "=" * 60)
    logger.info("DONE — europe_signals_monthly.parquet generated")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
