"""Ingest ECB interest rates (DFR and MRR) and resample to monthly frequency."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

# Paths
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Date range
DATE_START = "2002-01-01"
DATE_END   = "2025-06-01"


def load_rate(path: Path, col_name: str) -> pd.Series:
    """Read a BCE daily-change CSV and return a monthly series.

    Uses last() + ffill() - correct for step-function policy rate series.
    """
    df = pd.read_csv(path, usecols=["TIME_PERIOD", "OBS_VALUE"])
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    df = df.sort_values("TIME_PERIOD").set_index("TIME_PERIOD")

    monthly = df["OBS_VALUE"].resample("MS").last().ffill()
    monthly.name = col_name
    return monthly


def main() -> None:
    dfr = load_rate(RAW_DIR / "DFR.csv", "dfr")
    mrr = load_rate(RAW_DIR / "MRR.csv", "mrr")

    rates = pd.concat([dfr, mrr], axis=1).loc[DATE_START:DATE_END]

    gaps = rates.isna().sum()
    if gaps.any():
        logger.warning(f"NaN after ffill:\n{gaps[gaps > 0]}")
    else:
        logger.info("No NaN after ffill - OK")

    logger.info(f"Range: {rates.index.min().date()} - {rates.index.max().date()}")
    logger.info(f"Observations: {len(rates)}")
    logger.info(f"First rows:\n{rates.head()}")
    logger.info(f"Last rows:\n{rates.tail()}")
    logger.info(f"Statistics:\n{rates.describe().round(3).to_string()}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "ecb_rates_monthly.parquet"
    rates.to_parquet(out)
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
