"""Build the exogenous feature dataset for SARIMAX.

Joins Spain IPC (INE) with ECB rates and generates lag/diff features.

Features produced:
  dfr       - Deposit Facility Rate level (main policy variable)
  mrr       - Main Refinancing Rate level
  dfr_diff  - Monthly DFR change (policy action)
  dfr_lag3  - DFR lagged 3 months
  dfr_lag6  - DFR lagged 6 months (typical transmission: 6-18m)
  dfr_lag12 - DFR lagged 12 months

Input:  data/processed/ipc_spain_index.parquet
        data/processed/ecb_rates_monthly.parquet
Output: data/processed/features_exog.parquet
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    ipc   = pd.read_parquet(PROCESSED_DIR / "ipc_spain_index.parquet")[["indice_general"]]
    rates = pd.read_parquet(PROCESSED_DIR / "ecb_rates_monthly.parquet")

    df = ipc.join(rates, how="inner")
    df.index.freq = "MS"

    df["dfr_diff"]  = df["dfr"].diff()
    df["dfr_lag3"]  = df["dfr"].shift(3)
    df["dfr_lag6"]  = df["dfr"].shift(6)
    df["dfr_lag12"] = df["dfr"].shift(12)

    # Lags introduce NaN at the start - kept intentionally; each model trims as needed
    nan_counts = df.isna().sum()
    if nan_counts.any():
        logger.info(f"NaN per column (expected in initial lags):\n{nan_counts[nan_counts > 0].to_string()}")

    logger.info(f"Range: {df.index.min().date()} - {df.index.max().date()}")
    logger.info(f"Observations: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"First rows:\n{df.head()}")
    logger.info(f"Correlations with indice_general:\n{df.corr()['indice_general'].drop('indice_general').round(3).to_string()}")

    out = PROCESSED_DIR / "features_exog.parquet"
    df.to_parquet(out)
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
