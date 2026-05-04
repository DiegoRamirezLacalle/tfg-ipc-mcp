"""Merge institutional signals into features_c1.parquet.

Only EPU Europe columns are merged (EPU Spain and ESI Spain have weak
correlations <0.2 with IPC(t+1) and are excluded).

EPU Europe correlations with IPC(t+1) in 2015+:
  epu_europe_ma3:  +0.737
  epu_europe_log:  +0.701
  epu_europe_lag1: +0.682
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

FEATURES_PATH = ROOT / "data" / "processed" / "features_c1.parquet"
INST_PATH     = ROOT / "data" / "processed" / "institutional_signals_monthly.parquet"

COLS_TO_MERGE = ["epu_europe_log", "epu_europe_ma3", "epu_europe_lag1"]


def main() -> None:
    feat = pd.read_parquet(FEATURES_PATH)
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.set_index("date")

    inst = pd.read_parquet(INST_PATH)
    inst["date"] = pd.to_datetime(inst["date"])
    inst = inst.set_index("date")

    logger.info(f"Features pre-merge: {feat.shape}")
    logger.info(f"Institutional signals: {inst.shape}")

    for col in COLS_TO_MERGE:
        if col in inst.columns:
            feat[col] = inst[col].reindex(feat.index)
            logger.info(f"  Merged {col}: {feat[col].isna().sum()} NaN / {len(feat)} rows")

    for col in COLS_TO_MERGE:
        if col in feat.columns:
            feat[col] = feat[col].ffill(limit=2)

    logger.info(f"Features post-merge: {feat.shape}")
    for col in COLS_TO_MERGE:
        if col in feat.columns:
            first = feat[col].first_valid_index()
            logger.info(f"  {col}: {feat[col].isna().sum()} NaN, first valid: {first.date() if first else 'N/A'}")

    feat = feat.reset_index()
    feat.to_parquet(FEATURES_PATH, index=False)
    logger.info(f"Saved: {FEATURES_PATH}")
    logger.info(f"Final shape: {feat.shape}")
    logger.info(f"Columns: {feat.columns.tolist()}")


if __name__ == "__main__":
    main()
