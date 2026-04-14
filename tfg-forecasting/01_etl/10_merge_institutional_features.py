"""
10_merge_institutional_features.py — Merge institutional signals into features_c1.parquet

Only merges EPU Europe columns (EPU Spain and ESI Spain have weak correlations <0.2).
EPU Europe variables with IPC(t+1) correlation in 2015+:
  epu_europe_ma3:  +0.737
  epu_europe_log:  +0.701
  epu_europe_lag1: +0.682
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT / "data" / "processed" / "features_c1.parquet"
INST_PATH = ROOT / "data" / "processed" / "institutional_signals_monthly.parquet"

# Only EPU Europe (EPU Spain corr <0.2, ESI Spain corr <0.18 — too weak)
COLS_TO_MERGE = ["epu_europe_log", "epu_europe_ma3", "epu_europe_lag1"]


def main():
    # Load
    feat = pd.read_parquet(FEATURES_PATH)
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.set_index("date")

    inst = pd.read_parquet(INST_PATH)
    inst["date"] = pd.to_datetime(inst["date"])
    inst = inst.set_index("date")

    print(f"Features pre-merge: {feat.shape}")
    print(f"Institutional signals: {inst.shape}")

    # Merge only EPU Europe columns
    for col in COLS_TO_MERGE:
        if col in inst.columns:
            feat[col] = inst[col].reindex(feat.index)
            n_nan = feat[col].isna().sum()
            print(f"  Merged {col}: {n_nan} NaN / {len(feat)} rows")

    # Forward-fill small gaps
    for col in COLS_TO_MERGE:
        if col in feat.columns:
            feat[col] = feat[col].ffill(limit=2)

    print(f"\nFeatures post-merge: {feat.shape}")
    print(f"NaN summary:")
    for col in COLS_TO_MERGE:
        if col in feat.columns:
            n_nan = feat[col].isna().sum()
            first = feat[col].first_valid_index()
            print(f"  {col}: {n_nan} NaN, first valid: {first.date() if first else 'N/A'}")

    # Save
    feat = feat.reset_index()
    feat.to_parquet(FEATURES_PATH, index=False)
    print(f"\nGuardado: {FEATURES_PATH}")
    print(f"Shape final: {feat.shape}")
    print(f"Columnas: {feat.columns.tolist()}")


if __name__ == "__main__":
    main()
