"""Merge energy_prices_monthly into features_c1.parquet.

Performs a left join by date, forward-fills NaN gaps from Yahoo Finance,
and reports correlations of the 8 new variables with IPC(t+1).

Input:  data/processed/features_c1.parquet    (23 cols)
        data/processed/energy_prices_monthly.parquet (8 cols)
Output: data/processed/features_c1.parquet    (31 cols, overwritten)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed"

ENERGY_COLS = [
    "brent_log", "brent_ret", "brent_ma3", "brent_lag1",
    "ttf_log",   "ttf_ret",   "ttf_ma3",   "ttf_lag1",
]


def main() -> None:
    logger.info("=" * 60)
    logger.info("MERGE ENERGY PRICES INTO features_c1")
    logger.info("=" * 60)

    c1     = pd.read_parquet(PROCESSED_DIR / "features_c1.parquet")
    energy = pd.read_parquet(PROCESSED_DIR / "energy_prices_monthly.parquet")

    logger.info(f"features_c1: {c1.shape[0]} rows, {c1.shape[1]} cols")
    logger.info(f"energy:      {energy.shape[0]} rows, {energy.shape[1]} cols")

    if "date" not in c1.columns:
        c1 = c1.reset_index().rename(columns={c1.index.name or "index": "date"})
    c1["date"] = pd.to_datetime(c1["date"])

    energy = energy.reset_index()
    energy.rename(columns={energy.columns[0]: "date"}, inplace=True)
    energy["date"] = pd.to_datetime(energy["date"])

    existing = [c for c in ENERGY_COLS if c in c1.columns]
    if existing:
        logger.warning(f"Energy columns already exist in features_c1: {existing} - overwriting")
        c1 = c1.drop(columns=existing)

    c1 = c1.merge(energy[["date"] + ENERGY_COLS], on="date", how="left")

    for col in ENERGY_COLS:
        c1[col] = c1[col].ffill().bfill()

    logger.info(f"NaN after forward/back-fill: {c1[ENERGY_COLS].isna().sum().sum()}")

    out = PROCESSED_DIR / "features_c1.parquet"
    c1.to_parquet(out, index=False)
    logger.info(f"features_c1 updated: {c1.shape[0]} rows, {c1.shape[1]} cols")
    logger.info(f"Columns: {list(c1.columns)}")

    # Correlations with IPC(t+1)
    logger.info("=" * 60)
    logger.info("CORRELATION: ENERGY VARS vs IPC(t+1)")
    logger.info("=" * 60)

    c1_indexed = c1.set_index("date")
    ipc_lead   = c1_indexed["indice_general"].shift(-1)

    header = f"{'Variable':<18} {'Corr IPC(t+1)':>14} {'Corr diff':>12} {'Period':>18} {'N':>5}"
    logger.info(header)
    logger.info("-" * 72)

    corr_results = {}
    for col in ENERGY_COLS:
        series = c1_indexed[col]
        valid  = series.notna() & ipc_lead.notna()
        if valid.sum() < 20:
            logger.info(f"{col:<18} {'N/A':>14} {'N/A':>12} {'insufficient':>18} {valid.sum():>5}")
            continue

        corr = float(series[valid].corr(ipc_lead[valid]))
        ipc_diff_lead = c1_indexed["indice_general"].diff().shift(-1)
        valid_diff    = series.notna() & ipc_diff_lead.notna()
        corr_diff     = float(series[valid_diff].corr(ipc_diff_lead[valid_diff])) if valid_diff.sum() > 20 else 0.0

        start = series.dropna().index.min().strftime("%Y-%m")
        end   = series.dropna().index.max().strftime("%Y-%m")

        corr_results[col] = {"corr_level": corr, "corr_diff": corr_diff, "n": int(valid.sum())}
        logger.info(f"{col:<18} {corr:>+14.3f} {corr_diff:>+12.3f} {start+' - '+end:>18} {valid.sum():>5}")

    # Correlations for 2015+ subperiod
    logger.info(f"{'Variable':<18} {'Corr 2015+':>14} {'Diff 2015+':>12}")
    logger.info("-" * 48)
    mask_2015 = c1_indexed.index >= "2015-01-01"
    for col in ENERGY_COLS:
        series_15    = c1_indexed.loc[mask_2015, col]
        ipc_lead_15  = ipc_lead[mask_2015]
        valid        = series_15.notna() & ipc_lead_15.notna()
        if valid.sum() < 10:
            continue
        corr = float(series_15[valid].corr(ipc_lead_15[valid]))

        ipc_diff_lead_15 = c1_indexed["indice_general"].diff().shift(-1)[mask_2015]
        valid_d = series_15.notna() & ipc_diff_lead_15.notna()
        corr_d  = float(series_15[valid_d].corr(ipc_diff_lead_15[valid_d])) if valid_d.sum() > 10 else 0.0
        logger.info(f"{col:<18} {corr:>+14.3f} {corr_d:>+12.3f}")

    # Recommendation summary
    strong = [c for c, v in corr_results.items() if abs(v["corr_level"]) > 0.5]
    medium = [c for c, v in corr_results.items() if 0.3 < abs(v["corr_level"]) <= 0.5]
    weak   = [c for c, v in corr_results.items() if abs(v["corr_level"]) <= 0.3]

    logger.info("=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)
    logger.info(f"  Strong correlation (>0.5):  {strong or 'none'}")
    logger.info(f"  Medium correlation (0.3-0.5): {medium or 'none'}")
    logger.info(f"  Weak correlation (<0.3):    {weak or 'none'}")

    if strong:
        logger.info(f"  >>> LAUNCH C1 models with energy: {strong}")
    elif medium:
        logger.info(f"  >>> CONSIDER C1 models with energy: {medium}")
    else:
        logger.info("  >>> DO NOT launch energy models - correlations too weak.")


if __name__ == "__main__":
    main()
