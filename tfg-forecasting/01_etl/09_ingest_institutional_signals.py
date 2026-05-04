"""Download EPU Spain, EPU Europe, and ESI Spain institutional signals.

Sources:
  EPU Spain   — policyuncertainty.com (Spain_News_Index, Baker et al.)
  EPU Europe  — policyuncertainty.com (European_News_Index)
  ESI Spain   — Eurostat ei_bssi_m_r2 (EC Economic Sentiment Indicator, SA)

Transformations per series:
  _log   — log(x) to normalise scale
  _diff  — monthly difference (momentum)
  _ma3   — 3-month rolling mean (smoothed)
  _lag1  — 1-month lag

All features shifted +1 to prevent temporal leakage.
Data available from 2002 without sparsity.
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
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = ROOT / "data" / "processed" / "institutional_signals_monthly.parquet"
IPC_PATH    = ROOT / "data" / "processed" / "ipc_spain_index.parquet"


def download_epu() -> pd.DataFrame:
    """Download EPU Spain and EPU Europe from policyuncertainty.com."""
    url = "https://policyuncertainty.com/media/Europe_Policy_Uncertainty_Data.xlsx"
    logger.info("  Downloading EPU Europe (includes Spain_News_Index)...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = pd.read_excel(io.BytesIO(r.content))

    # Remove footnote rows at the bottom
    df = df[df["Year"].apply(lambda x: str(x).replace(".", "").isdigit())]
    df["Year"]  = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["date"]  = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.set_index("date").sort_index()
    df.index.freq = "MS"

    result = pd.DataFrame(index=df.index)
    result["epu_spain"]  = df["Spain_News_Index"].astype(float)
    result["epu_europe"] = df["European_News_Index"].astype(float)

    logger.info(f"    EPU Spain:  {result['epu_spain'].first_valid_index().date()} - "
                f"{result['epu_spain'].last_valid_index().date()}, "
                f"{result['epu_spain'].notna().sum()} valid")
    logger.info(f"    EPU Europe: {result['epu_europe'].first_valid_index().date()} - "
                f"{result['epu_europe'].last_valid_index().date()}, "
                f"{result['epu_europe'].notna().sum()} valid")
    return result


def download_esi() -> pd.Series:
    """Download ESI Spain from Eurostat (ei_bssi_m_r2, seasonally adjusted)."""
    url = ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/"
           "data/ei_bssi_m_r2?format=SDMX-CSV&sinceTimePeriod=1985M01")
    logger.info("  Downloading ESI Spain from Eurostat...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))

    esi = df[
        (df["indic"] == "BS-ESI-I") &
        (df["geo"]   == "ES") &
        (df["s_adj"] == "SA")
    ].copy()

    esi["date"] = pd.to_datetime(esi["TIME_PERIOD"])
    esi = esi.set_index("date").sort_index()

    s = esi["OBS_VALUE"].astype(float)
    s.name = "esi_spain"
    s.index.freq = "MS"

    logger.info(f"    ESI Spain:  {s.first_valid_index().date()} - "
                f"{s.last_valid_index().date()}, {s.notna().sum()} valid")
    return s


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Generate derived features and apply global shift +1."""
    result = pd.DataFrame(index=df.index)

    for col in ["epu_spain", "epu_europe", "esi_spain"]:
        if col not in df.columns:
            continue
        s = df[col].copy()
        s = s.ffill(limit=2)  # forward-fill gaps shorter than 3 months

        result[f"{col}_log"]  = np.log(s.clip(lower=0.01))
        result[f"{col}_diff"] = s.diff()
        result[f"{col}_ma3"]  = s.rolling(3, min_periods=2).mean()
        result[f"{col}_lag1"] = s.shift(1)

    # Global shift +1: all features use values known at t-1 when forecasting t
    result = result.shift(1)
    return result


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


def main() -> None:
    logger.info("=" * 60)
    logger.info("DOWNLOAD INSTITUTIONAL SIGNALS")
    logger.info("=" * 60)

    # Download
    epu = download_epu()
    esi = download_esi()

    raw = epu.copy()
    raw["esi_spain"] = esi.reindex(raw.index)
    logger.info(f"Raw series combined: {raw.shape}")

    # Transform
    df = transform(raw)
    logger.info(f"Transformed (with shift+1): {df.shape}")
    logger.info(f"Range: {df.index.min().date()} - {df.index.max().date()}")
    logger.info("NaN per column:")
    for col in df.columns:
        first_valid = df[col].first_valid_index()
        logger.info(f"  {col}: {df[col].notna().sum()} valid, "
                    f"first={first_valid.date() if first_valid else 'N/A'}")

    # Correlations
    logger.info("=" * 60)
    logger.info("CORRELATIONS WITH IPC(t+1)")
    logger.info("=" * 60)
    corr_df = correlate_with_ipc(df)
    logger.info(f"\n{corr_df.to_string()}")

    max_abs_2015 = corr_df["corr_2015"].abs().max() if "corr_2015" in corr_df.columns else 0
    max_abs_full = corr_df["corr_full"].abs().max()  if "corr_full"  in corr_df.columns else 0
    logger.info(f"Max |corr| 2015+: {max_abs_2015:.4f}")
    logger.info(f"Max |corr| full:  {max_abs_full:.4f}")

    if max_abs_2015 < 0.3 and max_abs_full < 0.3:
        logger.warning("ALL correlations < 0.3. Review before launching models.")
    else:
        logger.info("Correlations sufficient to proceed.")

    if "corr_2015" in corr_df.columns:
        top5 = corr_df["corr_2015"].abs().sort_values(ascending=False).head(5)
        logger.info("Top 5 by |corr_2015|:")
        for col, val in top5.items():
            sign = corr_df.loc[col, "corr_2015"]
            logger.info(f"  {col}: {sign:+.4f}")

    # Save (clip to 2002+)
    df_save = df.loc["2002-01-01":].copy().reset_index()
    if df_save.columns[0] != "date":
        df_save = df_save.rename(columns={df_save.columns[0]: "date"})

    ROOT.joinpath("data", "processed").mkdir(parents=True, exist_ok=True)
    df_save.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved: {OUTPUT_PATH}")
    logger.info(f"Shape: {df_save.shape}")


if __name__ == "__main__":
    main()
