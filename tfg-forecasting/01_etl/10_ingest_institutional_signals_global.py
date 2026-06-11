"""Download institutional signals for the C1_institutional Global experiment.

Downloads 10 signals for the Global CPI C1 pipeline.
Equivalent to 09_ingest_institutional_signals.py but with global coverage.

Signals:
  FRED (direct API):
    GEPUCURRENT    - Global Economic Policy Uncertainty Index (Davis et al.)
    PALLFNFINDEXM  - IMF All Commodity Price Index (2016=100)
    DTWEXBGS       - Broad Real Dollar Index (USD vs. major currencies)
    VIXCLS         - CBOE VIX (monthly mean)
    DGS10          - 10Y US Treasury yield (global financial conditions)
    FEDFUNDS       - Federal Funds Rate

  Direct downloads:
    GSCPI          - NY Fed Global Supply Chain Pressure Index
    GPR            - Geopolitical Risk Index (Caldara-Iacoviello, Fed)

  Reused from existing parquets:
    brent_log      - Brent price (log), from energy_prices_monthly.parquet
    dfr            - ECB Deposit Facility Rate, from ecb_rates_monthly.parquet

Transformations per base signal: _ma3, _lag1, _diff.
All series shifted +1 to prevent temporal leakage.
Final range: 2002-01-01 to 2024-12-01.

Output: data/processed/features_c1_global_institutional.parquet
"""

from __future__ import annotations

import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

OUTPUT_PATH = ROOT / "data" / "processed" / "features_c1_global_institutional.parquet"
CPI_PATH    = ROOT / "data" / "processed" / "cpi_global_monthly.parquet"

DATE_START = "2001-01-01"  # one month before target so shift+1 covers 2002-01
DATE_END   = "2024-12-31"
IDX        = pd.date_range("2002-01-01", "2024-12-01", freq="MS")


def _monthly_index(s: pd.Series) -> pd.Series:
    """Normalise index to freq=MS and sort."""
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.index = s.index + pd.offsets.MonthBegin(0)
    return s.sort_index()


def _derive(s: pd.Series, name: str) -> pd.DataFrame:
    """Generate _ma3, _lag1, _diff with shift+1 anti-leakage.

    NOTE: _lag1 is computed after the global shift(1), making it
    effectively lag-2 of the raw series. Do not change this.
    """
    s = s.shift(1)  # anti-leakage: month-t value enters model at t+1
    out = pd.DataFrame(index=s.index)
    out[f"{name}_ma3"]  = s.rolling(3).mean()
    out[f"{name}_lag1"] = s.shift(1)
    out[f"{name}_diff"] = s.diff(1)
    return out


def _fred(series_id: str) -> pd.Series:
    """Download a FRED series via direct URL and aggregate to monthly mean."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    logger.info(f"  FRED {series_id} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    val_col  = [c for c in df.columns if c != date_col][0]
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.Series(
        df[val_col].replace(".", np.nan).astype(float).values,
        index=df[date_col],
    ).dropna()
    # Resample to monthly mean (covers both daily and monthly source series)
    s = s.resample("MS").mean()
    s = s.reindex(pd.date_range(DATE_START, DATE_END, freq="MS")).ffill()
    logger.info(f"    {s.notna().sum()} obs  ({s.first_valid_index().date()} - {s.last_valid_index().date()})")
    return s


def download_fred_signals() -> dict[str, pd.Series]:
    fred_ids = {
        "gepu":     "GEPUCURRENT",
        "imf_comm": "PALLFNFINDEXM",
        "dxy":      "DTWEXBGS",
        "vix":      "VIXCLS",
        "usg10y":   "DGS10",
        "fedfunds": "FEDFUNDS",
    }
    return {name: _fred(sid) for name, sid in fred_ids.items()}


def download_gscpi() -> pd.Series:
    url = ("https://www.newyorkfed.org/medialibrary/research/interactives/"
           "gscpi/downloads/gscpi_data.xlsx")
    logger.info("  GSCPI (NY Fed) ...")
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    xls = pd.read_excel(io.BytesIO(r.content), sheet_name=None)

    monthly_sheets = [k for k in xls if "monthly" in k.lower() or "data" in k.lower()]
    df = xls[monthly_sheets[0]] if monthly_sheets else list(xls.values())[-1]

    date_col = [c for c in df.columns if "date" in str(c).lower()]
    val_col  = [c for c in df.columns if "gscpi" in str(c).lower()]
    if not date_col:
        date_col = [df.columns[0]]
    if not val_col:
        val_col = [df.columns[1]]

    s = pd.Series(df[val_col[0]].values, index=pd.to_datetime(df[date_col[0]])).dropna()
    s = _monthly_index(s)
    logger.info(f"    {s.notna().sum()} obs  ({s.index.min().date()} - {s.index.max().date()})")
    return s


def download_gpr() -> pd.Series:
    """Try multiple known URLs for the GPR dataset (Caldara-Iacoviello)."""
    urls = [
        "https://www.matteoiacoviello.com/gpr_files/gpr_data.xls",
        "https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.xls",
        "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
    ]
    logger.info("  GPR (Caldara-Iacoviello) ...")
    content = None
    for url in urls:
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200:
                content = r.content
                break
        except Exception:
            continue

    if content is None:
        logger.warning("GPR not available - using GPR=0 as fallback")
        return pd.Series(0.0, index=pd.date_range(DATE_START, DATE_END, freq="MS"), name="gpr")

    xls = pd.read_excel(io.BytesIO(content), sheet_name=None)
    df  = list(xls.values())[0]

    if "Year" in df.columns and "Month" in df.columns:
        df["date"] = pd.to_datetime(
            df["Year"].astype(int).astype(str) + "-" +
            df["Month"].astype(int).astype(str).str.zfill(2) + "-01"
        )
        val_col = [c for c in df.columns
                   if "gpr" in str(c).lower() and "threat" not in str(c).lower()
                   and c not in ("Year", "Month", "date")]
        s = pd.Series(df[val_col[0]].values, index=df["date"]).dropna()
    else:
        date_col = df.columns[0]
        val_col  = [c for c in df.columns[1:] if "gpr" in str(c).lower()]
        if not val_col:
            val_col = [df.columns[1]]
        s = pd.Series(df[val_col[0]].values, index=pd.to_datetime(df[date_col])).dropna()

    s = _monthly_index(s)
    logger.info(f"    {s.notna().sum()} obs  ({s.index.min().date()} - {s.index.max().date()})")
    return s


def load_existing() -> dict[str, pd.Series]:
    energy = pd.read_parquet(ROOT / "data" / "processed" / "energy_prices_monthly.parquet")
    ecb    = pd.read_parquet(ROOT / "data" / "processed" / "ecb_rates_monthly.parquet")
    return {
        "brent_log": energy["brent_log"],
        "dfr":       ecb["dfr"],
    }


def print_correlations(features: pd.DataFrame, cpi: pd.Series) -> list[str]:
    """Log correlations of each column with cpi_global_rate(t+1) across three periods."""
    target = cpi.shift(-1)

    periods = {
        "full":       (None, None),
        "2015+":      ("2015-01-01", None),
        "shock_2021+": ("2021-01-01", "2023-12-01"),
    }

    sep = "-" * 72
    logger.info(f"\n{sep}")
    logger.info("CORRELATIONS with cpi_global_rate(t+1)")
    logger.info(f"{'Signal':<30} {'Full':>10} {'2015+':>10} {'Shock':>10}")
    logger.info(sep)

    selected = []
    for col in sorted(features.columns):
        corrs = {}
        for pname, (start, end) in periods.items():
            mask = pd.Series(True, index=target.index)
            if start:
                mask &= target.index >= start
            if end:
                mask &= target.index <= end
            c = features.loc[mask, col].corr(target[mask])
            corrs[pname] = c
        mark = " *" if abs(corrs["full"]) >= 0.2 else "  "
        logger.info(f"  {col:<28} {corrs['full']:>10.3f} {corrs['2015+']:>10.3f} "
                    f"{corrs['shock_2021+']:>10.3f}{mark}")
        if abs(corrs["full"]) >= 0.2:
            selected.append(col)

    logger.info(sep)
    logger.info(f"  {len(selected)}/{len(features.columns)} signals with |corr| >= 0.2 (full period)\n")
    return selected


def main() -> None:
    logger.info("=" * 60)
    logger.info("ETL C1_institutional - CPI Global")
    logger.info("=" * 60)

    logger.info("\n1. Downloading FRED signals...")
    fred = download_fred_signals()

    logger.info("\n2. Downloading GSCPI (NY Fed)...")
    gscpi = download_gscpi()

    logger.info("\n3. Downloading GPR (Caldara-Iacoviello)...")
    gpr = download_gpr()

    logger.info("\n4. Loading existing parquets...")
    existing = load_existing()
    for k, s in existing.items():
        logger.info(f"  {k}: {s.notna().sum()} obs")

    raw = {**fred, "gscpi": gscpi, "gpr": gpr, **existing}

    # Align all raw series to monthly IDX 2002-2024
    df_raw = pd.DataFrame(index=IDX)
    for name, s in raw.items():
        aligned = s.reindex(IDX).ffill(limit=3)
        df_raw[name] = aligned

    logger.info(f"\n5. Series aligned to {IDX[0].date()} - {IDX[-1].date()}:")
    for col in df_raw.columns:
        missing = df_raw[col].isna().sum()
        first   = df_raw[col].first_valid_index()
        logger.info(f"  {col:<15} NaN={missing:>3}  ({first.date() if first is not None else 'all NaN'})")

    # Generate derived features (_ma3, _lag1, _diff with shift+1 inside _derive)
    logger.info("\n6. Generating derived features (_ma3, _lag1, _diff with shift+1)...")
    parts = [_derive(df_raw[name], name) for name in df_raw.columns]
    features = pd.concat(parts, axis=1).reindex(IDX)
    features = features.bfill(limit=3).ffill(limit=3)

    logger.info(f"  Total columns generated: {len(features.columns)}")
    logger.info(f"  Total NaN: {features.isna().sum().sum()}")

    # Correlations and column selection
    cpi = pd.read_parquet(CPI_PATH)["cpi_global_rate"].reindex(IDX)
    selected_cols = print_correlations(features, cpi)

    features["cpi_global_rate"] = cpi

    features.index.name = "date"
    features.to_parquet(OUTPUT_PATH)
    logger.info(f"\nSaved: {OUTPUT_PATH}")
    logger.info(f"Shape: {features.shape}")
    logger.info("Selected columns for C1_institutional (|corr|>=0.2):")
    for c in selected_cols:
        logger.info(f"  {c}")

    sel_path = ROOT / "data" / "processed" / "c1_global_inst_selected_cols.json"
    with open(sel_path, "w") as f:
        json.dump(selected_cols, f, indent=2)
    logger.info(f"\nSelected columns saved to: {sel_path}")

    logger.info("\n" + "=" * 60)
    logger.info("ETL C1_institutional COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
