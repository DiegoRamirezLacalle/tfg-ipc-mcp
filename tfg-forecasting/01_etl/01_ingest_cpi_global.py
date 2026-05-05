"""Download and ingest monthly global inflation data from the World Bank.

Source:
  World Bank Global Inflation Dataset
  Sheet: hcpi_m (Headline CPI Index, monthly, 186 countries)
  Time range: 2002-01 to 2024-12

Data format note:
  The hcpi_m sheet stores the consumer price INDEX (not the rate).
  There is no "World" row - only individual countries.
  The global monthly rate is computed as:
    (1) YoY rate per country: pct_change(12) on the index
    (2) Cross-sectional median across all countries with available data
  This replicates the World Bank HCPI_GLOBAL_MED methodology.

  cpi_global_rate is the YoY % change (median across countries).

Outputs:
  data/raw/cpi_global_raw.xlsx
  data/processed/cpi_global_monthly.parquet    (date, cpi_global_rate)
  data/snapshots/cpi_global_v1_YYYYMM.parquet
  data/processed/cpi_global_monthly_check.png
"""

from __future__ import annotations

import io
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

# Ensure UTF-8 output on Windows (prevents charmap errors in logger output)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from shared.logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"

SOURCE_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "1ad246272dbbc437c74323719506aa0c-0350012021/original/Inflation-data.xlsx"
)
SHEET      = "hcpi_m"
DATE_START = "2002-01-01"
DATE_END   = "2024-12-31"


def download_excel(url: str, dest: Path) -> bytes:
    logger.info(f"  Downloading {url} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    logger.info(f"  Saved: {dest}  ({len(r.content)/1024:.0f} KB)")
    return r.content


def load_hcpi_m(content: bytes) -> pd.DataFrame:
    """Load the hcpi_m sheet and return a wide DataFrame.

    Rows = countries (Country Code as index).
    Columns = pd.Timestamp (first day of month).
    """
    df = pd.read_excel(io.BytesIO(content), sheet_name=SHEET, engine="openpyxl")

    # Date columns: int/float with YYYYMM format (e.g. 197001)
    date_cols = {}
    for col in df.columns:
        if isinstance(col, (int, float)) and not np.isnan(float(col)):
            ival = int(col)
            year, month = divmod(ival, 100)
            if 1950 <= year <= 2030 and 1 <= month <= 12:
                date_cols[col] = pd.Timestamp(year, month, 1)

    # Keep only date columns
    meta_cols  = ["Country Code", "IMF Country Code", "Country", "Indicator Type", "Series Name"]
    date_orig  = list(date_cols.keys())

    # Exclude last row (footnote)
    df = df[df["Country Code"].notna() &
            ~df["Country Code"].astype(str).str.lower().str.startswith("note")]

    df_wide = df.set_index("Country Code")[date_orig].copy()
    df_wide.columns = [date_cols[c] for c in date_orig]
    df_wide = df_wide.astype(float)

    logger.info(f"  Countries: {len(df_wide)}  |  "
                f"Dates: {df_wide.columns.min().date()} to {df_wide.columns.max().date()}")
    return df_wide


def compute_global_rate(df_wide: pd.DataFrame, date_start: str, date_end: str) -> pd.Series:
    """Compute the cross-sectional median YoY inflation rate.

    Per country: YoY rate = pct_change(12) on the monthly index.
    Global rate = cross-sectional median across all countries with available data.
    Filtered to [date_start, date_end].
    """
    # YoY per country: (index_t / index_t-12 - 1) * 100
    yoy = df_wide.pct_change(periods=12, axis=1) * 100   # rows=countries, cols=dates

    # Transpose to (date, country)
    yoy_T = yoy.T
    yoy_T.index = pd.DatetimeIndex(yoy_T.index)
    yoy_T.index.freq = "MS"

    # Cross-sectional median
    global_med = yoy_T.median(axis=1, skipna=True)
    global_med.name = "cpi_global_rate"

    # Filter to date range
    mask   = (global_med.index >= date_start) & (global_med.index <= date_end)
    series = global_med[mask].dropna()

    # Coverage: number of countries with data per month
    n_countries = yoy_T[mask].notna().sum(axis=1)
    logger.info(f"  Mean coverage per month: {n_countries.mean():.0f} countries")
    logger.info(f"  Coverage min/max: {n_countries.min()}/{n_countries.max()} countries")

    return series


def log_stats(series: pd.Series) -> None:
    logger.info("─" * 52)
    logger.info("STATISTICS - cpi_global_rate (YoY median, %)")
    logger.info("─" * 52)
    logger.info(f"  Date range   : {series.index.min().date()} → {series.index.max().date()}")
    logger.info(f"  Observations : {len(series)}")
    logger.info(f"  NaN count    : {series.isna().sum()}")
    logger.info(f"  Mean         : {series.mean():.4f} %")
    logger.info(f"  Median       : {series.median():.4f} %")
    logger.info(f"  Min          : {series.min():.4f} %  ({series.idxmin().date()})")
    logger.info(f"  Max          : {series.max():.4f} %  ({series.idxmax().date()})")
    logger.info(f"  Std          : {series.std():.4f} %")
    logger.info("─" * 52)
    expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    missing  = expected.difference(series.index)
    logger.info(f"  Gaps: {'none' if len(missing) == 0 else missing.tolist()}")


def plot_series(series: pd.Series, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        "Global Monthly Inflation Rate - World Bank hcpi_m\n"
        "(cross-country YoY median of 186 countries, replicates HCPI_GLOBAL_MED)",
        fontsize=12, fontweight="bold", y=0.99,
    )

    SHADING = [
        ("Financial crisis", "2008-09-01", "2009-06-30", "#fff3cd", 0.55),
        ("Covid-19",         "2020-01-01", "2020-12-31", "#e8e8e8", 0.50),
        ("Inflation shock",  "2021-01-01", "2022-12-31", "#f8d7d7", 0.55),
        ("Normalization",    "2023-01-01", "2024-12-31", "#d7e8f8", 0.40),
    ]

    # Upper panel: series
    ax = axes[0]
    for label, s, e, color, alpha in SHADING:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=color, alpha=alpha, zorder=0, label=label)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.plot(series.index, series.values,
            color="#2166ac", linewidth=1.8, zorder=3, label="Global inflation (YoY, %)")
    ax.scatter(series.idxmax(), series.max(), color="#d62728", s=70, zorder=5,
               label=f"Max: {series.max():.2f}% ({series.idxmax().strftime('%Y-%m')})")
    ax.scatter(series.idxmin(), series.min(), color="#1f77b4", s=70,
               marker="v", zorder=5,
               label=f"Min: {series.min():.2f}% ({series.idxmin().strftime('%Y-%m')})")
    ax.set_ylabel("YoY inflation rate (%)", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_xlim(series.index.min(), series.index.max())
    ax.grid(axis="y", alpha=0.3)

    # Lower panel: rolling volatility
    ax2      = axes[1]
    roll_std = series.rolling(12, min_periods=6).std()
    ax2.fill_between(roll_std.index, roll_std.values, color="#9ecae1", alpha=0.7)
    ax2.plot(roll_std.index, roll_std.values, color="#2166ac", linewidth=1.0)
    ax2.set_ylabel("Rolling 12m std", fontsize=9)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_xlim(series.index.min(), series.index.max())
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\n  Plot saved: {out_path}")


def main() -> None:
    logger.info("=" * 60)
    logger.info("INGEST CPI GLOBAL - World Bank hcpi_m (186 countries)")
    logger.info("=" * 60)

    # 1. Download
    raw_path = RAW_DIR / "cpi_global_raw.xlsx"
    content  = download_excel(SOURCE_URL, raw_path)

    # 2. Load sheet as country × date matrix
    logger.info("\nLoading hcpi_m sheet...")
    df_wide = load_hcpi_m(content)

    # 3. Compute global rate (cross-sectional median YoY)
    logger.info("\nComputing global YoY rate (cross-sectional median)...")
    series = compute_global_rate(df_wide, DATE_START, DATE_END)

    # 4. Statistics
    log_stats(series)

    # 5. Output DataFrame
    df_out = series.rename("cpi_global_rate").to_frame()
    df_out.index.name = "date"
    logger.info(f"\n  First rows:\n{df_out.head(4).to_string()}")
    logger.info(f"  ...\n  Last rows:\n{df_out.tail(4).to_string()}")

    # 6. Save processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    proc_path = PROCESSED_DIR / "cpi_global_monthly.parquet"
    df_out.to_parquet(proc_path)
    logger.info(f"\n  Saved processed: {proc_path}")

    # 7. Versioned snapshot
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    version   = datetime.now().strftime("%Y%m")
    snap_path = SNAPSHOTS_DIR / f"cpi_global_v1_{version}.parquet"
    df_out.to_parquet(snap_path)
    logger.info(f"  Saved snapshot:  {snap_path}")

    # 8. Plot
    plot_series(series, PROCESSED_DIR / "cpi_global_monthly_check.png")

    logger.info("\n" + "=" * 60)
    logger.info("ETL complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
