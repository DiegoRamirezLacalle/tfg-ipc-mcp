"""Download HICP Eurozone (price level index, base 2015=100) from ECB SDMX.

Series: ICP.M.U2.N.000000.4.INX
  M       = monthly
  U2      = Euro Area (variable composition)
  N       = not seasonally adjusted
  000000  = all items (headline index)
  4/INX   = price level index (not growth rate)

Target range: 2002-01 to 2024-12.

Outputs:
  data/raw/hicp_europe_raw.csv
  data/processed/hicp_europe_index.parquet   (columns: date, hicp_index)
  data/snapshots/hicp_europe_v1_<YYYYMM>.parquet

Inline checks:
  - No NaN or monthly gaps
  - Value ~100 in 2015 (index base year)
  - Visible peak in 2022-2023
  - Plot saved to data/processed/hicp_europe_check.png
"""

from __future__ import annotations

import sys
from datetime import date
from io import StringIO
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

RAW_CSV  = ROOT / "data" / "raw" / "hicp_europe_raw.csv"
PROC_PQ  = ROOT / "data" / "processed" / "hicp_europe_index.parquet"
SNAP_DIR = ROOT / "data" / "snapshots"
PLOT_OUT = ROOT / "data" / "processed" / "hicp_europe_check.png"

ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/ICP/"
    "M.U2.N.000000.4.INX?format=csvdata"
)
START = "2002-01"
END   = "2024-12"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TFG-Academic-Research/1.0)",
    "Accept": "text/csv",
}


def download_raw() -> str:
    logger.info("Downloading HICP Eurozone from ECB SDMX...")
    r = httpx.get(ECB_URL, headers=HEADERS, timeout=60, follow_redirects=True)
    r.raise_for_status()
    logger.info(f"  HTTP {r.status_code} - {len(r.text):,} characters received")
    return r.text


def parse_series(csv_text: str) -> pd.DataFrame:
    df_raw = pd.read_csv(StringIO(csv_text))
    df = (
        df_raw[["TIME_PERIOD", "OBS_VALUE"]]
        .copy()
        .rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "hicp_index"})
    )
    df["date"]       = pd.to_datetime(df["date"].str.strip() + "-01")
    df["hicp_index"] = pd.to_numeric(df["hicp_index"], errors="coerce")
    df = df.dropna(subset=["hicp_index"]).sort_values("date").reset_index(drop=True)
    return df


def filter_range(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(START + "-01")) & (
        df["date"] <= pd.Timestamp(END + "-01")
    )
    df_f = df[mask].copy().reset_index(drop=True)
    logger.info(f"  Rows in range {START} – {END}: {len(df_f)}")
    return df_f


def verify(df: pd.DataFrame) -> None:
    logger.info("--- Quality check ---")

    nans = df["hicp_index"].isna().sum()
    logger.info(f"  NaN: {nans}  {'OK' if nans == 0 else 'WARNING'}")

    expected_months = pd.date_range(start=START + "-01", end=END + "-01", freq="MS")
    actual_months   = set(df["date"])
    missing = [d for d in expected_months if d not in actual_months]
    logger.info(f"  Expected months: {len(expected_months)}  |  Found: {len(df)}")
    if missing:
        logger.warning(f"  GAPS detected ({len(missing)}): {missing[:5]}...")
    else:
        logger.info("  No monthly gaps: OK")

    mean_2015 = df[df["date"].dt.year == 2015]["hicp_index"].mean()
    logger.info(f"  2015 mean: {mean_2015:.2f}  (expected ~100.0)")

    peak_row = df.loc[df["hicp_index"].idxmax()]
    logger.info(f"  Peak value: {peak_row['hicp_index']:.2f}  at {peak_row['date'].strftime('%Y-%m')}")

    logger.info("--- Basic statistics ---")
    logger.info(f"\n{df['hicp_index'].describe().round(3).to_string()}")
    logger.info(f"  First value: {df.iloc[0]['date'].strftime('%Y-%m')} = {df.iloc[0]['hicp_index']:.2f}")
    logger.info(f"  Last value:  {df.iloc[-1]['date'].strftime('%Y-%m')} = {df.iloc[-1]['hicp_index']:.2f}")


def plot_series(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(df["date"], df["hicp_index"], color="#1565c0", linewidth=1.5, label="HICP Eurozone")
    ax.axhline(100, color="#888", linewidth=0.8, linestyle="--", label="Base 2015=100")

    idx_max = df["hicp_index"].idxmax()
    ax.annotate(
        f"Peak: {df.loc[idx_max, 'hicp_index']:.1f}\n{df.loc[idx_max, 'date'].strftime('%Y-%m')}",
        xy=(df.loc[idx_max, "date"], df.loc[idx_max, "hicp_index"]),
        xytext=(30, -18), textcoords="offset points",
        fontsize=8, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8),
    )

    ax.set_title("HICP Eurozone - Price level index (base 2015=100)", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("HICP index")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=130)
    plt.close(fig)
    logger.info(f"  Plot saved: {PLOT_OUT}")


def save(df: pd.DataFrame) -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    PROC_PQ.parent.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(RAW_CSV, index=False)
    logger.info(f"  Raw CSV:   {RAW_CSV}")

    # date as column, not index - consistent with ipc_spain_index.parquet
    df.to_parquet(PROC_PQ, index=False)
    logger.info(f"  Processed: {PROC_PQ}")

    snap_path = SNAP_DIR / f"hicp_europe_v1_{date.today().strftime('%Y%m')}.parquet"
    df.to_parquet(snap_path, index=False)
    logger.info(f"  Snapshot:  {snap_path}")


def main() -> None:
    logger.info("=" * 55)
    logger.info("ETL - HICP Eurozone (ECB SDMX)")
    logger.info("=" * 55)

    csv_text = download_raw()

    df_all = parse_series(csv_text)
    logger.info(f"  Full series: {df_all['date'].min().strftime('%Y-%m')} "
                f"to {df_all['date'].max().strftime('%Y-%m')}  ({len(df_all)} obs)")

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(RAW_CSV, index=False)

    df = filter_range(df_all)

    verify(df)
    plot_series(df)
    save(df)

    logger.info("\n" + "=" * 55)
    logger.info("ETL complete.")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
