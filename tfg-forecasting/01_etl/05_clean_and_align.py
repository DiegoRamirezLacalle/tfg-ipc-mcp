"""Clean and align Spain IPC data from INE Excel.

Extracts the price index section, parses dates, checks for gaps,
and exports to a versioned parquet.

Input:  data/raw/IPC.xlsx
Output: data/processed/ipc_spain_index.parquet
        data/snapshots/<tag>.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

# Paths
RAW_FILE      = ROOT / "data" / "raw" / "IPC.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"

# INE Excel format constants
ROW_DATES      = 7     # row with date headers (0-indexed)
ROW_DATA_START = 8     # row of the general index
ROW_DATA_END   = 22    # row after the last ECOICOP group
COL_INDEX_START = 1    # first column of the "Index" section
COL_INDEX_END   = 291  # exclusive upper bound

ECOICOP_NAMES = {
    8:  "indice_general",
    9:  "01_alimentos_bebidas",
    10: "02_bebidas_alcoholicas_tabaco",
    11: "03_vestido_calzado",
    12: "04_vivienda_agua_electricidad",
    13: "05_muebles_hogar",
    14: "06_sanidad",
    15: "07_transporte",
    16: "08_informacion_comunicaciones",
    17: "09_ocio_cultura",
    18: "10_ensenanza",
    19: "11_restaurantes_alojamiento",
    20: "12_seguros_servicios_financieros",
    21: "13_cuidado_personal_proteccion_social",
}


def parse_ine_date(date_str: str) -> pd.Timestamp:
    """Convert '2024M05' to Timestamp('2024-05-01')."""
    year, month = date_str.split("M")
    return pd.Timestamp(year=int(year), month=int(month), day=1)


def load_and_clean(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the INE Excel and return a clean DataFrame with DatetimeIndex."""
    raw = pd.read_excel(path, sheet_name=0, header=None)

    # Extract dates from the index section
    date_cells = raw.iloc[ROW_DATES, COL_INDEX_START:COL_INDEX_END]
    dates = []
    for d in date_cells:
        if pd.notna(d) and isinstance(d, str) and "M" in d:
            dates.append(parse_ine_date(d.strip()))
        else:
            dates.append(pd.NaT)

    # Extract numeric values for each ECOICOP group
    records = {}
    for row_idx, col_name in ECOICOP_NAMES.items():
        values = raw.iloc[row_idx, COL_INDEX_START:COL_INDEX_END]
        records[col_name] = pd.to_numeric(values, errors="coerce").values

    df = pd.DataFrame(records, index=dates)
    df.index.name = "date"

    # Drop rows without a valid date or general index
    df = df[df.index.notna()]
    df = df.dropna(subset=["indice_general"])

    # Sort ascending (INE Excel is in descending order)
    df = df.sort_index()
    return df


def validate_monthly_continuity(df: pd.DataFrame) -> list[str]:
    """Check for gaps in the monthly series. Returns a list of issues."""
    issues = []
    expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq="MS")
    missing = expected.difference(df.index)
    if len(missing) > 0:
        issues.append(f"Missing {len(missing)} months: {missing.tolist()}")
    dupes = df.index[df.index.duplicated()]
    if len(dupes) > 0:
        issues.append(f"Duplicate dates: {dupes.tolist()}")
    return issues


def export(df: pd.DataFrame, tag: str | None = None) -> None:
    """Export to processed/ and optionally to snapshots/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "ipc_spain_index.parquet"
    df.to_parquet(out_path)
    logger.info(f"Saved: {out_path}  ({len(df)} rows, {df.columns.size} cols)")

    if tag:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        snap_path = SNAPSHOTS_DIR / f"ipc_spain_index_{tag}.parquet"
        df.to_parquet(snap_path)
        logger.info(f"Snapshot: {snap_path}")


def main() -> None:
    logger.info(f"Reading {RAW_FILE} ...")
    df = load_and_clean()

    logger.info(f"Clean series: {df.index.min().date()} -> {df.index.max().date()}")
    logger.info(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    logger.info(f"First rows:\n{df.head()}")
    logger.info(f"Last rows:\n{df.tail()}")
    logger.info(f"Statistics -- indice_general:\n{df['indice_general'].describe().to_string()}")

    issues = validate_monthly_continuity(df)
    if issues:
        logger.warning("Continuity issues detected:")
        for iss in issues:
            logger.warning(f"  - {iss}")
    else:
        logger.info("Monthly continuity OK — no gaps or duplicates")

    nan_counts = df.isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN per column:\n{nan_counts[nan_counts > 0]}")
    else:
        logger.info("No NaN in any column")

    tag = f"v1_{df.index.max().strftime('%Y%m')}"
    export(df, tag=tag)


if __name__ == "__main__":
    main()
