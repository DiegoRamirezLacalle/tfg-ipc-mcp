"""
05_clean_and_align.py — Limpieza del IPC (INE)

Lee el Excel crudo del INE, extrae solo la sección de índice en nivel,
parsea fechas, verifica gaps, y exporta a parquet versionado.

Entrada:  data/raw/IPC.xlsx
Salida:   data/processed/ipc_spain_index.parquet
          data/snapshots/<tag>.parquet (snapshot versionado)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]          # tfg-forecasting/
RAW_FILE = ROOT / "data" / "raw" / "IPC.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"

# ── Constantes del formato INE ───────────────────────────────
ROW_DATES = 7           # fila con fechas (0-indexed)
ROW_DATA_START = 8      # fila del índice general
ROW_DATA_END = 22       # fila después del último grupo ECOICOP
COL_INDEX_START = 1     # primera col de la sección "Índice"
COL_INDEX_END = 291     # col después de la última (exclusivo)

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
    """Convierte '2024M05' → Timestamp('2024-05-01')."""
    year, month = date_str.split("M")
    return pd.Timestamp(year=int(year), month=int(month), day=1)


def load_and_clean(path: Path = RAW_FILE) -> pd.DataFrame:
    """Lee el Excel del INE y devuelve un DataFrame limpio con DatetimeIndex."""

    raw = pd.read_excel(path, sheet_name=0, header=None)

    # 1. Extraer fechas de la sección de índice
    date_cells = raw.iloc[ROW_DATES, COL_INDEX_START:COL_INDEX_END]
    dates = []
    for d in date_cells:
        if pd.notna(d) and isinstance(d, str) and "M" in d:
            dates.append(parse_ine_date(d.strip()))
        else:
            dates.append(pd.NaT)

    # 2. Extraer valores numéricos de cada grupo ECOICOP
    records = {}
    for row_idx, col_name in ECOICOP_NAMES.items():
        values = raw.iloc[row_idx, COL_INDEX_START:COL_INDEX_END]
        records[col_name] = pd.to_numeric(values, errors="coerce").values

    df = pd.DataFrame(records, index=dates)
    df.index.name = "date"

    # 3. Eliminar filas sin fecha válida o sin dato en índice general
    df = df[df.index.notna()]
    df = df.dropna(subset=["indice_general"])

    # 4. Ordenar cronológicamente (el Excel viene descendente)
    df = df.sort_index()

    return df


def validate_monthly_continuity(df: pd.DataFrame) -> list[str]:
    """Verifica que no hay gaps en la serie mensual. Devuelve lista de problemas."""
    issues = []
    expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq="MS")
    missing = expected.difference(df.index)
    if len(missing) > 0:
        issues.append(f"Faltan {len(missing)} meses: {missing.tolist()}")

    # Duplicados
    dupes = df.index[df.index.duplicated()]
    if len(dupes) > 0:
        issues.append(f"Fechas duplicadas: {dupes.tolist()}")

    return issues


def export(df: pd.DataFrame, tag: str | None = None) -> None:
    """Exporta a processed/ y opcionalmente a snapshots/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    out_path = PROCESSED_DIR / "ipc_spain_index.parquet"
    df.to_parquet(out_path)
    print(f"Exportado: {out_path}  ({len(df)} filas, {df.columns.size} cols)")

    if tag:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        snap_path = SNAPSHOTS_DIR / f"ipc_spain_index_{tag}.parquet"
        df.to_parquet(snap_path)
        print(f"Snapshot:  {snap_path}")


def main() -> None:
    print(f"Leyendo {RAW_FILE} ...")
    df = load_and_clean()

    print(f"\nSerie limpia: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Filas: {len(df)}, Columnas: {list(df.columns)}")
    print(f"\nPrimeras filas:\n{df.head()}")
    print(f"\nUltimas filas:\n{df.tail()}")

    # Estadisticas basicas del indice general
    print(f"\nEstadisticas -- indice general:")
    print(df["indice_general"].describe().to_string())

    # Validacion de continuidad
    issues = validate_monthly_continuity(df)
    if issues:
        print("\n[!] Problemas de continuidad:")
        for iss in issues:
            print(f"  - {iss}")
    else:
        print("\nContinuidad mensual OK -- sin gaps ni duplicados")

    # NaN check por columna
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print(f"\nNaN por columna:\n{nan_counts[nan_counts > 0]}")
    else:
        print("Sin NaN en ninguna columna")

    # Exportar
    tag = f"v1_{df.index.max().strftime('%Y%m')}"
    export(df, tag=tag)


if __name__ == "__main__":
    main()
