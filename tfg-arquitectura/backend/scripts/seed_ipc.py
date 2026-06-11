"""
Seed script: loads all processed time-series datasets into the platform.

Datasets seeded:
  1. ipc-spain-ine       — Spanish IPC monthly index (INE, base 2021=100)
  2. cpi-global-monthly  — Global CPI monthly rate
  3. hicp-europe-monthly — European HICP monthly index (Eurostat)
  4. features-exog       — ECB rate features used by ridge-exog / sarimax

Run inside the backend container:
  docker compose exec backend python scripts/seed_ipc.py
"""
import os
import asyncio

import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# ── config ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://"
    f"{os.getenv('POSTGRES_USER','tfg')}:"
    f"{os.getenv('POSTGRES_PASSWORD','changeme')}@"
    f"{os.getenv('POSTGRES_HOST','localhost')}:"
    f"{os.getenv('POSTGRES_PORT','5432')}/"
    f"{os.getenv('POSTGRES_DB','tfg_experiments')}"
)

_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../tfg-forecasting/data/processed"
)

# ── dataset definitions ───────────────────────────────────────────────────────

DATASETS = [
    {
        "slug": "ipc-spain-ine",
        "name": "IPC España — INE",
        "description": "Índice de Precios de Consumo mensual. Fuente: INE (Instituto Nacional de Estadística). Base 2021=100.",
        "frequency": "monthly",
        "version": "v1",
        "parquet": "ipc_spain_index.parquet",
        "index_col": None,  # parquet has DatetimeIndex already
        "series": {
            "indice_general":                        ("IPC General",                       "index (2021=100)"),
            "01_alimentos_bebidas":                  ("Alimentos y bebidas no alcohólicas", "index"),
            "02_bebidas_alcoholicas_tabaco":         ("Bebidas alcohólicas y tabaco",       "index"),
            "03_vestido_calzado":                    ("Vestido y calzado",                  "index"),
            "04_vivienda_agua_electricidad":         ("Vivienda, agua, electricidad",       "index"),
            "05_muebles_hogar":                      ("Muebles y artículos del hogar",      "index"),
            "06_sanidad":                            ("Sanidad",                            "index"),
            "07_transporte":                         ("Transporte",                        "index"),
            "08_informacion_comunicaciones":         ("Información y comunicaciones",       "index"),
            "09_ocio_cultura":                       ("Ocio y cultura",                    "index"),
            "10_ensenanza":                          ("Enseñanza",                         "index"),
            "11_restaurantes_alojamiento":           ("Restaurantes y alojamiento",        "index"),
            "12_seguros_servicios_financieros":      ("Seguros y servicios financieros",   "index"),
            "13_cuidado_personal_proteccion_social": ("Cuidado personal y protección",     "index"),
        },
    },
    {
        "slug": "cpi-global-monthly",
        "name": "CPI Global Monthly",
        "description": "Global Consumer Price Index monthly rate. Aggregated cross-country series.",
        "frequency": "monthly",
        "version": "v1",
        "parquet": "cpi_global_monthly.parquet",
        "index_col": None,
        "series": {
            "cpi_global_rate": ("CPI Global Rate", "rate"),
        },
    },
    {
        "slug": "hicp-europe-monthly",
        "name": "HICP Europe Monthly",
        "description": "Harmonised Index of Consumer Prices (HICP) for the euro area. Source: Eurostat.",
        "frequency": "monthly",
        "version": "v1",
        "parquet": "hicp_europe_index.parquet",
        "index_col": "date",  # has a 'date' column rather than DatetimeIndex
        "series": {
            "hicp_index": ("HICP Index", "index"),
        },
    },
    {
        "slug": "features-exog",
        "name": "ECB Rate Features (exog)",
        "description": "ECB Deposit Facility Rate and Main Refinancing Rate with lags/diffs. Used as exogenous features for ridge-exog and sarimax adapters.",
        "frequency": "monthly",
        "version": "v1",
        "parquet": "features_exog.parquet",
        "index_col": None,
        "series": {
            "dfr":       ("ECB Deposit Facility Rate",        "percent"),
            "mrr":       ("ECB Main Refinancing Rate",        "percent"),
            "dfr_diff":  ("DFR Month-over-Month Change",      "percent"),
            "dfr_lag3":  ("DFR Lagged 3 Months",              "percent"),
            "dfr_lag6":  ("DFR Lagged 6 Months",              "percent"),
            "dfr_lag12": ("DFR Lagged 12 Months",             "percent"),
        },
    },
]


async def _upsert_dataset(session, ds_def: dict) -> int:
    result = await session.execute(
        text("SELECT id FROM datasets WHERE slug = :slug"),
        {"slug": ds_def["slug"]},
    )
    row = result.fetchone()
    if row:
        dataset_id = row[0]
        print(f"Dataset '{ds_def['slug']}' already exists (id={dataset_id}), skipping insert.")
    else:
        result = await session.execute(
            text("""
                INSERT INTO datasets (slug, name, description, frequency, version)
                VALUES (:slug, :name, :description, :frequency, :version)
                RETURNING id
            """),
            {k: ds_def[k] for k in ("slug", "name", "description", "frequency", "version")},
        )
        dataset_id = result.scalar()
        await session.commit()
        print(f"Created dataset '{ds_def['slug']}' (id={dataset_id})")
    return dataset_id


async def _seed_dataset(session, ds_def: dict) -> None:
    parquet_path = os.path.join(_DATA_DIR, ds_def["parquet"])
    if not os.path.exists(parquet_path):
        print(f"  [SKIP] {parquet_path} not found")
        return

    df = pd.read_parquet(parquet_path)

    # Normalise index to DatetimeIndex
    if ds_def["index_col"] and ds_def["index_col"] in df.columns:
        df = df.set_index(ds_def["index_col"])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"\n=== {ds_def['slug']} ===")
    print(f"  {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Columns: {list(df.columns)}")

    dataset_id = await _upsert_dataset(session, ds_def)

    for slug, (name, unit) in ds_def["series"].items():
        if slug not in df.columns:
            print(f"  [SKIP] column '{slug}' not in parquet")
            continue

        result = await session.execute(
            text("SELECT id FROM series WHERE dataset_id = :ds AND slug = :slug"),
            {"ds": dataset_id, "slug": slug},
        )
        srow = result.fetchone()
        if srow:
            series_id = srow[0]
            await session.execute(
                text("DELETE FROM observations WHERE series_id = :sid"),
                {"sid": series_id},
            )
            print(f"  Series '{slug}' exists (id={series_id}) — refreshing observations")
        else:
            result = await session.execute(
                text("""
                    INSERT INTO series (dataset_id, slug, name, unit)
                    VALUES (:ds, :slug, :name, :unit)
                    RETURNING id
                """),
                {"ds": dataset_id, "slug": slug, "name": name, "unit": unit},
            )
            series_id = result.scalar()
            print(f"  Created series '{slug}' (id={series_id})")

        values = [
            {"series_id": series_id, "timestamp": ts.to_pydatetime(), "value": float(val)}
            for ts, val in zip(df.index, df[slug])
            if pd.notna(val)
        ]
        if values:
            await session.execute(
                text("""
                    INSERT INTO observations (series_id, timestamp, value)
                    VALUES (:series_id, :timestamp, :value)
                """),
                values,
            )
        print(f"    → {len(values)} observations inserted")

    await session.commit()


async def seed():
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        for ds_def in DATASETS:
            await _seed_dataset(session, ds_def)

    await engine.dispose()
    print("\nDone! All datasets seeded successfully.")


if __name__ == "__main__":
    asyncio.run(seed())
