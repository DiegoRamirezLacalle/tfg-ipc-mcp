"""
ETL: load processed parquet files from tfg-forecasting into Postgres.

Run inside Docker:
  python -m app.etl.load_parquets

Idempotent - uses INSERT ... ON CONFLICT DO NOTHING for all records.
"""
import asyncio
import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import AsyncSessionLocal
from app.models.model_catalog import ModelCatalog, ModelType

log = logging.getLogger(__name__)

DATA_ROOT = Path(os.getenv("TFG_DATA_ROOT", "/app/tfg-forecasting/data/processed"))

# Parquet files -> (dataset_slug, dataset_name, frequency, unit_hint)
DATASETS = {
    "ipc_spain_index.parquet": (
        "ipc-spain",
        "IPC España (Índice de Precios al Consumo)",
        "monthly",
        "index (2021=100)",
    ),
    "ecb_rates_monthly.parquet": (
        "ecb-rates",
        "ECB Interest Rates",
        "monthly",
        "%",
    ),
    "features_exog.parquet": (
        "features-exog",
        "Exogenous Features (merged)",
        "monthly",
        "mixed",
    ),
}

MODEL_CATALOG_SEED = [
    {
        "slug": "naive-seasonal",
        "name": "Naive Seasonal",
        "model_type": "naive",
        "description": "Seasonal naïve baseline - repeats last observed seasonal cycle.",
        "supports_mcp": False,
    },
    {
        "slug": "sarima",
        "name": "SARIMA",
        "model_type": "arima",
        "description": "Seasonal ARIMA via statsmodels; order selected by AIC grid search.",
        "supports_mcp": False,
    },
    {
        "slug": "ridge-exog",
        "name": "Ridge Regression (exogenous)",
        "model_type": "ridge",
        "description": "Ridge regression with lag features + ECB rate exogenous variables.",
        "supports_mcp": True,
    },
    {
        "slug": "timesfm",
        "name": "TimesFM (Google)",
        "model_type": "timesfm",
        "description": "Google TimesFM foundation model for time series; zero-shot inference.",
        "supports_mcp": True,
    },
    {
        "slug": "chronos-2",
        "name": "Amazon Chronos-2",
        "model_type": "chronos",
        "description": "Amazon Chronos-2 pretrained transformer; zero-shot forecasting.",
        "supports_mcp": True,
    },
    {
        "slug": "timegpt",
        "name": "TimeGPT (Nixtla)",
        "model_type": "timegpt",
        "description": "Nixtla TimeGPT API-based foundation model; supports exogenous variables.",
        "supports_mcp": True,
    },
    {
        "slug": "ensemble-stack",
        "name": "Ensemble Stack",
        "model_type": "ensemble",
        "description": "Inverse-MAE weighted combiner. Set config={'stack_run_ids':[id1,id2,...]}.",
        "supports_mcp": False,
    },
]


async def _upsert_dataset(db: AsyncSession, slug: str, name: str, frequency: str, path: str) -> int:
    await db.execute(
        text(
            """
            INSERT INTO datasets (slug, name, frequency, source_path)
            VALUES (:slug, :name, :frequency, :path)
            ON CONFLICT (slug) DO NOTHING
            """
        ),
        {"slug": slug, "name": name, "frequency": frequency, "path": path},
    )
    row = await db.execute(text("SELECT id FROM datasets WHERE slug = :slug"), {"slug": slug})
    return row.scalar_one()


async def _upsert_series(
    db: AsyncSession, dataset_id: int, slug: str, name: str, unit: str
) -> int:
    await db.execute(
        text(
            """
            INSERT INTO series (dataset_id, slug, name, unit)
            VALUES (:dataset_id, :slug, :name, :unit)
            ON CONFLICT (dataset_id, slug) DO NOTHING
            """
        ),
        {"dataset_id": dataset_id, "slug": slug, "name": name, "unit": unit},
    )
    row = await db.execute(
        text("SELECT id FROM series WHERE dataset_id = :did AND slug = :slug"),
        {"did": dataset_id, "slug": slug},
    )
    return row.scalar_one()


async def _bulk_insert_observations(
    db: AsyncSession, series_id: int, timestamps: list, values: list
) -> int:
    if not timestamps:
        return 0
    rows = [
        {"series_id": series_id, "timestamp": ts.isoformat(), "value": float(v)}
        for ts, v in zip(timestamps, values)
        if v == v  # skip NaN
    ]
    if not rows:
        return 0
    await db.execute(
        text(
            """
            INSERT INTO observations (series_id, timestamp, value)
            SELECT :series_id, ts::timestamptz, val
            FROM (VALUES {placeholders}) AS t(ts, val)
            ON CONFLICT (series_id, timestamp) DO NOTHING
            """.format(
                placeholders=", ".join(
                    f"('{r['timestamp']}', {r['value']})" for r in rows
                )
            )
        ),
        {"series_id": series_id},
    )
    return len(rows)


async def _seed_model_catalog(db: AsyncSession) -> None:
    for m in MODEL_CATALOG_SEED:
        existing = await db.scalar(
            select(ModelCatalog).where(ModelCatalog.slug == m["slug"])
        )
        if existing:
            continue
        db.add(
            ModelCatalog(
                slug=m["slug"],
                name=m["name"],
                model_type=ModelType(m["model_type"]),
                description=m["description"],
                supports_mcp=m["supports_mcp"],
            )
        )
    await db.flush()
    log.info("model_catalog seeded (%d models)", len(MODEL_CATALOG_SEED))


async def load_all() -> None:
    async with AsyncSessionLocal() as db:
        await _seed_model_catalog(db)

        for filename, (slug, name, frequency, unit) in DATASETS.items():
            parquet_path = DATA_ROOT / filename
            if not parquet_path.exists():
                log.warning("parquet not found, skipping: %s", parquet_path)
                continue

            df = pd.read_parquet(parquet_path)
            # Normalise index to datetime
            df.index = pd.to_datetime(df.index, utc=True)

            dataset_id = await _upsert_dataset(db, slug, name, frequency, str(parquet_path))
            log.info("dataset '%s' id=%d", slug, dataset_id)

            total_obs = 0
            for col in df.columns:
                series_id = await _upsert_series(db, dataset_id, col, col, unit)
                n = await _bulk_insert_observations(
                    db, series_id, list(df.index), list(df[col])
                )
                total_obs += n

            await db.commit()
            log.info(
                "loaded dataset='%s' series=%d observations=%d",
                slug,
                len(df.columns),
                total_obs,
            )

    log.info("ETL complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(load_all())
