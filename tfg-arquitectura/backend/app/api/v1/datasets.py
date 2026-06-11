from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.models.dataset import Dataset, Observation, Series
from app.models.model_catalog import ModelCatalog
from app.schemas.dataset import DatasetOut, ModelCatalogOut, ObservationOut, SeriesOut

router = APIRouter(tags=["datasets"])


@router.get("/datasets", response_model=list[DatasetOut])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(select(Dataset).order_by(Dataset.id))
    return rows.scalars().all()


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@router.get("/datasets/{dataset_id}/series", response_model=list[SeriesOut])
async def list_series(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    rows = await db.execute(
        select(Series).where(Series.dataset_id == dataset_id).order_by(Series.id)
    )
    return rows.scalars().all()


@router.get("/series/{series_id}", response_model=SeriesOut)
async def get_series(series_id: int, db: AsyncSession = Depends(get_db)):
    s = await db.get(Series, series_id)
    if not s:
        raise HTTPException(status_code=404, detail="Series not found")
    return s


@router.get("/series/{series_id}/observations", response_model=list[ObservationOut])
async def list_observations(
    series_id: int,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    s = await db.get(Series, series_id)
    if not s:
        raise HTTPException(status_code=404, detail="Series not found")
    rows = await db.execute(
        select(Observation)
        .where(Observation.series_id == series_id)
        .order_by(Observation.timestamp)
        .offset(offset)
        .limit(limit)
    )
    return rows.scalars().all()


@router.get("/models", response_model=list[ModelCatalogOut])
async def list_models(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(
        select(ModelCatalog).where(ModelCatalog.is_active == True).order_by(ModelCatalog.id)  # noqa: E712
    )
    return rows.scalars().all()
