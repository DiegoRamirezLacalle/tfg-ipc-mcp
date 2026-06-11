from datetime import datetime

from pydantic import BaseModel


class DatasetOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    slug: str
    name: str
    description: str | None
    frequency: str
    version: str
    created_at: datetime


class SeriesOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    dataset_id: int
    name: str
    slug: str
    unit: str | None
    description: str | None
    created_at: datetime


class ObservationOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    timestamp: datetime
    value: float


class ModelCatalogOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    slug: str
    name: str
    model_type: str
    description: str | None
    supports_mcp: bool
    is_active: bool
