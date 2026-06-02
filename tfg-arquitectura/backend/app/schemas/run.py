from datetime import datetime

from pydantic import BaseModel


class PredictionOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    timestamp: datetime
    value: float
    lower_ci: float | None
    upper_ci: float | None


class MetricOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    value: float


class RunOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    experiment_id: int
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    created_at: datetime


class RunDetailOut(RunOut):
    predictions: list[PredictionOut] = []
    metrics: list[MetricOut] = []
