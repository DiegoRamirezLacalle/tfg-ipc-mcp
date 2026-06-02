from datetime import datetime

from pydantic import BaseModel


class MetricValues(BaseModel):
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None


class ComparisonRow(BaseModel):
    experiment_id: int
    experiment_name: str
    model_slug: str
    model_name: str
    horizon: int
    use_mcp: bool
    run_id: int | None
    run_finished_at: datetime | None
    metrics: MetricValues | None
