from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.run import RunOut  # noqa: F401 — re-exported for backwards compat


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    series_id: int
    model_id: int
    horizon: int = Field(default=12, ge=1, le=60)
    use_mcp: bool = False
    config: dict[str, Any] | None = None


class ExperimentOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    user_id: int
    name: str
    series_id: int
    model_id: int
    horizon: int
    use_mcp: bool
    config: dict[str, Any] | None
    status: str
    created_at: datetime
    updated_at: datetime


class ExperimentDetailOut(ExperimentOut):
    runs: list[RunOut] = []
