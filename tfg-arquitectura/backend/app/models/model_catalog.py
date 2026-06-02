import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base


class ModelType(str, enum.Enum):
    naive = "naive"
    arima = "arima"
    ridge = "ridge"
    timesfm = "timesfm"
    chronos = "chronos"
    timegpt = "timegpt"
    ensemble = "ensemble"


class ModelCatalog(Base):
    __tablename__ = "model_catalog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[ModelType] = mapped_column(
        Enum(ModelType, name="model_type"), nullable=False
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    supports_mcp: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
