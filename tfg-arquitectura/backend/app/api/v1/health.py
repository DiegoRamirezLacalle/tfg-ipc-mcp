from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db

router = APIRouter(tags=["system"])
log = structlog.get_logger()


@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)) -> dict:
    result: dict = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await db.execute(text("SELECT 1"))
        result["database"] = "ok"
    except Exception as exc:
        log.warning("health_db_fail", error=str(exc))
        result["status"] = "degraded"
        result["database"] = "fail"
    return result
