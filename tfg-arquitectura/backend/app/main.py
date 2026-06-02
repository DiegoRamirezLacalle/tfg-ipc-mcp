from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.v1 import v1_router
from app.config import settings
from app.core.exceptions import http_exception_handler, unhandled_exception_handler
from app.core.logging import configure_logging
from app.core.security import hash_password
from app.db.mongo import close_mongo
from app.db.postgres import AsyncSessionLocal, engine
from app.models.user import User, UserRole

configure_logging()
log = structlog.get_logger()


async def _seed_admin() -> None:
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        existing = await db.scalar(select(User).where(User.email == settings.ADMIN_EMAIL))
        if existing:
            return
        admin = User(
            email=settings.ADMIN_EMAIL,
            password_hash=hash_password(settings.ADMIN_PASSWORD),
            role=UserRole.admin,
        )
        db.add(admin)
        await db.commit()
        log.info("admin_seeded", email=settings.ADMIN_EMAIL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        log.info("startup", database="postgres", status="connected")
        await _seed_admin()
    except Exception as exc:
        log.warning("startup", database="postgres", status="unavailable", error=str(exc))
    yield
    await engine.dispose()
    await close_mongo()
    log.info("shutdown", status="clean")


_TAGS = [
    {"name": "system",      "description": "Health check and platform status."},
    {"name": "auth",        "description": "JWT-based sign-up / sign-in."},
    {"name": "users",       "description": "Role management (admin only)."},
    {"name": "datasets",    "description": "Time-series datasets and model catalog."},
    {"name": "experiments", "description": "Forecast experiment lifecycle (CRUD)."},
    {"name": "runs",        "description": "Trigger and inspect forecast runs."},
    {"name": "metrics",     "description": "Cross-experiment metric comparison tables."},
]

app = FastAPI(
    title="TFG Inflation Forecasting Platform",
    version="0.1.0",
    description=(
        "REST API for managing, executing, and comparing inflation-forecasting experiments. "
        "Supports naive baselines, SARIMA, and foundation models (TimesFM, Chronos-2, TimeGPT). "
        "Optional MCP semantic context enrichment via the `use_mcp` flag."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
    openapi_tags=_TAGS,
)

_origins = (
    [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    if settings.CORS_ORIGINS
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(v1_router)
