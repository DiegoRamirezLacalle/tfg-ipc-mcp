from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.datasets import router as datasets_router
from app.api.v1.drift import router as drift_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.health import router as health_router
from app.api.v1.metrics import router as metrics_router
from app.api.v1.news import router as news_router
from app.api.v1.runs import router as runs_router
from app.api.v1.users import router as users_router
from app.api.v1.whatif import router as whatif_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health_router)
v1_router.include_router(auth_router)
v1_router.include_router(users_router)
v1_router.include_router(datasets_router)
v1_router.include_router(experiments_router)
v1_router.include_router(runs_router)
v1_router.include_router(metrics_router)
v1_router.include_router(drift_router)
v1_router.include_router(whatif_router)
v1_router.include_router(news_router)
