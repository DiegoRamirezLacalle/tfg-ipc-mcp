import httpx
import pytest_asyncio
from sqlalchemy import text

from app.db.postgres import AsyncSessionLocal
from app.main import app


@pytest_asyncio.fixture(scope="session")
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest_asyncio.fixture(autouse=True)
async def clean_db():
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(
                "TRUNCATE TABLE metrics, predictions, runs, experiments, "
                "observations, series, datasets, model_catalog, users "
                "RESTART IDENTITY CASCADE"
            )
        )
        await db.commit()
    yield
