"""Test fixtures.

CRITICAL SAFETY: the integration tests TRUNCATE the whole schema between tests.
To make that safe, every test run is redirected to a DEDICATED database whose
name ends in ``_test`` - set in the environment *before* the app is imported, so
the engine in ``app.db.postgres`` (and every module that imported
``AsyncSessionLocal`` from it) binds to the test database, never production.

A hard assertion guarantees we can never truncate a non-``_test`` database, even
if the environment is misconfigured.
"""

import os

# ── Redirect to dedicated test databases BEFORE importing the app ─────────────
_BASE_DB = os.environ.get("POSTGRES_DB", "tfg_experiments")
if not _BASE_DB.endswith("_test"):
    os.environ["POSTGRES_DB"] = f"{_BASE_DB}_test"

# Mongo is isolated the same way so run-scoped caches (mcp_contexts) never leak
# into - or out of - production. get_mongo_db() keys off settings.MONGO_DB.
_BASE_MONGO = os.environ.get("MONGO_DB", "tfg_news")
if not _BASE_MONGO.endswith("_test"):
    os.environ["MONGO_DB"] = f"{_BASE_MONGO}_test"

# Belt-and-suspenders: refuse to run if we are not pointed at *_test databases.
assert os.environ["POSTGRES_DB"].endswith("_test"), (
    "Refusing to run tests: POSTGRES_DB is not a *_test database "
    f"({os.environ['POSTGRES_DB']!r}). Tests TRUNCATE the schema and must never "
    "touch production."
)
assert os.environ["MONGO_DB"].endswith("_test"), (
    f"Refusing to run tests: MONGO_DB is not a *_test database "
    f"({os.environ['MONGO_DB']!r})."
)

import asyncpg  # noqa: E402
import httpx  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.mongo import get_mongo_db  # noqa: E402
from app.db.postgres import AsyncSessionLocal, Base, engine  # noqa: E402
from app.main import app  # noqa: E402


async def _ensure_test_database() -> None:
    """Create the dedicated test database if it does not exist yet."""
    admin_dsn = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/postgres"
    )
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", settings.POSTGRES_DB
        )
        if not exists:
            # Identifier is derived from POSTGRES_DB (asserted *_test above).
            await conn.execute(f'CREATE DATABASE "{settings.POSTGRES_DB}"')
    finally:
        await conn.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _setup_test_database():
    """Create the test DB and all tables once per session."""
    assert settings.POSTGRES_DB.endswith("_test")
    await _ensure_test_database()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest_asyncio.fixture(autouse=True)
async def clean_db():
    # Final guard right before the destructive statements.
    assert settings.POSTGRES_DB.endswith("_test"), "TRUNCATE blocked: not a _test DB"
    assert settings.MONGO_DB.endswith("_test"), "Mongo wipe blocked: not a _test DB"
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(
                "TRUNCATE TABLE metrics, predictions, runs, experiments, "
                "observations, series, datasets, model_catalog, users "
                "RESTART IDENTITY CASCADE"
            )
        )
        await db.commit()
    # Postgres IDs reset each test (RESTART IDENTITY), so run-scoped Mongo caches
    # would collide with stale docs - clear them too for true isolation.
    mongo_db = get_mongo_db()
    await mongo_db["mcp_contexts"].delete_many({})
    yield
