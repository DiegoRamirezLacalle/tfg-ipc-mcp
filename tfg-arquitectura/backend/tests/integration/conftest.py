from datetime import datetime, timezone

import pytest_asyncio

from app.core.security import create_token, hash_password
from app.db.postgres import AsyncSessionLocal
from app.models.dataset import Dataset, Observation, Series
from app.models.model_catalog import ModelCatalog, ModelType
from app.models.user import User, UserRole


def _auth_headers(user: User) -> dict:
    token = create_token(str(user.id), {"email": user.email, "role": user.role.value})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def researcher():
    async with AsyncSessionLocal() as db:
        user = User(
            email="researcher@test.com",
            password_hash=hash_password("pass123"),
            role=UserRole.researcher,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return {"user": user, "headers": _auth_headers(user)}


@pytest_asyncio.fixture
async def admin():
    async with AsyncSessionLocal() as db:
        user = User(
            email="admin@test.com",
            password_hash=hash_password("pass123"),
            role=UserRole.admin,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return {"user": user, "headers": _auth_headers(user)}


@pytest_asyncio.fixture
async def viewer():
    async with AsyncSessionLocal() as db:
        user = User(
            email="viewer@test.com",
            password_hash=hash_password("pass123"),
            role=UserRole.viewer,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return {"user": user, "headers": _auth_headers(user)}


@pytest_asyncio.fixture
async def catalog(researcher):
    """Seed one series + one model (no observations); return their ids."""
    async with AsyncSessionLocal() as db:
        ds = Dataset(slug="test-ds", name="Test DS", frequency="monthly")
        db.add(ds)
        await db.flush()
        s = Series(dataset_id=ds.id, slug="s1", name="S1", unit="index")
        db.add(s)
        m = ModelCatalog(
            slug="sarima-test",
            name="SARIMA test",
            model_type=ModelType.arima,
        )
        db.add(m)
        await db.flush()
        sid, mid = s.id, m.id
        await db.commit()
    return {"series_id": sid, "model_id": mid}


@pytest_asyncio.fixture
async def run_catalog(researcher):
    """Seed a series with 48 monthly observations + naive + sarima model entries."""
    async with AsyncSessionLocal() as db:
        ds = Dataset(slug="run-ds", name="Run DS", frequency="monthly")
        db.add(ds)
        await db.flush()

        s = Series(dataset_id=ds.id, slug="run-series", name="Run Series", unit="index")
        db.add(s)
        await db.flush()

        # 48 synthetic monthly observations starting 2020-01-01
        # Use a simple trend+seasonal pattern so SARIMA converges fast
        for i in range(48):
            month = i % 12
            value = 100.0 + i * 0.1 + (2.0 if month in (6, 7) else 0.0)
            ts = datetime(2020 + i // 12, (month) + 1, 1, tzinfo=timezone.utc)
            db.add(Observation(series_id=s.id, timestamp=ts, value=value))

        m_naive = ModelCatalog(
            slug="naive-seasonal",
            name="Naive Seasonal",
            model_type=ModelType.naive,
        )
        m_sarima = ModelCatalog(
            slug="sarima",
            name="SARIMA",
            model_type=ModelType.arima,
        )
        # Slug not in registry → tests "not implemented" path
        m_unsupported = ModelCatalog(
            slug="not-a-model",
            name="Not A Model",
            model_type=ModelType.naive,
        )
        m_ridge = ModelCatalog(
            slug="ridge-exog",
            name="Ridge Exog",
            model_type=ModelType.ridge,
        )
        m_timesfm = ModelCatalog(
            slug="timesfm",
            name="TimesFM",
            model_type=ModelType.timesfm,
        )
        m_chronos = ModelCatalog(
            slug="chronos-2",
            name="Chronos-2",
            model_type=ModelType.chronos,
        )
        m_timegpt = ModelCatalog(
            slug="timegpt",
            name="TimeGPT",
            model_type=ModelType.timegpt,
        )
        db.add_all([m_naive, m_sarima, m_unsupported, m_ridge, m_timesfm, m_chronos, m_timegpt])
        await db.flush()

        ids = {
            "series_id": s.id,
            "naive_model_id": m_naive.id,
            "sarima_model_id": m_sarima.id,
            "unsupported_model_id": m_unsupported.id,
            "ridge_model_id": m_ridge.id,
            "timesfm_model_id": m_timesfm.id,
            "chronos_model_id": m_chronos.id,
            "timegpt_model_id": m_timegpt.id,
        }
        await db.commit()
    return ids
