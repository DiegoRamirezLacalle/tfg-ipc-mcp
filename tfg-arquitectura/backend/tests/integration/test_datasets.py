from datetime import datetime, timezone

import pytest
import pytest_asyncio

from app.db.postgres import AsyncSessionLocal
from app.models.dataset import Dataset, Observation, Series
from app.models.model_catalog import ModelCatalog, ModelType


@pytest_asyncio.fixture
async def seeded_dataset():
    """Insert one dataset with two series and a few observations, return ids."""
    async with AsyncSessionLocal() as db:
        ds = Dataset(slug="test-ds", name="Test Dataset", frequency="monthly")
        db.add(ds)
        await db.flush()

        s1 = Series(dataset_id=ds.id, slug="series-a", name="Series A", unit="index")
        s2 = Series(dataset_id=ds.id, slug="series-b", name="Series B", unit="%")
        db.add_all([s1, s2])
        await db.flush()

        obs = [
            Observation(series_id=s1.id, timestamp=datetime(2024, m, 1, tzinfo=timezone.utc), value=float(m))
            for m in range(1, 4)
        ]
        db.add_all(obs)
        await db.commit()

        return {"dataset_id": ds.id, "series_ids": [s1.id, s2.id]}


@pytest_asyncio.fixture
async def seeded_model():
    async with AsyncSessionLocal() as db:
        m = ModelCatalog(
            slug="test-model",
            name="Test Model",
            model_type=ModelType.arima,
            supports_mcp=False,
        )
        db.add(m)
        await db.commit()
        return m.id


@pytest.mark.asyncio
async def test_list_datasets_empty(client):
    resp = await client.get("/api/v1/datasets")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_datasets_returns_seeded(client, seeded_dataset):
    resp = await client.get("/api/v1/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["slug"] == "test-ds"
    assert data[0]["frequency"] == "monthly"


@pytest.mark.asyncio
async def test_get_dataset_404(client):
    resp = await client.get("/api/v1/datasets/9999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_dataset_ok(client, seeded_dataset):
    did = seeded_dataset["dataset_id"]
    resp = await client.get(f"/api/v1/datasets/{did}")
    assert resp.status_code == 200
    assert resp.json()["id"] == did


@pytest.mark.asyncio
async def test_list_series_for_dataset(client, seeded_dataset):
    did = seeded_dataset["dataset_id"]
    resp = await client.get(f"/api/v1/datasets/{did}/series")
    assert resp.status_code == 200
    slugs = {s["slug"] for s in resp.json()}
    assert slugs == {"series-a", "series-b"}


@pytest.mark.asyncio
async def test_list_series_dataset_404(client):
    resp = await client.get("/api/v1/datasets/9999/series")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_series_ok(client, seeded_dataset):
    sid = seeded_dataset["series_ids"][0]
    resp = await client.get(f"/api/v1/series/{sid}")
    assert resp.status_code == 200
    assert resp.json()["slug"] == "series-a"


@pytest.mark.asyncio
async def test_get_series_404(client):
    resp = await client.get("/api/v1/series/9999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_observations_returns_data(client, seeded_dataset):
    sid = seeded_dataset["series_ids"][0]
    resp = await client.get(f"/api/v1/series/{sid}/observations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert data[0]["value"] == 1.0


@pytest.mark.asyncio
async def test_observations_pagination(client, seeded_dataset):
    sid = seeded_dataset["series_ids"][0]
    resp = await client.get(f"/api/v1/series/{sid}/observations?limit=2&offset=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_observations_series_404(client):
    resp = await client.get("/api/v1/series/9999/observations")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_models_empty(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_models_returns_seeded(client, seeded_model):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["slug"] == "test-model"
    assert data[0]["model_type"] == "arima"
