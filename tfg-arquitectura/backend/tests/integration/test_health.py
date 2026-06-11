import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_has_required_keys(client):
    resp = await client.get("/api/v1/health")
    body = resp.json()
    assert "status" in body
    assert "timestamp" in body
    assert "database" in body


@pytest.mark.asyncio
async def test_health_status_is_valid(client):
    resp = await client.get("/api/v1/health")
    assert resp.json()["status"] in ("ok", "degraded")
