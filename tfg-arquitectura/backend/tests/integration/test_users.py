import pytest


@pytest.mark.asyncio
async def test_admin_promotes_viewer_to_researcher(client, admin, viewer):
    resp = await client.patch(
        f"/api/v1/users/{viewer['user'].id}/role",
        json={"role": "researcher"},
        headers=admin["headers"],
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "researcher"


@pytest.mark.asyncio
async def test_admin_demotes_researcher_to_viewer(client, admin, researcher):
    resp = await client.patch(
        f"/api/v1/users/{researcher['user'].id}/role",
        json={"role": "viewer"},
        headers=admin["headers"],
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "viewer"


@pytest.mark.asyncio
async def test_researcher_cannot_change_roles(client, researcher, viewer):
    resp = await client.patch(
        f"/api/v1/users/{viewer['user'].id}/role",
        json={"role": "researcher"},
        headers=researcher["headers"],
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_role_change_requires_auth(client, viewer):
    resp = await client.patch(
        f"/api/v1/users/{viewer['user'].id}/role",
        json={"role": "researcher"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_role_change_user_not_found(client, admin):
    resp = await client.patch(
        "/api/v1/users/9999/role",
        json={"role": "researcher"},
        headers=admin["headers"],
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_cannot_demote_self(client, admin):
    resp = await client.patch(
        f"/api/v1/users/{admin['user'].id}/role",
        json={"role": "viewer"},
        headers=admin["headers"],
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_promoted_user_can_create_experiment(client, admin, viewer, run_catalog):
    # promote viewer → researcher
    await client.patch(
        f"/api/v1/users/{viewer['user'].id}/role",
        json={"role": "researcher"},
        headers=admin["headers"],
    )
    # Token still has old role, but DB role is what's checked
    resp = await client.post(
        "/api/v1/experiments",
        json={
            "name": "post-promote exp",
            "series_id": run_catalog["series_id"],
            "model_id": run_catalog["naive_model_id"],
        },
        headers=viewer["headers"],
    )
    assert resp.status_code == 201
