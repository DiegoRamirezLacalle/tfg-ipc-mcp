import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _exp_payload(catalog, **overrides):
    return {
        "name": "Test experiment",
        "series_id": catalog["series_id"],
        "model_id": catalog["model_id"],
        "horizon": 12,
        "use_mcp": False,
        **overrides,
    }


# ── create ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_experiment_researcher(client, researcher, catalog):
    resp = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog),
        headers=researcher["headers"],
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Test experiment"
    assert body["horizon"] == 12
    assert body["status"] == "created"
    assert body["user_id"] == researcher["user"].id


@pytest.mark.asyncio
async def test_create_experiment_admin(client, admin, catalog):
    resp = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog, name="Admin exp"),
        headers=admin["headers"],
    )
    assert resp.status_code == 201


@pytest.mark.asyncio
async def test_create_experiment_viewer_forbidden(client, viewer, catalog):
    resp = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog),
        headers=viewer["headers"],
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_create_experiment_no_auth(client, catalog):
    resp = await client.post("/api/v1/experiments", json=_exp_payload(catalog))
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_create_experiment_invalid_series(client, researcher, catalog):
    resp = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog, series_id=9999),
        headers=researcher["headers"],
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_experiment_invalid_model(client, researcher, catalog):
    resp = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog, model_id=9999),
        headers=researcher["headers"],
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_experiment_horizon_out_of_range(client, researcher, catalog):
    resp = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog, horizon=0),
        headers=researcher["headers"],
    )
    assert resp.status_code == 422


# ── list ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_experiments_empty(client, researcher):
    resp = await client.get("/api/v1/experiments", headers=researcher["headers"])
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_experiments_own_only(client, researcher, admin, catalog):
    # researcher creates one, admin creates one
    await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog, name="Researcher exp"),
        headers=researcher["headers"],
    )
    await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog, name="Admin exp"),
        headers=admin["headers"],
    )

    r_resp = await client.get("/api/v1/experiments", headers=researcher["headers"])
    assert len(r_resp.json()) == 1
    assert r_resp.json()[0]["name"] == "Researcher exp"


@pytest.mark.asyncio
async def test_admin_sees_all_experiments(client, researcher, admin, catalog):
    await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog, name="Researcher exp"),
        headers=researcher["headers"],
    )
    a_resp = await client.get("/api/v1/experiments", headers=admin["headers"])
    assert len(a_resp.json()) >= 1


@pytest.mark.asyncio
async def test_list_experiments_requires_auth(client):
    resp = await client.get("/api/v1/experiments")
    assert resp.status_code == 401


# ── detail ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_experiment_detail(client, researcher, catalog):
    create = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog),
        headers=researcher["headers"],
    )
    exp_id = create.json()["id"]

    resp = await client.get(f"/api/v1/experiments/{exp_id}", headers=researcher["headers"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == exp_id
    assert body["runs"] == []


@pytest.mark.asyncio
async def test_get_experiment_detail_not_found(client, researcher):
    resp = await client.get("/api/v1/experiments/9999", headers=researcher["headers"])
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_experiment_detail_other_user_forbidden(client, researcher, admin, catalog):
    create = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog),
        headers=researcher["headers"],
    )
    exp_id = create.json()["id"]

    # admin can see it
    resp = await client.get(f"/api/v1/experiments/{exp_id}", headers=admin["headers"])
    assert resp.status_code == 200

    # viewer cannot
    viewer_resp = await client.post(
        "/api/v1/auth/signup",
        json={"email": "v2@test.com", "password": "password123"},
    )
    viewer_token = viewer_resp.json()["access_token"]
    resp2 = await client.get(
        f"/api/v1/experiments/{exp_id}",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert resp2.status_code == 403


# ── delete ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_experiment(client, researcher, catalog):
    create = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog),
        headers=researcher["headers"],
    )
    exp_id = create.json()["id"]

    del_resp = await client.delete(f"/api/v1/experiments/{exp_id}", headers=researcher["headers"])
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/api/v1/experiments/{exp_id}", headers=researcher["headers"])
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_experiment_other_user_forbidden(client, researcher, admin, catalog):
    create = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog, name="R exp"),
        headers=researcher["headers"],
    )
    exp_id = create.json()["id"]

    # admin CAN delete
    resp = await client.delete(f"/api/v1/experiments/{exp_id}", headers=admin["headers"])
    assert resp.status_code == 204


# ── list runs ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_runs_empty(client, researcher, catalog):
    create = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog),
        headers=researcher["headers"],
    )
    exp_id = create.json()["id"]
    resp = await client.get(f"/api/v1/experiments/{exp_id}/runs", headers=researcher["headers"])
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_runs_after_trigger(client, researcher, run_catalog):
    exp_resp = await client.post(
        "/api/v1/experiments",
        json={
            "name": "with runs",
            "series_id": run_catalog["series_id"],
            "model_id": run_catalog["naive_model_id"],
            "horizon": 6,
        },
        headers=researcher["headers"],
    )
    exp_id = exp_resp.json()["id"]
    await client.post(f"/api/v1/experiments/{exp_id}/runs", headers=researcher["headers"])
    await client.post(f"/api/v1/experiments/{exp_id}/runs", headers=researcher["headers"])

    resp = await client.get(f"/api/v1/experiments/{exp_id}/runs", headers=researcher["headers"])
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_list_runs_other_user_forbidden(client, researcher, catalog):
    create = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog),
        headers=researcher["headers"],
    )
    exp_id = create.json()["id"]
    signup = await client.post(
        "/api/v1/auth/signup",
        json={"email": "third@test.com", "password": "password123"},
    )
    other_token = signup.json()["access_token"]
    resp = await client.get(
        f"/api/v1/experiments/{exp_id}/runs",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_runs_experiment_not_found(client, researcher):
    resp = await client.get("/api/v1/experiments/9999/runs", headers=researcher["headers"])
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_experiment_viewer_forbidden(client, viewer, researcher, catalog):
    create = await client.post(
        "/api/v1/experiments",
        json=_exp_payload(catalog),
        headers=researcher["headers"],
    )
    exp_id = create.json()["id"]

    resp = await client.delete(f"/api/v1/experiments/{exp_id}", headers=viewer["headers"])
    assert resp.status_code == 403
