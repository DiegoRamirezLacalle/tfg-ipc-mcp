import pytest

# -- helpers ------------------------------------------------------------------

async def _full_run(client, headers, series_id, model_id, horizon=6, name="exp"):
    """Create an experiment, trigger a run, wait for it to finish. Returns (exp_id, run_id)."""
    exp = await client.post(
        "/api/v1/experiments",
        json={"name": name, "series_id": series_id, "model_id": model_id, "horizon": horizon},
        headers=headers,
    )
    assert exp.status_code == 201, exp.text
    exp_id = exp.json()["id"]

    trigger = await client.post(f"/api/v1/experiments/{exp_id}/runs", headers=headers)
    assert trigger.status_code == 202, trigger.text
    run_id = trigger.json()["id"]

    detail = await client.get(f"/api/v1/runs/{run_id}", headers=headers)
    return exp_id, detail.json()["id"]


# -- compare -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_two_models(client, researcher, run_catalog):
    naive_exp_id, _ = await _full_run(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
        name="Naive exp",
    )
    sarima_exp_id, _ = await _full_run(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["sarima_model_id"],
        horizon=3, name="SARIMA exp",
    )

    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [naive_exp_id, sarima_exp_id]},
        headers=researcher["headers"],
    )
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2

    slugs = {r["model_slug"] for r in rows}
    assert slugs == {"naive-seasonal", "sarima"}

    for row in rows:
        assert row["metrics"] is not None
        assert row["metrics"]["mae"] is not None
        assert row["metrics"]["rmse"] is not None
        assert row["run_id"] is not None


@pytest.mark.asyncio
async def test_compare_no_done_run(client, researcher, catalog):
    """Experiment without a completed run returns a row with metrics=None."""
    exp = await client.post(
        "/api/v1/experiments",
        json={
            "name": "No-run exp",
            "series_id": catalog["series_id"],
            "model_id": catalog["model_id"],
        },
        headers=researcher["headers"],
    )
    exp_id = exp.json()["id"]

    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [exp_id]},
        headers=researcher["headers"],
    )
    assert resp.status_code == 200
    row = resp.json()[0]
    assert row["run_id"] is None
    assert row["metrics"] is None


@pytest.mark.asyncio
async def test_compare_preserves_order(client, researcher, run_catalog):
    """Response rows match the order of the requested experiment_ids."""
    exp1_id, _ = await _full_run(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"], name="A",
    )
    exp2_id, _ = await _full_run(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["sarima_model_id"], horizon=3, name="B",
    )

    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [exp2_id, exp1_id]},
        headers=researcher["headers"],
    )
    assert resp.status_code == 200
    rows = resp.json()
    assert rows[0]["experiment_id"] == exp2_id
    assert rows[1]["experiment_id"] == exp1_id


@pytest.mark.asyncio
async def test_compare_deduplicates_ids(client, researcher, run_catalog):
    """Duplicate experiment_ids are silently deduplicated."""
    exp_id, _ = await _full_run(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"], name="dedup",
    )

    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [exp_id, exp_id, exp_id]},
        headers=researcher["headers"],
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_compare_requires_auth(client, run_catalog, researcher):
    exp_id, _ = await _full_run(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"], name="auth test",
    )
    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [exp_id]},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_compare_not_found(client, researcher):
    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [9999]},
        headers=researcher["headers"],
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_compare_other_user_forbidden(client, researcher, admin, run_catalog):
    """A researcher cannot compare another user's experiment."""
    exp_id, _ = await _full_run(
        client, admin["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"], name="admin exp",
    )
    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [exp_id]},
        headers=researcher["headers"],
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_compare_admin_sees_any(client, researcher, admin, run_catalog):
    """Admin can compare experiments owned by any user."""
    exp_id, _ = await _full_run(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"], name="researcher exp",
    )
    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [exp_id]},
        headers=admin["headers"],
    )
    assert resp.status_code == 200
    assert resp.json()[0]["experiment_id"] == exp_id


@pytest.mark.asyncio
async def test_compare_too_many_ids(client, researcher):
    ids = list(range(1, 22))  # 21 > _MAX_EXPERIMENTS=20
    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": ids},
        headers=researcher["headers"],
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_compare_empty_returns_empty_list(client, researcher):
    resp = await client.get(
        "/api/v1/metrics/compare",
        params={},
        headers=researcher["headers"],
    )
    # FastAPI treats missing required Query(...) as 422
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_compare_response_shape(client, researcher, run_catalog):
    """Validate every field in the ComparisonRow schema is present and typed."""
    exp_id, _ = await _full_run(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"], name="shape test",
    )
    resp = await client.get(
        "/api/v1/metrics/compare",
        params={"experiment_ids": [exp_id]},
        headers=researcher["headers"],
    )
    assert resp.status_code == 200
    row = resp.json()[0]

    required_keys = {
        "experiment_id", "experiment_name", "model_slug", "model_name",
        "horizon", "use_mcp", "run_id", "run_finished_at", "metrics",
    }
    assert required_keys == set(row.keys())
    assert set(row["metrics"].keys()) == {"mae", "rmse", "mape"}
