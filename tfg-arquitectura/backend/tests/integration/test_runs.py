import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

async def _make_experiment(client, headers, series_id, model_id, horizon=6, use_mcp=False):
    resp = await client.post(
        "/api/v1/experiments",
        json={
            "name": "Run test exp",
            "series_id": series_id,
            "model_id": model_id,
            "horizon": horizon,
            "use_mcp": use_mcp,
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _trigger_and_get(client, headers, exp_id):
    trigger = await client.post(f"/api/v1/experiments/{exp_id}/runs", headers=headers)
    assert trigger.status_code == 202, trigger.text
    run_id = trigger.json()["id"]
    detail = await client.get(f"/api/v1/runs/{run_id}", headers=headers)
    return run_id, detail.json()


# ── trigger run ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trigger_run_returns_202_pending(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
    )
    resp = await client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=researcher["headers"],
    )
    assert resp.status_code == 202
    body = resp.json()
    # Response body snapshots the run before the background task runs
    assert body["status"] == "pending"
    assert "id" in body


@pytest.mark.asyncio
async def test_trigger_run_naive_completes(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "done"
    assert detail["finished_at"] is not None


@pytest.mark.asyncio
async def test_trigger_run_sarima_completes(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["sarima_model_id"],
        horizon=3,
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "done"


@pytest.mark.asyncio
async def test_trigger_run_unsupported_model_fails(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["unsupported_model_id"],
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "failed"
    assert "not implemented" in detail["error_message"]


@pytest.mark.asyncio
async def test_trigger_run_use_mcp_stub(client, researcher, run_catalog):
    """use_mcp=True must NOT break execution in F6 (handled by stub log)."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
        use_mcp=True,
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "done"


@pytest.mark.asyncio
async def test_trigger_run_series_no_observations(client, researcher, catalog):
    """catalog fixture has a series with no observations → 422."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        catalog["series_id"], catalog["model_id"],
    )
    resp = await client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers=researcher["headers"],
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_trigger_run_requires_auth(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
    )
    resp = await client.post(f"/api/v1/experiments/{exp_id}/runs")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_trigger_run_other_user_forbidden(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
    )
    signup = await client.post(
        "/api/v1/auth/signup",
        json={"email": "other@test.com", "password": "password123"},
    )
    other_token = signup.json()["access_token"]
    resp = await client.post(
        f"/api/v1/experiments/{exp_id}/runs",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_trigger_run_experiment_not_found(client, researcher):
    resp = await client.post(
        "/api/v1/experiments/9999/runs",
        headers=researcher["headers"],
    )
    assert resp.status_code == 404


# ── get run ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_run_not_found(client, researcher):
    resp = await client.get("/api/v1/runs/9999", headers=researcher["headers"])
    assert resp.status_code == 404


# ── predictions ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_predictions(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
        horizon=6,
    )
    run_id, _ = await _trigger_and_get(client, researcher["headers"], exp_id)

    resp = await client.get(f"/api/v1/runs/{run_id}/predictions", headers=researcher["headers"])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 6
    for p in data:
        assert "timestamp" in p
        assert "value" in p


@pytest.mark.asyncio
async def test_get_predictions_paginated(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
        horizon=6,
    )
    run_id, _ = await _trigger_and_get(client, researcher["headers"], exp_id)

    resp = await client.get(
        f"/api/v1/runs/{run_id}/predictions?limit=3&offset=2",
        headers=researcher["headers"],
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 3


# ── metrics ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_metrics(client, researcher, run_catalog):
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
    )
    run_id, _ = await _trigger_and_get(client, researcher["headers"], exp_id)

    resp = await client.get(f"/api/v1/runs/{run_id}/metrics", headers=researcher["headers"])
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()}
    assert names == {"mae", "rmse", "mape"}
    for m in resp.json():
        assert m["value"] >= 0.0


@pytest.mark.asyncio
async def test_get_metrics_not_found(client, researcher):
    resp = await client.get("/api/v1/runs/9999/metrics", headers=researcher["headers"])
    assert resp.status_code == 404


# ── ridge-exog adapter ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trigger_run_ridge_exog_completes(client, researcher, run_catalog):
    """ridge-exog must complete in AR-only mode when features-exog dataset is absent."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["ridge_model_id"],
        horizon=6,
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "done"
    assert detail["finished_at"] is not None


@pytest.mark.asyncio
async def test_trigger_run_ridge_exog_has_metrics(client, researcher, run_catalog):
    """ridge-exog run produces MAE, RMSE, MAPE."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["ridge_model_id"],
        horizon=6,
    )
    run_id, _ = await _trigger_and_get(client, researcher["headers"], exp_id)
    resp = await client.get(f"/api/v1/runs/{run_id}/metrics", headers=researcher["headers"])
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()}
    assert names == {"mae", "rmse", "mape"}


# ── F7: foundation model adapters (graceful failure) ──────────────────────────

@pytest.mark.asyncio
async def test_trigger_run_timesfm_missing_lib(client, researcher, run_catalog):
    """TimesFM adapter must fail gracefully when timesfm package is not installed."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["timesfm_model_id"],
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "failed"
    assert detail["error_message"] is not None


@pytest.mark.asyncio
async def test_trigger_run_chronos_missing_lib(client, researcher, run_catalog):
    """Chronos-2 adapter must fail gracefully when chronos-forecasting is not installed."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["chronos_model_id"],
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "failed"
    assert detail["error_message"] is not None


@pytest.mark.asyncio
async def test_trigger_run_timegpt_no_api_key(client, researcher, run_catalog):
    """TimeGPT adapter must fail gracefully without NIXTLA_API_KEY."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["timegpt_model_id"],
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "failed"
    assert detail["error_message"] is not None


# ── F8: MCP context endpoint ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_mcp_context_not_found(client, researcher, run_catalog):
    """Runs without MCP context return 404 on the mcp-context endpoint."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
    )
    run_id, _ = await _trigger_and_get(client, researcher["headers"], exp_id)
    resp = await client.get(f"/api/v1/runs/{run_id}/mcp-context", headers=researcher["headers"])
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_trigger_run_use_mcp_completes_without_server(client, researcher, run_catalog):
    """use_mcp=True must not prevent run from completing even when MCP server is unreachable."""
    exp_id = await _make_experiment(
        client, researcher["headers"],
        run_catalog["series_id"], run_catalog["naive_model_id"],
        use_mcp=True,
    )
    run_id, detail = await _trigger_and_get(client, researcher["headers"], exp_id)
    assert detail["status"] == "done"
