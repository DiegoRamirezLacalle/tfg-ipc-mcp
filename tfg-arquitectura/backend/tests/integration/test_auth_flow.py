import pytest


@pytest.mark.asyncio
async def test_signup_returns_token(client):
    resp = await client.post(
        "/api/v1/auth/signup",
        json={"email": "researcher@test.com", "password": "password123"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "researcher@test.com"
    assert body["user"]["role"] == "viewer"


@pytest.mark.asyncio
async def test_signup_duplicate_email_returns_409(client):
    payload = {"email": "dup@test.com", "password": "password123"}
    await client.post("/api/v1/auth/signup", json=payload)
    resp = await client.post("/api/v1/auth/signup", json=payload)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_signup_short_password_returns_422(client):
    resp = await client.post(
        "/api/v1/auth/signup",
        json={"email": "short@test.com", "password": "abc"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_login_valid_credentials(client):
    await client.post(
        "/api/v1/auth/signup",
        json={"email": "login@test.com", "password": "password123"},
    )
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "login@test.com", "password": "password123"},
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_login_wrong_password_returns_401(client):
    await client.post(
        "/api/v1/auth/signup",
        json={"email": "wrongpw@test.com", "password": "password123"},
    )
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "wrongpw@test.com", "password": "wrongpassword"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_with_valid_token(client):
    signup = await client.post(
        "/api/v1/auth/signup",
        json={"email": "me@test.com", "password": "password123"},
    )
    token = signup.json()["access_token"]
    resp = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "me@test.com"


@pytest.mark.asyncio
async def test_me_without_token_returns_401(client):
    resp = await client.get("/api/v1/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_role_not_assignable_via_public_signup(client):
    resp = await client.post(
        "/api/v1/auth/signup",
        json={"email": "notadmin@test.com", "password": "password123", "role": "admin"},
    )
    assert resp.status_code == 201
    assert resp.json()["user"]["role"] == "viewer"
