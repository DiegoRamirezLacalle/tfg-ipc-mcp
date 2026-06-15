"""
Seed script: creates and triggers runs for all baseline models on indice_general.

Creates one experiment per model, both with and without MCP, then waits for
all runs to complete. Run inside the backend container:
  docker compose exec backend python scripts/seed_experiments.py
"""
import os
import time

import httpx

BASE_URL = "http://localhost:8000/api/v1"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@tfg.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme-dev")

# Models to run (slug -> display label)
MODELS_TO_RUN = [
    "naive-seasonal",
    "arima",
    "auto-arima",
    "sarima",
    "sarimax",
    "ridge-exog",
    "timegpt",
]

TARGET_DATASET_SLUG = "ipc-spain-ine"
TARGET_SERIES_SLUG  = "indice_general"
HORIZON             = 12
USE_MCP_VALUES      = [False, True]   # run each model with and without MCP


def login(client: httpx.Client) -> str:
    r = client.post(f"{BASE_URL}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    r.raise_for_status()
    token = r.json()["access_token"]
    print(f"Logged in as {ADMIN_EMAIL}")
    return token


def get_series_id(client: httpx.Client, headers: dict) -> int:
    r = client.get(f"{BASE_URL}/datasets", headers=headers)
    r.raise_for_status()
    ds = next((d for d in r.json() if d["slug"] == TARGET_DATASET_SLUG), None)
    if not ds:
        raise RuntimeError(f"Dataset '{TARGET_DATASET_SLUG}' not found - run seed_ipc.py first")

    r = client.get(f"{BASE_URL}/datasets/{ds['id']}/series", headers=headers)
    r.raise_for_status()
    series = next((s for s in r.json() if s["slug"] == TARGET_SERIES_SLUG), None)
    if not series:
        raise RuntimeError(f"Series '{TARGET_SERIES_SLUG}' not found in dataset")

    print(f"Series: {series['name']} (id={series['id']})")
    return series["id"]


def get_model_map(client: httpx.Client, headers: dict) -> dict[str, int]:
    r = client.get(f"{BASE_URL}/models", headers=headers)
    r.raise_for_status()
    return {m["slug"]: m["id"] for m in r.json() if m["is_active"]}


def seed(client: httpx.Client, headers: dict):
    series_id = get_series_id(client, headers)
    model_map = get_model_map(client, headers)

    run_ids = []

    for slug in MODELS_TO_RUN:
        if slug not in model_map:
            print(f"  [SKIP] model '{slug}' not in catalog")
            continue

        for use_mcp in USE_MCP_VALUES:
            # Skip MCP=true for models that don't benefit (naive)
            if use_mcp and slug == "naive-seasonal":
                continue

            label = "MCP" if use_mcp else "no-MCP"
            exp_name = f"{slug} h={HORIZON} [{label}]"

            # Create experiment
            payload = {
                "name": exp_name,
                "series_id": series_id,
                "model_id": model_map[slug],
                "horizon": HORIZON,
                "use_mcp": use_mcp,
            }
            r = client.post(f"{BASE_URL}/experiments", json=payload, headers=headers)
            r.raise_for_status()
            exp_id = r.json()["id"]

            # Trigger run
            r = client.post(f"{BASE_URL}/experiments/{exp_id}/runs", headers=headers)
            r.raise_for_status()
            run_id = r.json()["id"]
            run_ids.append((slug, label, exp_id, run_id))
            print(f"  Queued: {exp_name} -> experiment {exp_id}, run {run_id}")

    print(f"\n{len(run_ids)} runs queued. Waiting for completion...")
    return run_ids


def wait_for_runs(client: httpx.Client, headers: dict, run_ids: list):
    pending = list(run_ids)
    max_wait = 600   # 10 minutes
    start = time.time()

    while pending and time.time() - start < max_wait:
        still_pending = []
        for slug, label, exp_id, run_id in pending:
            r = client.get(f"{BASE_URL}/runs/{run_id}", headers=headers)
            r.raise_for_status()
            status = r.json()["status"]
            if status in ("done", "failed"):
                icon = "" if status == "done" else ""
                print(f"  {icon} {slug} [{label}] run {run_id}: {status}")
                if status == "failed":
                    error = r.json().get("error_message", "")
                    print(f"    Error: {error[:120]}")
            else:
                still_pending.append((slug, label, exp_id, run_id))
        pending = still_pending
        if pending:
            time.sleep(8)

    if pending:
        print(f"\n[TIMEOUT] {len(pending)} runs still pending after {max_wait}s")
    else:
        elapsed = int(time.time() - start)
        print(f"\nAll runs completed in {elapsed}s.")


def main():
    with httpx.Client(timeout=30) as client:
        token = login(client)
        headers = {"Authorization": f"Bearer {token}"}
        run_ids = seed(client, headers)
        wait_for_runs(client, headers, run_ids)


if __name__ == "__main__":
    main()
