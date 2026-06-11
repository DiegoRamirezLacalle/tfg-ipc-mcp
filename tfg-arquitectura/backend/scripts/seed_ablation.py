"""
Ablation seed: C0 vs C1_mcp for the three foundation models.

Creates and triggers:
  - chronos-2    h=12  [no-MCP]   (C0)
  - chronos-2    h=12  [MCP]      (C1_mcp)
  - timesfm      h=12  [no-MCP]   (C0)
  - timesfm      h=12  [MCP]      (C1_mcp)
  - timegpt      h=12  [MCP]      (C1_mcp — C0 already seeded)

Run inside the backend container:
  docker compose exec backend python scripts/seed_ablation.py
"""

import os
import sys
import time

import httpx

BASE_URL       = "http://localhost:8000/api/v1"
ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL",    "admin@tfg.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme-dev")

EXPERIMENTS = [
    # (slug, use_mcp)
    ("chronos-2", False),
    ("chronos-2", True),
    ("timesfm",   False),
    ("timesfm",   True),
    ("timegpt",   True),   # C0 already seeded; add C1_mcp
]

TARGET_DATASET = "ipc-spain-ine"
TARGET_SERIES  = "indice_general"
HORIZON        = 12


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def login(client: httpx.Client) -> str:
    r = client.post(f"{BASE_URL}/auth/login",
                    json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    r.raise_for_status()
    print(f"Logged in as {ADMIN_EMAIL}")
    return r.json()["access_token"]


def resolve_series(client: httpx.Client, h: dict) -> int:
    r = client.get(f"{BASE_URL}/datasets", headers=h)
    r.raise_for_status()
    ds = next((d for d in r.json() if d["slug"] == TARGET_DATASET), None)
    if not ds:
        sys.exit(f"Dataset '{TARGET_DATASET}' not found — run seed_ipc.py first")

    r = client.get(f"{BASE_URL}/datasets/{ds['id']}/series", headers=h)
    r.raise_for_status()
    series = next((s for s in r.json() if s["slug"] == TARGET_SERIES), None)
    if not series:
        sys.exit(f"Series '{TARGET_SERIES}' not found")

    print(f"Series: {series['name']}  (id={series['id']})")
    return series["id"]


def resolve_models(client: httpx.Client, h: dict) -> dict[str, int]:
    r = client.get(f"{BASE_URL}/models", headers=h)
    r.raise_for_status()
    return {m["slug"]: m["id"] for m in r.json() if m["is_active"]}


def seed(client: httpx.Client, h: dict) -> list[tuple]:
    series_id  = resolve_series(client, h)
    model_map  = resolve_models(client, h)

    run_ids = []
    for slug, use_mcp in EXPERIMENTS:
        if slug not in model_map:
            print(f"  [SKIP] '{slug}' not in active model catalog")
            continue

        label    = "MCP" if use_mcp else "no-MCP"
        exp_name = f"{slug} h={HORIZON} [{label}]"

        r = client.post(f"{BASE_URL}/experiments", headers=h, json={
            "name":     exp_name,
            "series_id": series_id,
            "model_id":  model_map[slug],
            "horizon":   HORIZON,
            "use_mcp":   use_mcp,
        })
        r.raise_for_status()
        exp_id = r.json()["id"]

        r = client.post(f"{BASE_URL}/experiments/{exp_id}/runs", headers=h)
        r.raise_for_status()
        run_id = r.json()["id"]

        run_ids.append((slug, label, exp_id, run_id))
        print(f"  Queued: {exp_name}  → exp {exp_id}, run {run_id}")

    return run_ids


def wait(client: httpx.Client, h: dict, run_ids: list[tuple]):
    print(f"\n{len(run_ids)} runs queued.  Waiting (max 20 min)…\n")
    pending  = list(run_ids)
    deadline = time.time() + 1200

    while pending and time.time() < deadline:
        still = []
        for slug, label, exp_id, run_id in pending:
            r = client.get(f"{BASE_URL}/runs/{run_id}", headers=h)
            r.raise_for_status()
            st = r.json()["status"]
            if st in ("done", "failed"):
                icon = "✓" if st == "done" else "✗"
                print(f"  {icon} run {run_id}  {slug} [{label}]  → {st}")
                if st == "failed":
                    print(f"      {r.json().get('error_message','')[:140]}")
            else:
                still.append((slug, label, exp_id, run_id))
        pending = still
        if pending:
            print(f"    …{len(pending)} still running, sleeping 12s…")
            time.sleep(12)

    if pending:
        print(f"\n[TIMEOUT] {len(pending)} runs still in-flight.")
    else:
        print("\nAll runs completed.")

    # Print final metrics
    print("\n── Metrics ────────────────────────────────────────────────────────")
    print(f"{'slug':<14} {'cond':<8} {'run':>4}  {'MAE':>8} {'RMSE':>8} {'MAPE':>7}")
    for slug, label, exp_id, run_id in run_ids:
        r = client.get(f"{BASE_URL}/runs/{run_id}/metrics", headers=h)
        if r.status_code == 200:
            m = {x["name"]: x["value"] for x in r.json()}
            mae  = m.get("mae",  float("nan"))
            rmse = m.get("rmse", float("nan"))
            mape = m.get("mape", float("nan"))
            print(f"  {slug:<14} {label:<8} {run_id:>4}  {mae:>8.4f} {rmse:>8.4f} {mape:>6.2f}%")
        else:
            print(f"  {slug:<14} {label:<8} {run_id:>4}  (no metrics)")


def main():
    with httpx.Client(timeout=60) as client:
        token = login(client)
        h     = _headers(token)
        runs  = seed(client, h)
        wait(client, h, runs)


if __name__ == "__main__":
    main()
