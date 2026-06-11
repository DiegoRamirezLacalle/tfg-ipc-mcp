import httpx, time

BASE = "http://localhost:8000/api/v1"
client = httpx.Client(timeout=60)
token = client.post(f"{BASE}/auth/login", json={"email": "admin@tfg.local", "password": "changeme-dev"}).json()["access_token"]
h = {"Authorization": f"Bearer {token}"}

run_ids = [20, 21]
for _ in range(30):
    results = {rid: client.get(f"{BASE}/runs/{rid}", headers=h).json() for rid in run_ids}
    pending = [rid for rid, r in results.items() if r["status"] not in ("done", "failed")]
    for rid, r in results.items():
        if r["status"] in ("done", "failed"):
            icon = "✓" if r["status"] == "done" else "✗"
            print(f"  {icon} run {rid}: {r['status']}  {r.get('error_message','')[:100]}")
    if not pending:
        break
    print(f"  ... waiting ({len(pending)} still running)")
    time.sleep(8)

# Print metrics for done runs
print()
for rid in run_ids:
    r = client.get(f"{BASE}/runs/{rid}", headers=h).json()
    if r["status"] == "done":
        metrics = client.get(f"{BASE}/runs/{rid}/metrics", headers=h).json()
        m = {x["name"]: round(x["value"], 4) for x in metrics}
        print(f"run {rid} metrics: {m}")
