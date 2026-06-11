# tfg-arquitectura

Web platform for the thesis experiments: run forecasting models (classical,
deep and foundation) on inflation series, with optional MCP semantic context,
and compare the results.

## Services

| Service     | Port  | Role                                                  |
|-------------|-------|-------------------------------------------------------|
| gateway     | 80    | nginx: static frontend build + `/api` reverse proxy   |
| frontend    | 3000  | Vite dev server (hot reload)                          |
| backend     | 8000  | FastAPI: auth, experiments, runs, metrics, what-if    |
| mcp_server  | 8080  | MCP server: news signals + FinBERT sentiment (SSE)    |
| postgres    | 5432  | datasets, series, experiments, runs, metrics          |
| mongo       | 27017 | cached news documents                                 |
| mlflow      | 5000  | run tracking                                          |

The backend talks to the MCP server over SSE to fetch exogenous news signals;
Postgres holds the experiment domain, Mongo caches ingested news, and MLflow
tracks runs.

## Quick start

From the repository root:

```bash
cp .env.example .env        # fill in POSTGRES_*, JWT_SECRET, NIXTLA_API_KEY
docker compose up -d --build
docker compose exec backend alembic upgrade head
docker compose exec backend python scripts/seed_ipc.py
# → http://localhost:3000 (dev) or http://localhost:80 (gateway build)
```

## Tests

```bash
docker compose exec backend pytest -q
```

Integration tests hit the real Postgres/Mongo from the compose stack.

## Seeding

| Script                        | What it does                                        |
|-------------------------------|-----------------------------------------------------|
| `scripts/seed_ipc.py`         | loads the processed parquets (IPC, CPI, HICP, exog) |
| `scripts/seed_experiments.py` | creates + triggers runs for the baseline models     |
| `scripts/seed_ablation.py`    | C0 vs C1_mcp ablation for the foundation models     |

## Layout

```
backend/     FastAPI app (api/v1, core, db, etl, forecasting, mcp, models,
             schemas, services), Alembic migrations, seed scripts, tests
frontend/    React + Vite + Tailwind (src/styles holds the design tokens)
gateway/     nginx config + two-stage Dockerfile (builds the frontend)
mcp_server/  standalone MCP server exposing news signals over SSE
```

Design mockups live in `../docs/design/stitch/`; development history in
`../docs/dev-diary.md`.
