# Build context: repo root (docker-compose sets context: .)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Make both the backend package and the shared/ utilities importable
ENV PYTHONPATH=/app/tfg-arquitectura/backend:/app \
    UV_SYSTEM_PYTHON=1

# Install backend deps (cached layer - only re-runs if this list changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.29" \
    "sqlalchemy[asyncio]>=2.0" \
    "asyncpg>=0.29" \
    "alembic>=1.13" \
    "pydantic-settings>=2.2" \
    "pyjwt>=2.9" \
    "bcrypt>=4.2" \
    "motor>=3.4" \
    "structlog>=24.4" \
    "httpx>=0.27" \
    "python-multipart>=0.0.9" \
    "email-validator>=2.1" \
    "pytest>=8.0" \
    "pytest-asyncio>=0.23" \
    "pandas>=2.2" \
    "pyarrow>=16" \
    "statsmodels>=0.14" \
    "scipy>=1.13" \
    "scikit-learn>=1.4" \
    "mcp[cli]>=1.0" \
    "pmdarima>=2.0" \
    "nixtla>=0.5" \
    "mlflow>=2.16"

# -- Heavy ML layer - CPU PyTorch + foundation-model packages ----------------
# Separate layer so the base deps cache is not invalidated on ML changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2" \
    "chronos-forecasting>=2.0" \
    "timesfm"

# Copy application code (overridden by bind-mount in dev)
COPY tfg-arquitectura/backend/app        ./tfg-arquitectura/backend/app
COPY tfg-arquitectura/backend/migrations  ./tfg-arquitectura/backend/migrations
COPY tfg-arquitectura/backend/alembic.ini ./tfg-arquitectura/backend/alembic.ini
COPY shared                              ./shared

WORKDIR /app/tfg-arquitectura/backend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
