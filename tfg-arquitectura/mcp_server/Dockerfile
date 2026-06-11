FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    UV_SYSTEM_PYTHON=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# Base deps (fast layer - no ML weights)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    "mcp[cli]>=1.0" \
    "pandas>=2.2" \
    "pyarrow>=16" \
    "numpy>=1.26" \
    "pymongo>=4.6" \
    "feedparser>=6.0" \
    "httpx>=0.27"

# Heavy ML layer - CPU PyTorch + FinBERT for news sentiment scoring
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2" \
    "transformers>=4.40"

COPY tfg-arquitectura/mcp_server/server.py ./tfg-arquitectura/mcp_server/server.py
COPY tfg-forecasting/data/processed        ./tfg-forecasting/data/processed

EXPOSE 8080
CMD ["python", "tfg-arquitectura/mcp_server/server.py"]
