# Entry point — delegates to app/main.py
# Usage: uvicorn main:app  OR  uvicorn app.main:app (both work)
from app.main import app  # noqa: F401
