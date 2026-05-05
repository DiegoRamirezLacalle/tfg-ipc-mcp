"""
mcp_server_global.py
---------------------
Servidor MCP (FastMCP, transporte stdio) para el pipeline semántico global.

Herramientas:
  1. fetch_fomc       - FOMC statements (Fed Reserve)
  2. fetch_ecb_press  - ECB press conference transcripts
  3. fetch_bls_cpi    - US BLS CPI news releases

Cada herramienta llama al scraper correspondiente, guarda en fichero JSON
y opcionalmente en MongoDB.

Requiere:
    pip install "mcp[cli]" httpx beautifulsoup4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

mcp = FastMCP(name="tfg-ipc-mcp-global")


@mcp.tool()
def fetch_fomc(start_year: int = 2002, end_year: int = 2024) -> str:
    """
    Descarga FOMC statements históricos (Federal Reserve).
    Guarda textos en data/raw/global_raw/fomc/{YYYY-MM}.json.
    Resumible: salta meses ya descargados.
    Devuelve resumen de descarga.
    """
    from fetch_fomc_historical import fetch_fomc as _fetch
    docs = _fetch(start_year=start_year, end_year=end_year, use_mongo=False)
    raw_dir = ROOT / "data" / "raw" / "global_raw" / "fomc"
    total_files = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
    return json.dumps({
        "source": "fomc",
        "new_docs": len(docs),
        "total_files": total_files,
        "range": f"{start_year}-{end_year}",
        "output_dir": str(raw_dir),
    })


@mcp.tool()
def fetch_ecb_press(start_year: int = 2002, end_year: int = 2024) -> str:
    """
    Descarga transcripciones de ECB press conferences históricas.
    Guarda en data/raw/global_raw/ecb_press/{YYYY-MM}.json.
    Resumible: salta meses ya descargados.
    Devuelve resumen de descarga.
    """
    from fetch_ecb_press_historical import fetch_ecb_press as _fetch
    docs = _fetch(start_year=start_year, end_year=end_year, use_mongo=False)
    raw_dir = ROOT / "data" / "raw" / "global_raw" / "ecb_press"
    total_files = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
    return json.dumps({
        "source": "ecb_press",
        "new_docs": len(docs),
        "total_files": total_files,
        "range": f"{start_year}-{end_year}",
        "output_dir": str(raw_dir),
    })


@mcp.tool()
def fetch_bls_cpi(start_year: int = 2002, end_year: int = 2024) -> str:
    """
    Descarga BLS CPI news releases históricas.
    Guarda en data/raw/global_raw/bls_cpi/{YYYY-MM}.json.
    Resumible: salta meses ya descargados.
    Devuelve resumen de descarga.
    """
    from fetch_bls_cpi_historical import fetch_bls_cpi as _fetch
    docs = _fetch(start_year=start_year, end_year=end_year, use_mongo=False)
    raw_dir = ROOT / "data" / "raw" / "global_raw" / "bls_cpi"
    total_files = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
    return json.dumps({
        "source": "bls_cpi",
        "new_docs": len(docs),
        "total_files": total_files,
        "range": f"{start_year}-{end_year}",
        "output_dir": str(raw_dir),
    })


@mcp.tool()
def get_pipeline_status() -> str:
    """
    Devuelve el estado actual del pipeline global:
    cuántos ficheros hay descargados por fuente y cuántos procesados.
    """
    status = {}
    for source in ("fomc", "ecb_press", "bls_cpi"):
        raw_dir = ROOT / "data" / "raw" / "global_raw" / source
        if not raw_dir.exists():
            status[source] = {"total": 0, "processed": 0}
            continue

        files = list(raw_dir.glob("*.json"))
        total = len(files)
        processed = 0
        for f in files:
            try:
                docs = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(docs, list) and docs and docs[0].get("processed"):
                    processed += 1
            except Exception:
                pass
        status[source] = {"total": total, "processed": processed}

    # Parquet de señales
    parquet_path = ROOT / "data" / "processed" / "mcp_signals_global.parquet"
    features_path = ROOT / "data" / "processed" / "features_c1_global_full.parquet"
    status["mcp_signals_parquet"] = parquet_path.exists()
    status["features_c1_full_parquet"] = features_path.exists()

    return json.dumps(status, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
