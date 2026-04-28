# Audit — 05_mcp_pipeline/mcp_client.py

## Propósito
Wrapper síncrono `MCPPipeline` sobre el cliente MCP async. Lanza `mcp_server.py` como subproceso vía stdio y expone métodos `fetch_gdelt`/`fetch_rss`. Usa thread dedicado con `run_coroutine_threadsafe` para evitar conflictos de cancel scope entre `asyncio` y `anyio` en Python 3.10.

## Inputs / Outputs
- **Reads**: nada directo (delega en mcp_server).
- **Writes**: nada directo.

## Dependencias internas
- Llama a `mcp_server.py` por path.

## Métricas
- LOC: 168 · Funcionales ~125 · Comentarios/blank ~43 · `print(`: bajo.

## Code smells
- Docstrings en español.
- Manejo manual de event loop en thread separado — workaround necesario por compatibilidad asyncio/anyio.
- `timeout=120` hardcoded.

## Riesgo de refactor
**ALTO**. Si la sincronización del event loop se rompe, el pipeline MCP completo deja de funcionar.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings.
3. NO cambiar la mecánica de threads + asyncio (workaround validado).
4. Mantener firma de `MCPPipeline` y métodos públicos.
