# Audit — 05_mcp_pipeline/mcp_server.py

## Propósito
Servidor MCP (FastMCP, transporte stdio) con 5 herramientas: `fetch_gdelt_spain`, `fetch_rss_official`, `search_news`, `get_macro_news`, `get_entity_news`. Almacena resultados en MongoDB (`news_raw`) y parquets versionados. Es el backend de adquisición del pipeline MCP de España.

## Inputs / Outputs
- **Reads**: 
  - GDELT v2 vía HTTP (CSV.zip por timestamp).
  - RSS feeds (BCE, INE, BdE).
- **Writes**:
  - `data/raw/gdelt_spain_raw/` (CSVs descargados).
  - MongoDB `tfg_ipc_mcp.news_raw`.

## Dependencias internas
- Ninguna (no importa de `shared/`).

## Métricas
- LOC: 402 · Funcionales ~310 · Comentarios/blank ~92 · `print(`: alto (a verificar).

## Code smells
- Docstring/comentarios en español.
- MongoDB hardcoded `localhost:27017` — aceptable para investigación.
- Constantes magic numbers (`TOP_N_PER_DAY = 100`, `EVENT_CODE_MIN=100`).
- 5 tools con lógica HTTP + parsing + persistencia mezclada.
- Probablemente prints para debug.

## Riesgo de refactor
**ALTO**. Es el core del pipeline MCP que llena MongoDB. Aunque se ejecuta una vez por refresh, cualquier cambio en la lógica de filtrado GDELT (event codes, columnas) cambia los datos brutos almacenados. Bit-exactitud no aplica directamente (los datos están cacheados), pero romper el contrato del API rompe `agent_extractor.py` y `news_to_features.py`.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener firmas de las 5 tools (esquema MCP).
4. Mantener constantes y filtros GDELT exactos.
5. NO cambiar nombres de colecciones MongoDB.
