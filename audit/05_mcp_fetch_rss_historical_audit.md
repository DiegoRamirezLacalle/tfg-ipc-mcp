# Audit — 05_mcp_pipeline/fetch_rss_historical.py

## Propósito
Descarga histórico de comunicados oficiales 2015-2024 de tres fuentes (BCE press releases, INE notas de prensa mensuales, BdE RSS), normaliza a JSON `data/raw/rss_raw/{source}/YYYY-MM.json` con caching (skipea si ya existe) y rate limiting (1s entre peticiones HTTP). Opcionalmente inserta en MongoDB.

## Inputs / Outputs
- **Reads**:
  - URLs de BCE / INE / BdE (HTTP).
- **Writes**:
  - `data/raw/rss_raw/{source}/YYYY-MM.json`
  - MongoDB (con `--mongo`).

## Dependencias internas
- Ninguna (no importa de `shared/`).

## Métricas
- LOC: 887 (el más largo del módulo) · Funcionales ~700 · Comentarios/blank ~187.

## Code smells
- Docstring/comentarios en español.
- 887 líneas — uno de los archivos más grandes; probablemente parsers HTML específicos por fuente con muchos selectors hardcoded.
- Posible duplicación entre los 3 fetchers (BCE / INE / BdE).
- `argparse` para CLI — buen patrón.
- Rate limiting hardcoded a 1s.

## Riesgo de refactor
**MEDIO**. Es ETL one-shot que ya pobló MongoDB. Romper los parsers no rompe modelos ya entrenados pero impide refresh. Salidas en `data/raw/rss_raw/` no son consumidas directamente por modelos.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Considerar dividir en helpers por fuente (BCE/INE/BdE) PERO el plan dice no introducir abstracciones — limitarse a logger + traducción.
4. NO cambiar formato del JSON normalizado ni nombres de archivos.
