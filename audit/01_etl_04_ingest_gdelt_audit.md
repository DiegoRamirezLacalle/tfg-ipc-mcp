# Audit — 01_etl/04_ingest_gdelt.py

## Propósito
Stub vacío. PROJECT_CONTEXT lo describe como "GDELT BigQuery → news_signals.parquet", pero no hay implementación.

## Inputs / Outputs
- N/A.

## Dependencias internas
- N/A.

## Métricas
- LOC totales: 0.

## Code smells
- Stub vacío. La generación real de `news_signals.parquet` parece vivir en `05_mcp_pipeline/` (`agent_extractor.py`, `news_to_features.py`) y en algún paso fuera de este árbol.

## Riesgo de refactor
**BAJO**. Eliminar tras confirmar que nadie invoca el path por nombre.

## Acciones propuestas (FASE 3)
1. Verificar con grep que ningún script/notebook llama a `04_ingest_gdelt.py`.
2. Eliminar el archivo.
