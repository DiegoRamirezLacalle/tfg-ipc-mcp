# Audit — 01_etl/01_ingest_ipc_spain.py

## Propósito
Stub vacío. Por nombre y ubicación debería descargar/preparar el IPC de España (INE), pero el archivo tiene 0 bytes.

## Inputs / Outputs
- N/A (sin contenido).

## Dependencias internas
- N/A.

## Métricas
- LOC totales: 0.
- Sin código.

## Code smells
- Stub vacío que confunde. PROJECT_CONTEXT lo lista como activo (`Descarga IPC España (INE) → ipc_spain_index.parquet`), pero la lógica real vive en `05_clean_and_align.py` (que parsea `data/raw/IPC.xlsx` y produce `ipc_spain_index.parquet`).
- Probable artefacto de una reorganización abortada.

## Riesgo de refactor
**BAJO**. Eliminarlo no afecta a nadie (no tiene contenido). Antes de borrar, confirmar con un `grep` que ningún Makefile / notebook lo invoca por nombre.

## Acciones propuestas (FASE 3)
1. Verificar con grep/glob que nadie referencia este path.
2. Eliminar el archivo si no hay referencias.
3. Si se conserva, dejar al menos un docstring que apunte a `05_clean_and_align.py` como el script real.
