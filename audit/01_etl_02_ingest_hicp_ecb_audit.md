# Audit — 01_etl/02_ingest_hicp_ecb.py

## Propósito
Stub vacío. Debería descargar el HICP de la Eurozona desde la ECB SDW. La lógica viva está en `11_ingest_hicp_europe.py`.

## Inputs / Outputs
- N/A.

## Dependencias internas
- N/A.

## Métricas
- LOC totales: 0.

## Code smells
- Igual que `01_ingest_ipc_spain.py`: archivo vacío referenciado en PROJECT_CONTEXT pero suplantado por `11_ingest_hicp_europe.py`.

## Riesgo de refactor
**BAJO**. Eliminar tras confirmar que ningún script/notebook lo invoca.

## Acciones propuestas (FASE 3)
1. Verificar con grep/glob que nadie depende de él.
2. Eliminar.
