# Audit — shared/constants.py

## Propósito
Constantes globales (IDs de series, ventanas temporales train/val/test, condiciones C0/C1, frecuencia mensual) compartidas entre `tfg-forecasting/` y `tfg-arquitectura/`.

## Inputs / Outputs
- **Reads**: nada (solo definiciones).
- **Writes**: nada.

## Dependencias internas
- Ninguna.

## Métricas
- LOC totales: 28
- Funcionales: ~14
- Comentarios/docstrings/blank: ~14
- Sin debug.

## Code smells
- Comentarios de sección (`# ── ... ──`) y docstring de módulo en español.
- `SERIES_ECB_RATE` y la lista `ALL_SERIES` no parecen usados por ningún script de modelado (a verificar con grep antes de eliminar).
- `DATE_VAL_END = "2022-06-01"` no coincide con la metodología documentada en `PROJECT_CONTEXT.md` (test rolling 2021-01 a 2024-12, sin val explícito separado). Posible constante stale.
- Mezcla de IDs de serie con configuración de splits y horizonte en un mismo módulo: aceptable a este tamaño.

## Riesgo de refactor
**BAJO**. Solo strings y constantes. Cualquier consumer importa por nombre, así que renombres rompen muchos sitios — pero como el contenido es trivial, basta con traducir comentarios.

## Acciones propuestas (FASE 3)
1. Traducir docstring y comentarios de sección a inglés.
2. Verificar uso real de `SERIES_HICP_EA`, `SERIES_ECB_RATE`, `ALL_SERIES`, `DATE_VAL_END`, `CONDITION_C0`, `CONDITION_C1`, `FORECAST_HORIZON`. Eliminar las que no estén en uso.
3. Mantener nombres y valores numéricos exactos.
