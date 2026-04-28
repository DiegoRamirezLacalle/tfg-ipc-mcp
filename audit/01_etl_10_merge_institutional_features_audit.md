# Audit — 01_etl/10_merge_institutional_features.py

## Propósito
Mergea solo 3 columnas de EPU Europe (`epu_europe_log`, `epu_europe_ma3`, `epu_europe_lag1`) desde `institutional_signals_monthly.parquet` en `features_c1.parquet`, con forward-fill de hasta 2 NaN.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`, `data/processed/institutional_signals_monthly.parquet`.
- **Writes**: `data/processed/features_c1.parquet` (sobreescribe).

## Dependencias internas
- `import sys` no usado.

## Métricas
- LOC totales: 69
- Funcionales: ~50
- Comentarios/docstrings/blank: ~19
- Debug prints: ~10.

## Code smells
- `import sys` muerto.
- Docstring en inglés (anomalía positiva en este módulo).
- Comentarios mezclados ES/EN.
- Sobreescribe parquet de entrada (igual smell que `08_merge_energy_features.py`).
- Conflicto de nombres: hay otro archivo `10_ingest_institutional_signals_global.py` en la misma carpeta — ambos empiezan con `10_`. La numeración debería renombrarse para evitar ambigüedad pero el plan dice **no cambiar nombres de scripts**.

## Riesgo de refactor
**MEDIO-ALTO**. Toca `features_c1.parquet`, base de TODOS los modelos C1 de España. Mantener orden de columnas y comportamiento del ffill exacto.

## Acciones propuestas (FASE 3)
1. Eliminar `import sys`.
2. Traducir comentarios sueltos a inglés.
3. Sustituir prints por logger.
4. Mantener nombre del script (regla del refactor) aunque haya colisión de prefijo.
5. NO cambiar `COLS_TO_MERGE` ni el ffill(limit=2).
