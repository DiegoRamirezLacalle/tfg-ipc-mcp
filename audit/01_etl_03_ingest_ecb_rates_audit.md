# Audit — 01_etl/03_ingest_ecb_rates.py

## Propósito
Lee `DFR.csv` y `MRR.csv` (descargas BCE), resamplea a mensual con `last().ffill()` y exporta `ecb_rates_monthly.parquet`.

## Inputs / Outputs
- **Reads**: `data/raw/DFR.csv`, `data/raw/MRR.csv`.
- **Writes**: `data/processed/ecb_rates_monthly.parquet`.

## Dependencias internas
- Ninguna (no usa `shared/`).

## Métricas
- LOC totales: 74
- Funcionales: ~50
- Comentarios/docstrings/blank: ~24
- Debug prints: 7. Sin código muerto.

## Code smells
- Docstrings y comentarios en español.
- `DATE_END = "2025-06-01"` hardcoded — quizá obsoleto (la metodología termina en 2024-12).
- Validación de gaps con `print(f"[!] NaN tras ffill...")` en lugar de logger/excepción.
- Pequeño y legible — uno de los scripts más limpios del módulo.

## Riesgo de refactor
**BAJO**. Producto único: `ecb_rates_monthly.parquet`, consumido por varios scripts (06, 08, 10_global, 13_europe). Cambios en el resampling rompen pipelines downstream.

## Acciones propuestas (FASE 3)
1. Traducir docstrings/comentarios.
2. Sustituir `print` por logger.
3. Mantener `DATE_END` o moverlo a `shared/constants.py`.
4. Mantener firma de `load_rate(path, col_name)` y comportamiento bit-exacto del resample.
