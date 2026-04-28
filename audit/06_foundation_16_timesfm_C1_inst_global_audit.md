# Audit — 06_models_foundation/16_timesfm_C1_inst_global.py

## Propósito
TimesFM sobre CPI Global condición C1_inst (mismo conjunto de covariables que `15_chronos2_C1_inst_global.py`, vía Ridge externo + StandardScaler).

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_global_institutional.parquet`.
- **Writes**:
  - `08_results/timesfm_C1_inst_global_predictions.parquet`
  - `08_results/timesfm_C1_inst_global_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 147 · Funcionales ~120 · Comentarios/blank ~27 · `print(`: ~6.

## Code smells
- Imports comprimidos.
- Mismo patrón Ridge de la versión Spain pero con target `cpi_global_rate`.

## Riesgo de refactor
**ALTO**. Resultados publicados.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar Ridge + StandardScaler.
