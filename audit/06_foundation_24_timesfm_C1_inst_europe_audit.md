# Audit — 06_models_foundation/24_timesfm_C1_inst_europe.py

## Propósito
TimesFM HICP Eurozona condición C1_inst (Ridge externo + StandardScaler con señales institucionales).

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_europe.parquet`.
- **Writes**:
  - `08_results/timesfm_C1_inst_europe_predictions.parquet`
  - `08_results/timesfm_C1_inst_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 148 · Funcionales ~125 · Comentarios/blank ~23 · `print(`: ~6.

## Code smells
- Imports comprimidos.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar Ridge + StandardScaler ni `XREG_COVS`.
