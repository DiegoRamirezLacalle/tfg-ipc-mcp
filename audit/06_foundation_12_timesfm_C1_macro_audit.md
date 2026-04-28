# Audit — 06_models_foundation/12_timesfm_C1_macro.py

## Propósito
TimesFM IPC España con condición C1_macro (mix Brent + TTF + EPU). Variante compacta de `02_timesfm_C1.py` con `XREG_COVS` macro.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`.
- **Writes**:
  - `08_results/timesfm_C1_macro_predictions.parquet`
  - `08_results/timesfm_C1_macro_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 115 · Funcionales ~95 · Comentarios/blank ~20 · `print(`: ~5.

## Code smells
- Imports comprimidos.
- Estilo compacto.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar Ridge + StandardScaler ni `XREG_COVS`.
