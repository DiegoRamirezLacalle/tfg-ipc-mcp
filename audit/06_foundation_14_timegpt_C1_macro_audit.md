# Audit — 06_models_foundation/14_timegpt_C1_macro.py

## Propósito
TimeGPT IPC España condición C1_macro (Brent + TTF + EPU). Estilo compacto.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C1_macro_predictions.parquet`
  - `08_results/timegpt_C1_macro_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 127 · Funcionales ~105 · Comentarios/blank ~22 · `print(`: ~6.

## Code smells
- Imports comprimidos.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar `EXOG_COLS`.
