# Audit — 06_models_foundation/17_timegpt_C1_inst_global.py

## Propósito
TimeGPT sobre CPI Global condición C1_inst. Carga `.env`, exógenas con shift+1.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_global_institutional.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C1_inst_global_predictions.parquet`
  - `08_results/timegpt_C1_inst_global_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 165 · Funcionales ~135 · Comentarios/blank ~30 · `print(`: ~8.

## Code smells
- Imports comprimidos.
- API key check.
- PROJECT_CONTEXT documenta que TimeGPT con exógenas degrada severamente (+77% vs ARIMA h=12 en Global).

## Riesgo de refactor
**ALTO**. Resultados publicados.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar `EXOG_COLS` ni patrón shift+1.
