# Audit — 04_models_deep/04_backtesting_rolling_deep_global.py

## Propósito
Rolling-origin de los 3 deep models sobre CPI Global (`cpi_global_rate`). Origenes trimestrales, `max_steps=300`. Output `deep_rolling_metrics_global.json`.

## Inputs / Outputs
- **Reads**: `data/processed/cpi_global_monthly.parquet`.
- **Writes**:
  - `08_results/deep_rolling_predictions_global.parquet`
  - `08_results/deep_rolling_metrics_global.json` (CONTRATO).

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 224 · Funcionales ~175 · Comentarios/blank ~49 · `print(`: ~29.

## Code smells
- Docstring/comentarios en español.
- Patrón paralelo a `04_backtesting_rolling_deep_europe.py`.
- MAE/RMSE/MASE re-implementados.

## Riesgo de refactor
**ALTO**. Producto Global publicado en PROJECT_CONTEXT y referenciado por notebooks.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener hiperparámetros, seed, ORIGIN_FREQ.
4. NO cambiar esquemas de salida.
