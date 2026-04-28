# Audit — 03_models_baseline/07_autoarima_global.py

## Propósito
AutoARIMA dinámico para `cpi_global_rate`: re-selección de órdenes en cada origen rolling con `auto_arima(seasonal=False, ...)` (sin componente estacional, consistente con el ARIMA fijo de Global). Sin exógena.

## Inputs / Outputs
- **Reads**: `data/processed/cpi_global_monthly.parquet`.
- **Writes**:
  - `08_results/autoarima_global_predictions.parquet`
  - `08_results/autoarima_global_metrics.json`
  - `08_results/autoarima_global_orders.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_TEST_END). NO usa `shared.metrics`.

## Métricas
- LOC: 213 · Funcionales ~165 · Comentarios/blank ~48 · `print(`: 23.

## Code smells
- Mismas que `07_autoarima_europe.py`: docstrings ES, MAE/RMSE/MASE in-place, prints, try/except genérico.

## Riesgo de refactor
**ALTO**. AutoARIMA Global es el modelo que mejora a ARIMA fijo en Global (PROJECT_CONTEXT) — números publicados en la memoria.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. Mantener parámetros EXACTOS de `auto_arima`.
4. NO cambiar esquemas de outputs.
