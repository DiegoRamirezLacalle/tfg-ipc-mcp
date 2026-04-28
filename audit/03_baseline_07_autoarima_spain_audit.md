# Audit — 03_models_baseline/07_autoarima_spain.py

## Propósito
AutoARIMA dinámico para IPC España: re-selección de órdenes SARIMA(p,1,q)(P,1,Q)12 en cada origen rolling. Sin exógena.

## Inputs / Outputs
- **Reads**: `data/processed/features_exog.parquet` (usa `indice_general`).
- **Writes**:
  - `08_results/autoarima_spain_predictions.parquet`
  - `08_results/autoarima_spain_metrics.json`
  - `08_results/autoarima_spain_orders.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_TEST_END). NO usa `shared.metrics`.

## Métricas
- LOC: 211 · Funcionales ~165 · Comentarios/blank ~46 · `print(`: 23.

## Code smells
- Mismas que `07_autoarima_europe.py`/`global.py`.

## Riesgo de refactor
**ALTO**. AutoARIMA Spain está integrado en `metrics_summary_final.json` y aparece en notebooks de evaluación.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. Mantener parámetros `auto_arima` exactos.
4. NO cambiar esquemas de outputs.
