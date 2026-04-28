# Audit — 03_models_baseline/07_autoarima_europe.py

## Propósito
AutoARIMA dinámico para HICP Eurozona: en cada uno de los 48 orígenes rolling re-selecciona los órdenes vía `pmdarima.auto_arima(seasonal=True, m=12, ...)`. Sin exógena. MASE scale fija. Genera predicciones, métricas y log de órdenes seleccionados por origen.

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet`.
- **Writes**:
  - `08_results/autoarima_europe_predictions.parquet`
  - `08_results/autoarima_europe_metrics.json`
  - `08_results/autoarima_europe_orders.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_TEST_END). NO usa `shared.metrics`.

## Métricas
- LOC: 212 · Funcionales ~165 · Comentarios/blank ~47 · `print(`: 23.

## Code smells
- Docstring/comentarios en español.
- 23 prints (incluye warnings de `auto_arima` con `try/except`).
- MAE/RMSE/MASE re-implementados in-place.
- `tqdm` para progreso.
- Bloque `try/except` muy genérico al ajustar `auto_arima` (atrapa cualquier excepción y registra `None`); aceptable pero merece logger.warning con tipo de excepción.

## Riesgo de refactor
**ALTO**. Genera datos de AutoARIMA Europa que aparecen en PROJECT_CONTEXT (`MAE 0.376` h=1, `2.510` h=12). Bit-exactitud crítica.

## Acciones FASE 3
1. Logger (con `logger.exception` en el except).
2. Traducir docstrings/comentarios.
3. Mantener parámetros EXACTOS de `auto_arima` (max_p, max_q, max_P, max_Q, criterion, stepwise, etc.) — leer todo el archivo en FASE 3 antes de tocar.
4. NO cambiar esquemas de outputs.
