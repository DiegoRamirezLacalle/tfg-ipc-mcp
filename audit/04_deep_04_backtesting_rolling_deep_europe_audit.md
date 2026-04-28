# Audit — 04_models_deep/04_backtesting_rolling_deep_europe.py

## Propósito
Rolling-origin de los 3 deep models (LSTM, N-BEATS, N-HiTS) sobre HICP Eurozona. Origenes trimestrales (3MS), `max_steps=300`. MASE scale fija sobre el train inicial.

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet`.
- **Writes**:
  - `08_results/deep_rolling_predictions_europe.parquet`
  - `08_results/deep_rolling_metrics_europe.json` (CONTRATO, lo cargan notebooks).

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_TEST_END).

## Métricas
- LOC: 235 · Funcionales ~185 · Comentarios/blank ~50 · `print(`: ~29.

## Code smells
- Docstring/comentarios en español.
- Comentario en NBEATS dice "Estacionalidad significativa (Fs=0.664) -> stacks trend+seasonality" pero el código usa `["identity", "identity", "identity"]` — inconsistencia documentada (probablemente cambio reciente que no se reflejó en el comentario).
- 29 prints.
- MAE/RMSE/MASE re-implementados in-place.

## Riesgo de refactor
**ALTO**. Producto Europa publicado en PROJECT_CONTEXT.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Resolver comentario stale del NBEATS (mantener stacks actuales y actualizar comentario).
4. NO cambiar hiperparámetros, seed, ORIGIN_FREQ ni esquema de salida.
