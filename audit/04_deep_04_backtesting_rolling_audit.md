# Audit — 04_models_deep/04_backtesting_rolling.py

## Propósito
Rolling-origin expanding-window de los 3 deep models (LSTM, N-BEATS, N-HiTS) sobre IPC España. Por viabilidad computacional usa `ORIGIN_FREQ="3MS"` (origenes trimestrales) y `max_steps=300`. Mantiene hiperparámetros y seed=42 para reproducibilidad.

## Inputs / Outputs
- **Reads**: `data/processed/ipc_spain_index.parquet`.
- **Writes**:
  - `04_models_deep/results/deep_rolling_predictions.parquet`
  - `04_models_deep/results/deep_rolling_metrics.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_TEST_END). NO usa `shared.metrics`.

## Métricas
- LOC: 230 · Funcionales ~180 · Comentarios/blank ~50 · `print(`: ~27.

## Code smells
- Docstring/comentarios en español.
- `tqdm` para progreso.
- 3 modelos entrenados por origen × 4 horizontes × N orígenes — coste alto pero gestionado por origen trimestral.
- MAE/RMSE/MASE re-implementados in-place en vez de usar `shared.metrics`.
- Stacks N-BEATS adaptados según horizonte.

## Riesgo de refactor
**ALTO**. Producto principal de deep España (`deep_rolling_metrics.json`), referenciado en PROJECT_CONTEXT y notebooks.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener hiperparámetros, seed, ORIGIN_FREQ trimestral, max_steps=300, stacks N-BEATS.
4. NO cambiar esquema del parquet/JSON de salida.
