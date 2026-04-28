# Audit — 07_evaluation/build_metrics_summary_final.py

## Propósito
Construye `metrics_summary_final.json` (master de España con ~24 modelos): baseline (naive/arima/sarima/sarimax), deep (lstm/nbeats/nhits) y foundation (timesfm/chronos2/timegpt × C0/C1/C1_energy/C1_inst/C1_macro). Lee parquets de predicciones, recalcula MAE/RMSE/MASE con la misma `mase_scale` (sobre 2002-2020) y unifica.

## Inputs / Outputs
- **Reads**:
  - `data/processed/ipc_spain_index.parquet` (para MASE scale)
  - `08_results/*_predictions.parquet` (todos los modelos)
  - `03_models_baseline/results/rolling_predictions.parquet`
  - `04_models_deep/results/deep_rolling_predictions.parquet`
- **Writes**: `08_results/metrics_summary_final.json` (CONTRATO MAYOR; cargado por todos los notebooks de evaluación).

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END).

## Métricas
- LOC: 182 · Funcionales ~145 · Comentarios/blank ~37.

## Code smells
- Docstring en español.
- MAE/RMSE/MASE re-implementados en lugar de `shared.metrics`.
- `compute_mase_scale()` recalcula la escala desde el train; debe coincidir con la usada por los rolling backtests.

## Riesgo de refactor
**MUY ALTO**. Es el JSON master del TFG. Cambiar la fórmula de MASE scale (incluso por un cast) cambia todos los números publicados.

## Acciones FASE 3
1. Logger.
2. Traducir docstring.
3. NO tocar `compute_mase_scale` ni `metrics_from_parquet`.
4. NO cambiar el esquema de `metrics_summary_final.json` (las claves `h1`/`h3`/`h6`/`h12` y sub-claves `MAE`/`RMSE`/`MASE`/`n_evals`).
5. Validación post-refactor: diff JSON vs baseline_pre_refactor línea a línea.
