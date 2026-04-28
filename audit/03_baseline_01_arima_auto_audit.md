# Audit — 03_models_baseline/01_arima_auto.py

## Propósito
Ajusta un ARIMA(p,1,q) no-estacional vía `pmdarima.auto_arima` sobre el train inicial (2002→2020) del IPC España, diagnostica residuos (Ljung-Box), evalúa sobre validación (2021-01→2022-06) y exporta summary + métricas estáticas.

## Inputs / Outputs
- **Reads**: `data/processed/ipc_spain_index.parquet`.
- **Writes**:
  - `03_models_baseline/results/arima_summary.txt`
  - `03_models_baseline/results/arima_metrics.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_VAL_END, FORECAST_HORIZON — esta última no usada).
- `shared.metrics` (mae, rmse, mase).

## Métricas
- LOC: 170 · Funcionales ~135 · Comentarios/blank ~35 · `print(`: 27.

## Code smells
- Docstrings/comentarios en español (sin tildes).
- 27 prints — pasar a logger.
- `FORECAST_HORIZON` importado pero no usado.
- Mezcla print de tablas y comparación punto a punto en `main()`.

## Riesgo de refactor
**BAJO**. Outputs viven en `results/` local de baseline; otros scripts no parecen leerlos.

## Acciones FASE 3
1. Quitar `FORECAST_HORIZON` no usado.
2. Logger en lugar de prints.
3. Traducir docstrings/comentarios.
4. Mantener firma de `fit_arima`, `forecast_and_evaluate`, `save_results` y nombres de archivos de salida.
