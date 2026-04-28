# Audit — 03_models_baseline/04_backtesting_rolling.py

## Propósito
Rolling-origin expanding-window de los 4 baselines de España (`naive`, `arima`, `sarima`, `sarimax`) sobre 48 orígenes (2021-01→2024-12) con horizontes [1,3,6,12]. Genera predicciones tidy y métricas por modelo/horizonte.

## Inputs / Outputs
- **Reads**: `data/processed/features_exog.parquet`.
- **Writes**:
  - `03_models_baseline/results/rolling_predictions.parquet`
  - `03_models_baseline/results/rolling_metrics.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_TEST_END), `shared.metrics`.

## Métricas
- LOC: 270 · Funcionales ~210 · Comentarios/blank ~60 · `print(`: 27.

## Code smells
- Docstring/comentarios en español.
- Órdenes hardcoded (`ARIMA_ORDER = (1,1,2)`, `SARIMA_ORDER = (0,1,1)`, `SARIMA_SORDER = (0,1,1,12)`) — vienen del análisis del 01/02 pero no se cargan dinámicamente. Aceptable si está documentado por qué el orden es ése.
- 27 prints.
- `tqdm` para progreso (correcto, no es print de debug).
- MASE scale fija a partir del train inicial (cumple PROJECT_CONTEXT).

## Riesgo de refactor
**ALTO**. Es la pieza canónica de baselines España; sus resultados son los que aparecen en `metrics_summary_final.json` y se comparan en notebooks 02, 03_regime, 04_ablation. Bit-exactitud crítica en `forecast_naive`, `fit_*`, y en la asignación de `mase_scale`.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. NO cambiar órdenes ni MASE scale.
4. Mantener esquema EXACTO del parquet de salida (`origin`, `fc_date`, `model`, `horizon`, `step`, `y_pred`, `y_true`, `error`, `abs_error`, etc. — verificar lectores en `05_metrics_baseline.py`).
