# Audit — 03_models_baseline/06_sarimax_global_institutional.py

## Propósito
ARIMAX(3,1,0) sobre `cpi_global_rate` con 6 señales institucionales globales (`imf_comm_ma3`, `brent_log_ma3`, `dfr_ma3`, `gscpi_ma3`, `fedfunds_ma3`, `usg10y_ma3`). Hace evaluación estática + rolling expanding-window (48 orígenes), MASE scale fija sobre el train inicial. Output es la condición C1_inst para el baseline Global.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_global_institutional.parquet`.
- **Writes**:
  - `08_results/arimax_C1_inst_global_metrics.json`
  - `08_results/rolling_predictions_C1_inst_global.parquet`
  - `08_results/rolling_metrics_C1_inst_global.json` (CONTRATO; lo cargan los notebooks de evaluación globales).

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_VAL_END, DATE_TEST_END). NO importa `shared.metrics`.

## Métricas
- LOC: 248 · Funcionales ~195 · Comentarios/blank ~53 · `print(`: 27.

## Code smells
- Docstring/comentarios en español, con tildes (algunas Unicode visibles).
- Re-implementación de MAE/RMSE/MASE in-place (patrón repetido en el módulo).
- 27 prints.
- `tqdm` para progreso.
- `EXOG_COLS` hardcoded (consistente con `c1_global_inst_selected_cols.json`).
- `X.ffill().bfill()` aplica relleno bidireccional sobre las exógenas — mantener exacto.

## Riesgo de refactor
**ALTO**. Genera el JSON `rolling_metrics_C1_inst_global.json` referenciado en PROJECT_CONTEXT.md y consumido por notebooks. Bit-exactitud crítica.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. Mantener `EXOG_COLS` y orden ARIMA(3,1,0).
4. NO cambiar esquemas de outputs.
