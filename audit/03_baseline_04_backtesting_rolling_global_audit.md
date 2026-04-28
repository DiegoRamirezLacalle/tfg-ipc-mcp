# Audit — 03_models_baseline/04_backtesting_rolling_global.py

## Propósito
Rolling-origin de los 4 baselines Global (`naive`, `arima` (3,1,0), `arima111` (1,1,1), `arimax` ARIMA(3,1,0)+FEDFUNDS) sobre 48 orígenes con horizontes [1,3,6,12]. Genera predicciones tidy y métricas globales.

## Inputs / Outputs
- **Reads**:
  - `data/processed/cpi_global_monthly.parquet`
  - `data/processed/fedfunds_monthly.parquet`
- **Writes**:
  - `08_results/rolling_predictions_global.parquet`
  - `08_results/rolling_metrics_global.json` (CONTRATO).

## Dependencias internas
- `shared.constants`. NO importa `shared.metrics`.

## Métricas
- LOC: 303 · Funcionales ~240 · Comentarios/blank ~63 · `print(`: 44.

## Code smells
- Docstring/comentarios en español.
- 44 prints.
- MAE/RMSE/MASE re-implementados in-place (igual que el de Europa) en vez de usar `shared.metrics` — duplicación.
- Órdenes hardcoded (consistente con script 03 / 02_global).
- `tqdm` para progreso.

## Riesgo de refactor
**ALTO**. Producto crítico Global. Sus métricas alimentan `metrics_summary_final.json` y los notebooks de evaluación. Bit-exactitud requerida.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. Mantener órdenes hardcoded y la MASE scale fija.
4. NO cambiar esquema de outputs.
