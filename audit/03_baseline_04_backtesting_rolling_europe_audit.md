# Audit — 03_models_baseline/04_backtesting_rolling_europe.py

## Propósito
Rolling-origin de los 3 baselines Europa (`naive`, `sarima`, `sarimax`) sobre 48 orígenes con horizontes [1,3,6,12]. Carga el orden seleccionado por `01_arima_auto_europe.py` desde su JSON; si no existe, usa fallback `(2,1,1)(1,1,1,12)`.

## Inputs / Outputs
- **Reads**:
  - `data/processed/hicp_europe_index.parquet`
  - `data/processed/ecb_rates_monthly.parquet`
  - `08_results/arima_europe_metrics.json` (órdenes)
- **Writes**:
  - `08_results/rolling_predictions_europe.parquet`
  - `08_results/rolling_metrics_europe.json` (CONTRATO: lo cargan notebooks de evaluación)

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_TEST_END). NO importa `shared.metrics` (calcula MAE/RMSE/MASE in-place).

## Métricas
- LOC: 236 · Funcionales ~185 · Comentarios/blank ~51 · `print(`: 29.

## Code smells
- Docstring/comentarios en español.
- 29 prints.
- MASE/MAE/RMSE re-implementados en lugar de importar `shared.metrics` — duplicación y posible discrepancia futura.
- `import numpy as np` usado.
- `tqdm` para progreso.
- Cuando el JSON de órdenes no existe, fallback silencioso: peligroso, debería logger.warning.

## Riesgo de refactor
**ALTO**. Producto crítico de Europa. Los notebooks `02/03/04_evaluation_europe.ipynb` cargan `rolling_metrics_europe.json`. Bit-exactitud requerida.

## Acciones FASE 3
1. Logger (con warning explícito en el fallback de órdenes).
2. Traducir docstrings/comentarios.
3. Sustituir cálculo in-place de MAE/RMSE/MASE por `shared.metrics` SOLO si reproduce el mismo número exacto. Si hay duda, no tocar.
4. Mantener esquema EXACTO del parquet y del JSON de salida.
