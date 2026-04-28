# Audit — 03_models_baseline/01_arima_auto_europe.py

## Propósito
SARIMA estacional (m=12) vía `auto_arima` para HICP Eurozona: ajusta sobre train inicial, diagnostica con Ljung-Box, evalúa estáticamente y guarda summary + JSON de métricas y órdenes (consumido luego por `04_backtesting_rolling_europe.py`).

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet`.
- **Writes**:
  - `08_results/arima_europe_summary.txt`
  - `08_results/arima_europe_metrics.json` (CONTRATO: orden y seasonal_order leídos por scripts 04 y 07).

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_VAL_END), `shared.metrics` (mae, rmse, mase).

## Métricas
- LOC: 127 · Funcionales ~95 · Comentarios/blank ~32 · `print(`: 13.

## Code smells
- Docstrings/comentarios en español.
- `import numpy as np` aparentemente no usado dentro del archivo (verificar).
- `warnings.filterwarnings("ignore")` global.
- Devuelve `(model.order, model.seasonal_order)` desde `main()` — usado solo para introspección.

## Riesgo de refactor
**MEDIO**. El JSON de salida define los órdenes que consume el rolling de Europa. Si se refactoriza el JSON (claves o tipos), hay que mover en sincronía con `04_backtesting_rolling_europe.py:48-58`.

## Acciones FASE 3
1. Verificar `numpy` y eliminarlo si no se usa.
2. Logger.
3. Traducir docstrings/comentarios.
4. NO cambiar el esquema de `arima_europe_metrics.json` (claves `order`, `seasonal_order`).
