# Audit — 03_models_baseline/01_arima_auto_global.py

## Propósito
ARIMA no-estacional (D=0) vía `auto_arima` para `cpi_global_rate` (tasa YoY del CPI Global). Ajuste sobre train inicial, diagnóstico de residuos, evaluación estática y guardado del modelo seleccionado.

## Inputs / Outputs
- **Reads**: `data/processed/cpi_global_monthly.parquet`.
- **Writes**:
  - `08_results/arima_global_summary.txt`
  - `08_results/arima_global_metrics.json` (consumido por `02_sarima_global.py` y `04_backtesting_rolling_global.py`).

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_VAL_END), `shared.metrics`.

## Métricas
- LOC: 204 · Funcionales ~165 · Comentarios/blank ~39 · `print(`: 34.

## Code smells
- Docstrings/comentarios en español.
- 34 prints (ruido alto en consola).
- `import numpy as np` que probablemente no se usa (verificar).
- Estructura claramente derivada de `01_arima_auto.py` (Spain) — patrón ARIMA-only para series ya pre-diferenciadas (rate vs. index).

## Riesgo de refactor
**MEDIO**. Define el orden ARIMA(3,1,0) que reusa `04_backtesting_rolling_global.py`. Mantener formato JSON.

## Acciones FASE 3
1. Verificar `numpy` y eliminarlo si no se usa.
2. Logger.
3. Traducir docstrings/comentarios.
4. NO cambiar el esquema de `arima_global_metrics.json`.
