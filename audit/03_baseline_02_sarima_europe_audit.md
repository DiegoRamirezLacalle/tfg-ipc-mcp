# Audit — 03_models_baseline/02_sarima_europe.py

## Propósito
SARIMA(0,1,1)(0,1,1)12 fijo (airline model) sobre HICP Eurozona usando `statsmodels.SARIMAX`. Sirve como punto de comparación con el SARIMA auto-seleccionado del script 01.

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet`.
- **Writes**:
  - `08_results/sarima_europe_summary.txt`
  - `08_results/sarima_europe_metrics.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_VAL_END), `shared.metrics`.

## Métricas
- LOC: 84 · Funcionales ~60 · Comentarios/blank ~24 · `print(`: 9.

## Code smells
- Constantes `ORDER`, `SEASONAL_ORDER` a nivel módulo (correcto).
- Docstring corto pero en español.
- `import numpy as np` posiblemente no usado.
- `warnings.filterwarnings("ignore")` global.
- El más limpio del módulo, casi minimal.

## Riesgo de refactor
**BAJO**. Output autocontenido; nadie más lo lee.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Verificar/eliminar imports no usados.
