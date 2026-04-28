# Audit — 03_models_baseline/03_sarimax_europe.py

## Propósito
SARIMAX para HICP Eurozona usando DFR (Deposit Facility Rate del BCE) como exógena. Reusa los órdenes seleccionados por `01_arima_auto_europe.py`. Evaluación estática.

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet`, `data/processed/ecb_rates_monthly.parquet`.
- **Writes**:
  - `08_results/arimax_europe_summary.txt`
  - `08_results/arimax_europe_metrics.json`

## Dependencias internas
- `shared.constants`, `shared.metrics`.

## Métricas
- LOC: 111 · Funcionales ~85 · Comentarios/blank ~26 · `print(`: 12.

## Code smells
- Docstrings/comentarios en español.
- `import numpy as np` posiblemente no usado.
- `warnings.filterwarnings("ignore")` global.
- Patrón limpio y compacto.

## Riesgo de refactor
**BAJO**. Pequeño y autocontenido; nadie lee el JSON downstream.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Eliminar imports no usados.
