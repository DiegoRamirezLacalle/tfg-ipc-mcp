# Audit — 03_models_baseline/03_sarimax_global.py

## Propósito
ARIMAX para `cpi_global_rate` con FEDFUNDS como exógena. Descarga FEDFUNDS de FRED si no existe, lo cachea en `data/processed/fedfunds_monthly.parquet`, ajusta `auto_arima(exog=fedfunds)`, evalúa estáticamente y guarda summary + JSON.

## Inputs / Outputs
- **Reads**: 
  - `data/processed/cpi_global_monthly.parquet`
  - `data/processed/fedfunds_monthly.parquet` (cache, generada por este mismo script si falta)
- **Writes**:
  - `data/raw/fedfunds_raw.csv`
  - `data/processed/fedfunds_monthly.parquet`
  - `08_results/arimax_global_summary.txt`
  - `08_results/arimax_global_metrics.json`

## Dependencias internas
- `shared.constants`, `shared.metrics`.

## Métricas
- LOC: 339 · Funcionales ~265 · Comentarios/blank ~74 · `print(`: 51.

## Code smells
- 51 prints — el script más verboso del módulo.
- Mezcla ETL (descarga FEDFUNDS) + modelado en el mismo script. Idealmente la descarga viviría en `01_etl/` (como FRED de `10_ingest_institutional_signals_global.py`), pero está aquí por razones históricas. NO mover (regla del refactor).
- Docstrings/comentarios en español.
- `import numpy as np` puede no usarse.
- `warnings` no aparece importado pero hay `warnings.filterwarnings`? — verificar al refactorizar.
- `ARIMA(3,1,0)` orden hardcoded (esperable, viene del script 01).

## Riesgo de refactor
**ALTO**. Genera `fedfunds_monthly.parquet` que consume `04_backtesting_rolling_global.py:72`. La paridad de la serie cacheada y la exógena pasada al modelo son críticas.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. NO mover descarga de FEDFUNDS a `01_etl/` (regla: no cambiar nombres ni esquemas).
4. NO cambiar el formato del parquet `fedfunds_monthly.parquet` (lo lee otro script).
