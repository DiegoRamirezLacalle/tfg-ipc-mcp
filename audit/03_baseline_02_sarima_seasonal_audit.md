# Audit — 03_models_baseline/02_sarima_seasonal.py

## Propósito
SARIMA estacional (m=12) vía `auto_arima` sobre IPC España, con diagnóstico de residuos y comparación contra ARIMA del script 01. Variante explícita "con estacionalidad" del baseline.

## Inputs / Outputs
- **Reads**: `data/processed/ipc_spain_index.parquet`.
- **Writes**:
  - `03_models_baseline/results/sarima_summary.txt`
  - `03_models_baseline/results/sarima_metrics.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_VAL_END, FORECAST_HORIZON), `shared.metrics`.

## Métricas
- LOC: 211 · Funcionales ~170 · Comentarios/blank ~41 · `print(`: 36.

## Code smells
- Docstrings/comentarios en español.
- Patrón muy similar a `01_arima_auto.py` (Spain) — duplicación amplia, pero aceptable porque es un baseline diferente y la salida va a la misma carpeta `results/`.
- 36 prints.
- `FORECAST_HORIZON` importado y posiblemente no usado.
- `import numpy as np` puede no usarse.

## Riesgo de refactor
**BAJO**. Outputs autocontenidos en `results/` local.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. Eliminar imports no usados.
