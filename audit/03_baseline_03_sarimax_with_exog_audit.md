# Audit — 03_models_baseline/03_sarimax_with_exog.py

## Propósito
SARIMAX para IPC España con DFR como exógena, usando `auto_arima` con `exogenous=X_train`. Variante "Spain" del script 03 con la exógena del BCE.

## Inputs / Outputs
- **Reads**: `data/processed/features_exog.parquet` (output de `01_etl/06_feature_engineering_exog.py`).
- **Writes**:
  - `03_models_baseline/results/sarimax_summary.txt`
  - `03_models_baseline/results/sarimax_metrics.json`

## Dependencias internas
- `shared.constants`, `shared.metrics`.

## Métricas
- LOC: 217 · Funcionales ~170 · Comentarios/blank ~47 · `print(`: 36.

## Code smells
- Docstring/comentarios en español.
- 36 prints.
- `import numpy as np` posiblemente no usado.
- Depende de `features_exog.parquet`, candidato a quedar huérfano si nadie más lo consume — verificar.

## Riesgo de refactor
**BAJO-MEDIO**. Outputs autocontenidos en `results/` local.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. Eliminar imports no usados.
