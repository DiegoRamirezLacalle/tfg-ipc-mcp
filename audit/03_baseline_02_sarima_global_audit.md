# Audit — 03_models_baseline/02_sarima_global.py

## Propósito
A pesar del nombre "sarima", ajusta DOS ARIMAs no-estacionales fijos sobre `cpi_global_rate`: ARIMA(1,1,1) (referencia parsimónica) y ARIMA(3,1,0) (ganador AIC del script 01, recargado del JSON). Genera summary y métricas comparativas.

## Inputs / Outputs
- **Reads**:
  - `data/processed/cpi_global_monthly.parquet`
  - `08_results/arima_global_metrics.json` (recupera el ARIMA(3,1,0) del script 01)
- **Writes**:
  - `08_results/arima111_global_summary.txt`
  - `08_results/arima111_global_metrics.json`

## Dependencias internas
- `shared.constants`, `shared.metrics`.

## Métricas
- LOC: 227 · Funcionales ~180 · Comentarios/blank ~47 · `print(`: 41.

## Code smells
- Nombre "sarima_global" engañoso (no es estacional); explicado en docstring pero confunde.
- 41 prints muy verbosos.
- Docstrings/comentarios en español.
- Doble ajuste manual replica lógica del script 01 — duplicación.
- `import numpy` quizás no usado.

## Riesgo de refactor
**MEDIO**. Output `arima111_global_metrics.json` lo consume `04_backtesting_rolling_global.py` (aunque allí el orden está hardcoded como `ARIMA111_ORDER = (1,1,1)`, así que la dependencia es solo nominal).

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. Verificar imports no usados.
4. Considerar reducir el bloque doble (refit del ARIMA del script 01 podría ser un compute one-shot — pero NO cambiar la lógica numérica).
