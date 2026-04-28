# Audit — 04_models_deep/05_metrics_deep_global.py

## Propósito
Consolida métricas rolling baseline + deep para CPI Global. Tabla comparativa, ranking, plot `all_models_mae_by_horizon_global.png`.

## Inputs / Outputs
- **Reads**:
  - `08_results/rolling_metrics_global.json`
  - `08_results/deep_rolling_metrics_global.json`
- **Writes**:
  - `08_results/deep_global_report.txt`
  - `08_results/deep_global_summary.json`
  - `08_results/figures/all_models_mae_by_horizon_global.png`

## Dependencias internas
- Ninguna.

## Métricas
- LOC: 189 · Funcionales ~145 · Comentarios/blank ~44 · `print(`: ~15.

## Code smells
- Docstring/comentarios en español.
- `MODEL_COLORS` hardcoded.
- Patrón paralelo a `05_metrics_deep.py` adaptado a Global (modelos baseline distintos: arima/arima111/arimax).

## Riesgo de refactor
**MEDIO**. Output principalmente humano.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/strings.
3. Mantener consistencia visual del plot si ya está en la memoria.
