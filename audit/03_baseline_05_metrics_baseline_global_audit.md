# Audit — 03_models_baseline/05_metrics_baseline_global.py

## Propósito
Versión Global del consolidador del módulo: integra métricas estáticas y rolling de los 4 baselines globales, desglosa por sub-período (pre-crisis / shock Fed / normalización), genera ranking y 4 plots.

## Inputs / Outputs
- **Reads**:
  - `08_results/{arima,arima111,arimax}_global_metrics.json`
  - `08_results/rolling_metrics_global.json`
  - `08_results/rolling_predictions_global.parquet`
- **Writes**:
  - `08_results/baseline_global_report.txt`
  - `08_results/baseline_global_summary.json`
  - `08_results/figures/rolling_mae_by_horizon_global.png`
  - `08_results/figures/error_by_period_global.png`
  - `08_results/figures/rolling_errors_h1_h12_global.png`
  - (un cuarto savefig; verificar nombre)

## Dependencias internas
- `import sys` aparente pero sin uso real.

## Métricas
- LOC: 427 (el más largo del módulo) · Funcionales ~330 · Comentarios/blank ~97 · `print(`: 15.

## Code smells
- `import sys` muerto.
- Docstring/comentarios en español; strings de plots en español.
- 4 funciones `plot_*` con estilos hardcoded; colores via `MODEL_COLORS`.
- Mucho código repetido entre las 3 funciones de plots — extraer helper común sería natural pero rompe paridad de figuras.

## Riesgo de refactor
**MEDIO**. Outputs principalmente humanos, pero las figuras pueden estar referenciadas en la memoria del TFG.

## Acciones FASE 3
1. Eliminar `import sys`.
2. Logger.
3. Traducir docstring/strings de plots a inglés.
4. NO refactorizar la estructura de plots si las figuras ya están aprobadas en la memoria (mantener mismos titles/colores/labels o consultar al usuario).
