# Audit — 04_models_deep/05_metrics_deep.py

## Propósito
Consolida métricas rolling baseline + deep (España): tabla comparativa por horizonte, ranking global, plot `all_models_mae_by_horizon.png`. Combina `03_models_baseline/results/rolling_metrics.json` + `04_models_deep/results/deep_rolling_metrics.json`.

## Inputs / Outputs
- **Reads**:
  - `03_models_baseline/results/rolling_metrics.json`
  - `04_models_deep/results/deep_rolling_metrics.json`
- **Writes**:
  - `04_models_deep/results/deep_report.txt`
  - `04_models_deep/results/deep_summary.json`
  - `04_models_deep/results/plots/all_models_mae_by_horizon.png`

## Dependencias internas
- `sys.path.insert` sin importar de `shared/` realmente — eliminar.

## Métricas
- LOC: 200 · Funcionales ~155 · Comentarios/blank ~45 · `print(`: ~13.

## Code smells
- `sys.path.insert` muerto.
- Docstring/comentarios en español; strings de plot/título en español.
- `MODEL_COLORS` hardcoded.

## Riesgo de refactor
**MEDIO**. Output principalmente humano, pero el plot puede aparecer en la memoria del TFG.

## Acciones FASE 3
1. Eliminar `sys.path.insert`.
2. Logger.
3. Traducir docstring/strings; mantener title/legend del plot consistente con la memoria si ya está aprobado.
