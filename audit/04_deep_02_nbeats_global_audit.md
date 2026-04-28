# Audit — 04_models_deep/02_nbeats_global.py

## Propósito
N-BEATS para CPI Global. Análogo al de España con stacks adaptados a la falta de estacionalidad de la serie global (D=0). Evaluación estática por horizonte.

## Inputs / Outputs
- **Reads**: vía `_helpers_global.load_nf_format_global`.
- **Writes**: `08_results/nbeats_global_metrics.json`.

## Dependencias internas
- `_helpers_global.{RESULTS_DIR, load_nf_format_global, evaluate_forecast, print_comparison}`.

## Métricas
- LOC: 91 · Funcionales ~68 · Comentarios/blank ~23 · `print(`: ~12.

## Code smells
- Docstrings/comentarios en español.
- Probablemente con stacks `[identity, identity, identity]` (sin estacionalidad para Global).
- Patrón paralelo a `02_nbeats.py`.

## Riesgo de refactor
**MEDIO**. Output referenciado en PROJECT_CONTEXT.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener stacks y hiperparámetros.
