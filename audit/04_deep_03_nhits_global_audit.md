# Audit — 04_models_deep/03_nhits_global.py

## Propósito
N-HiTS para CPI Global. Análogo a `03_nhits.py` con `_helpers_global` y target `cpi_global_rate`.

## Inputs / Outputs
- **Reads**: vía `_helpers_global.load_nf_format_global`.
- **Writes**: `08_results/nhits_global_metrics.json`.

## Dependencias internas
- `_helpers_global`.

## Métricas
- LOC: 91 · Funcionales ~68 · Comentarios/blank ~23 · `print(`: ~12.

## Code smells
- Mismas que el N-HiTS de España: docstring ES, prints, posibles imports muertos.

## Riesgo de refactor
**MEDIO**. Output referenciado en PROJECT_CONTEXT.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener hiperparámetros y seed.
