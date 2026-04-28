# Audit — 04_models_deep/03_nhits.py

## Propósito
N-HiTS (Challu et al. 2023) sobre IPC España: `input_size=24, max_steps=500, seed=42`. Variante de N-BEATS con multi-rate sampling. Evaluación estática por horizonte.

## Inputs / Outputs
- **Reads**: vía `_helpers.load_nf_format`.
- **Writes**: `04_models_deep/results/nhits_metrics.json`.

## Dependencias internas
- `_helpers.{RESULTS_DIR, load_nf_format, evaluate_forecast, print_comparison}`.

## Métricas
- LOC: 98 · Funcionales ~72 · Comentarios/blank ~26 · `print(`: ~12.

## Code smells
- Docstrings/comentarios en español.
- `import pandas as pd` posiblemente no usado.
- Patrón paralelo a `02_nbeats.py`.

## Riesgo de refactor
**MEDIO**. Mantener hiperparámetros y seed.

## Acciones FASE 3
1. Verificar imports no usados.
2. Logger.
3. Traducir docstring/comentarios.
