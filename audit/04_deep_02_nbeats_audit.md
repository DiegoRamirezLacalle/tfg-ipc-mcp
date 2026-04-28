# Audit — 04_models_deep/02_nbeats.py

## Propósito
N-BEATS variante "generic" (Oreshkin et al. 2020) sobre IPC España. Para `h<4` usa stacks `[identity, identity, identity]` (los stacks trend/seasonality requieren `h ≥ 2*n_harmonics`); para `h≥4` usa `[identity, trend, seasonality]`. `input_size=24, max_steps=500, seed=42`.

## Inputs / Outputs
- **Reads**: vía `_helpers.load_nf_format`.
- **Writes**: `04_models_deep/results/nbeats_metrics.json`.

## Dependencias internas
- `_helpers.{RESULTS_DIR, load_nf_format, evaluate_forecast, print_comparison}`.

## Métricas
- LOC: 105 · Funcionales ~75 · Comentarios/blank ~30 · `print(`: ~12.

## Code smells
- Docstring/comentarios en español.
- `import pandas as pd` posiblemente no usado.
- Lógica de stacks bien comentada (workaround documentado).
- N-BEATS gana España h=1 (PROJECT_CONTEXT) — paridad numérica importa.

## Riesgo de refactor
**MEDIO**. N-BEATS h=1 ★ del TFG. Mantener estructura de stacks y semilla.

## Acciones FASE 3
1. Verificar `pandas` no usado.
2. Logger.
3. Traducir docstring/comentarios.
4. NO cambiar stacks ni hiperparámetros.
