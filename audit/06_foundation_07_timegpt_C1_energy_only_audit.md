# Audit — 06_models_foundation/07_timegpt_C1_energy_only.py

## Propósito
TimeGPT con SOLO energía como exógenas (sin MCP). Variante minimalista para aislar la contribución del precio energético.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C1_energy_only_predictions.parquet`
  - `08_results/timegpt_C1_energy_only_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 274 · Funcionales ~210 · Comentarios/blank ~64 · `print(`: ~20.

## Code smells
- Docstring/comentarios en español.
- Patrón paralelo a `07_timegpt_C1_energy.py` con `EXOG_COLS` reducidos.
- Conflicto de prefijo: dos archivos `07_*.py`.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener `EXOG_COLS` y nombres de archivos.
