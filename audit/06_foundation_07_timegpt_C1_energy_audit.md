# Audit — 06_models_foundation/07_timegpt_C1_energy.py

## Propósito
TimeGPT C1_energy: IPC + señales MCP + energía (Brent, TTF). Patrón de exógenas idéntico a `06_timegpt_C1.py` con set extendido.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C1_energy_predictions.parquet`
  - `08_results/timegpt_C1_energy_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 329 · Funcionales ~250 · Comentarios/blank ~79 · `print(`: ~22.

## Code smells
- Mismas que `06_timegpt_C1.py`.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. NO tocar lógica de exógenas ni `EXOG_COLS`.
