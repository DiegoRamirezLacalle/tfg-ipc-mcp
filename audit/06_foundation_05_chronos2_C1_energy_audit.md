# Audit — 06_models_foundation/05_chronos2_C1_energy.py

## Propósito
Chronos-2 sobre IPC España con condición C1_energy (covariables: brent_log, brent_ret, brent_ma3, ttf_log, ttf_ret, ttf_ma3 — energía MCP combinada con MCP). Mismo patrón que `04_chronos2_C1.py` pero con set de exógenas distinto.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_energy_predictions.parquet`
  - `08_results/chronos2_C1_energy_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 371 · Funcionales ~285 · Comentarios/blank ~86 · `print(`: ~18.

## Code smells
- Docstring/comentarios en español.
- Mismas estructuras que `04_chronos2_C1.py` con `EXOG_COLS` distintas — duplicación esperada.

## Riesgo de refactor
**ALTO**. Resultados publicados.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener `EXOG_COLS`, `Q_IDX`, neutralización futura.
