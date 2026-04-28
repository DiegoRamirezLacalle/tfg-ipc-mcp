# Audit — 06_models_foundation/08_chronos2_C1_energy_only.py

## Propósito
Chronos-2 con SOLO energía como exógenas. Variante minimalista para aislar la contribución del precio energético sobre IPC España.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_energy_only_predictions.parquet`
  - `08_results/chronos2_C1_energy_only_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 305 · Funcionales ~235 · Comentarios/blank ~70 · `print(`: ~17.

## Code smells
- Docstring/comentarios en español.
- Patrón paralelo a `04_chronos2_C1.py` y `05_chronos2_C1_energy.py` con `EXOG_COLS` reducidos.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener `EXOG_COLS`, `Q_IDX`, neutralización futura.
