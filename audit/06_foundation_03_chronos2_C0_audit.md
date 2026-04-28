# Audit — 06_models_foundation/03_chronos2_C0.py

## Propósito
Chronos-2 (`amazon/chronos-2`, 21 cuantiles) sobre IPC España condición C0. Rolling-origin 48 orígenes. Devuelve cuantiles p10/p50/p90 y usa p50 como predicción puntual. Soporta `use_reg_token=True` para covariables (no usadas en C0).

## Inputs / Outputs
- **Reads**: `data/processed/features_exog.parquet` (`indice_general`).
- **Writes**:
  - `08_results/chronos2_C0_predictions.parquet`
  - `08_results/chronos2_C0_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 203 · Funcionales ~155 · Comentarios/blank ~48 · `print(`: ~12.

## Code smells
- Docstring/comentarios en español.
- `Q_IDX` con magic numbers ({"p10": 2, "p50": 10, "p90": 18}) bien comentados.
- `device_map="cpu"` hardcoded — aceptable.
- Import perezoso de `chronos`.

## Riesgo de refactor
**ALTO**. Salida en `metrics_summary_final.json`.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener `CHRONOS_MODEL_ID`, `Q_IDX`, `device_map`.
