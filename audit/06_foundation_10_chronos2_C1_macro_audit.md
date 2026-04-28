# Audit — 06_models_foundation/10_chronos2_C1_macro.py

## Propósito
Chronos-2 IPC España condición C1_macro: mix de señales macro (Brent + TTF + EPU). Estilo compacto.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_macro_predictions.parquet`
  - `08_results/chronos2_C1_macro_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 119 · Funcionales ~100 · Comentarios/blank ~19 · `print(`: ~5.

## Code smells
- Imports comprimidos.
- Mismas convenciones que `09_chronos2_C1_inst.py`.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. Mantener `EXOG_COLS`, `Q_IDX`.
