# Audit — 06_models_foundation/13_timegpt_C1_inst.py

## Propósito
TimeGPT IPC España condición C1_inst (3 EPU Europe). Estilo compacto, 123 LOC. Carga `.env`.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C1_inst_predictions.parquet`
  - `08_results/timegpt_C1_inst_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 123 · Funcionales ~100 · Comentarios/blank ~23 · `print(`: ~6.

## Code smells
- Imports comprimidos.
- API key check.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar `EXOG_COLS` ni patrón shift+1.
