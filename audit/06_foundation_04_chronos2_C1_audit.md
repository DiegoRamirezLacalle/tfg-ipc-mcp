# Audit — 06_models_foundation/04_chronos2_C1.py

## Propósito
Chronos-2 sobre IPC España condición C1_mcp con covariables MCP nativas (use_reg_token). Rolling-origin con `past_covariates` y `future_covariates` (para el horizonte futuro las MCP se neutralizan a 0 o se forward-filean).

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_predictions.parquet`
  - `08_results/chronos2_C1_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 371 · Funcionales ~285 · Comentarios/blank ~86 · `print(`: ~18.

## Code smells
- Docstring/comentarios en español.
- 371 líneas — uno de los más grandes; lógica de neutralización futura de MCP merece extraerse pero el plan dice no introducir abstracciones.
- Magic numbers de `Q_IDX`.
- Probable presencia de `MCP_NEUTRAL_COLS` (verificable).

## Riesgo de refactor
**ALTO**. Bit-exactitud requerida. PROJECT_CONTEXT documenta que C1_mcp degrada todos los modelos; ese resultado debe preservarse exacto.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. NO tocar la neutralización futura de MCP ni `Q_IDX`.
4. NO cambiar esquemas.
