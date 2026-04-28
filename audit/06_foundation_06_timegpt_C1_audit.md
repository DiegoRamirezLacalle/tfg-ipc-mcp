# Audit — 06_models_foundation/06_timegpt_C1.py

## Propósito
TimeGPT con condición C1_mcp: pasa exógenas via `X_df` (futuras) y `df` histórico. Para el horizonte futuro: `signal_available=1.0`, resto forward-fill del último conocido. Mismas covariables que `02_timesfm_C1.py` (Fix2). Soporta `--test-run`/`--full`.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C1_predictions.parquet`
  - `08_results/timegpt_C1_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 331 · Funcionales ~250 · Comentarios/blank ~81 · `print(`: ~22.

## Code smells
- Docstring/comentarios en español.
- Misma carga de `.env` con magic string.
- TimeGPT con exógenas — fragil ante shock energético (PROJECT_CONTEXT documenta MAE +534% sin shift+1).

## Riesgo de refactor
**ALTO**. PROJECT_CONTEXT lo documenta como caso crítico de leakage histórico. Bit-exactitud y patrón shift+1 obligatorio.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. NO tocar el patrón de exógenas (shift+1, neutralización futura).
4. NO cambiar esquemas.
