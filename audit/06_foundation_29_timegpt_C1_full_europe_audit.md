# Audit — 06_models_foundation/29_timegpt_C1_full_europe.py

## Propósito
TimeGPT HICP Eurozona C1_full (institucionales + MCP). Variante final.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_europe.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C1_full_europe_predictions.parquet`
  - `08_results/timegpt_C1_full_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 175 · Funcionales ~140 · Comentarios/blank ~35 · `print(`: ~10.

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
