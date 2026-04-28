# Audit — 06_models_foundation/27_timegpt_C1_inst_europe.py

## Propósito
TimeGPT HICP Eurozona condición C1_inst. Carga `.env`. Patrón shift+1 en exógenas.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_europe.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C1_inst_europe_predictions.parquet`
  - `08_results/timegpt_C1_inst_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 174 · Funcionales ~140 · Comentarios/blank ~34 · `print(`: ~10.

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
