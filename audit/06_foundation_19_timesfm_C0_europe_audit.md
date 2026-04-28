# Audit — 06_models_foundation/19_timesfm_C0_europe.py

## Propósito
TimesFM 2.5 sobre HICP Eurozona condición C0. Rolling-origin 48 orígenes.

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet`.
- **Writes**:
  - `08_results/timesfm_C0_europe_predictions.parquet`
  - `08_results/timesfm_C0_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 221 · Funcionales ~170 · Comentarios/blank ~51 · `print(`: ~12.

## Code smells
- Docstring/comentarios en español.
- Patrón paralelo a `01_timesfm_C0.py` adaptado a HICP.

## Riesgo de refactor
**ALTO**. PROJECT_CONTEXT lo destaca como `TimesFM C0` con MAE 0.353 h=1, segundo mejor en Europa.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener config TimesFM y manejo del índice.
