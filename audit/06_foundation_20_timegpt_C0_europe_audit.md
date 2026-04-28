# Audit — 06_models_foundation/20_timegpt_C0_europe.py

## Propósito
TimeGPT (Nixtla) sobre HICP Eurozona condición C0. Rolling-origin con `--test-run`/`--full`.

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet`, `.env`.
- **Writes**:
  - `08_results/timegpt_C0_europe_predictions.parquet`
  - `08_results/timegpt_C0_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 269 · Funcionales ~205 · Comentarios/blank ~64 · `print(`: ~20.

## Code smells
- Docstring/comentarios en español.
- API key check.
- Patrón paralelo a `05_timegpt_C0.py`.

## Riesgo de refactor
**ALTO**. Resultados publicados, llamadas de pago.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener `SERIES_ID`, formato Nixtla, controles de coste.
