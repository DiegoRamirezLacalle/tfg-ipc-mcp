# Audit — 04_models_deep/_helpers_global.py

## Propósito
Helpers para deep models Global: `load_nf_format_global()` carga `cpi_global_rate` en formato long con `unique_id="CPI_GLOBAL"`; `evaluate_forecast()` y `print_comparison()` análogos a `_helpers.py` (Spain).

## Inputs / Outputs
- **Reads**: `data/processed/cpi_global_monthly.parquet`.
- **Writes**: nada.

## Dependencias internas
- `shared.constants`, `shared.metrics`.

## Métricas
- LOC: 66 · Funcionales ~50 · Comentarios/blank ~16 · `print(`: 2.

## Code smells
- Docstrings en español.
- `numpy` importado pero quizás no usado.
- Duplicación con `_helpers.py` (Spain) — los 3 helpers parametrizables podrían fusionarse en uno con `unique_id` y ruta como argumentos. Pero el plan dice no cambiar nombres ni introducir abstracciones — mantener separados.

## Riesgo de refactor
**MEDIO**. Importado por scripts deep de Global.

## Acciones FASE 3
1. Verificar uso de `numpy` y eliminarlo si no se usa.
2. Traducir docstrings.
3. Logger en `print_comparison`.
4. Mantener firmas y `unique_id`.
