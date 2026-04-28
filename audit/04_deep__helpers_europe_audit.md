# Audit — 04_models_deep/_helpers_europe.py

## Propósito
Helpers para los deep models de Europa: `load_nf_format_europe()` carga HICP (con `df.set_index('date')` implícito vía rename) y devuelve formato long; `evaluate_forecast()` envuelve `shared.metrics`. UNIQUE_ID = "HICP_EUROPE".

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet`.
- **Writes**: nada.

## Dependencias internas
- `shared.constants`, `shared.metrics`.

## Métricas
- LOC: 44 · Funcionales ~30 · Comentarios/blank ~14 · `print(`: 0.

## Code smells
- `numpy` importado y NO usado.
- Docstring breve en español.
- No expone `print_comparison` (a diferencia de `_helpers.py`); puede ser intencional.

## Riesgo de refactor
**MEDIO**. Lo importan los scripts deep de Europa: 01_lstm_univariate_europe, 02_nbeats_europe, 03_nhits_europe, 04_backtesting_rolling_deep_europe.

## Acciones FASE 3
1. Eliminar `numpy` no usado.
2. Traducir docstring.
3. Mantener firmas y `UNIQUE_ID`.
