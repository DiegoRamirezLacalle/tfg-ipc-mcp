# Audit — 06_models_foundation/23_chronos2_C1_full_europe.py

## Propósito
Chronos-2 HICP Eurozona C1_full = C1_inst (8 cols: epu_europe_ma3, brent_ma3, esi_eurozone, eurusd_ma3, dfr, dfr_ma3, ttf_ma3, breakeven_5y_lag1) + C1_mcp (5 cols: bce_*, gdelt_tone_ma6, signal_available). Neutralización futura: MCP_NEUTRAL_COLS → 0, signal_available → 1, resto forward-fill.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_europe.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_full_europe_predictions.parquet`
  - `08_results/chronos2_C1_full_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 151 · Funcionales ~125 · Comentarios/blank ~26 · `print(`: ~6.

## Code smells
- Imports comprimidos.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar `C1_INST`, `C1_MCP`, `MCP_NEUTRAL_COLS` ni neutralización.
