# Audit — 06_models_foundation/22_chronos2_C1_mcp_europe.py

## Propósito
Chronos-2 HICP Eurozona condición C1_mcp (señales MCP: bce_shock_score, bce_tone_numeric, bce_cumstance, gdelt_tone_ma6, signal_available). Estilo compacto.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_europe.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_mcp_europe_predictions.parquet`
  - `08_results/chronos2_C1_mcp_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 145 · Funcionales ~120 · Comentarios/blank ~25 · `print(`: ~6.

## Code smells
- Imports comprimidos.
- Lógica de neutralización futura: MCP cols → 0, signal_available → 1.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar `MCP_NEUTRAL_COLS` ni neutralización.
