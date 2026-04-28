# Audit — 06_models_foundation/25_timesfm_C1_mcp_europe.py

## Propósito
TimesFM HICP Eurozona condición C1_mcp (Ridge externo + StandardScaler con señales MCP: BCE press releases + GDELT tone).

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_europe.parquet`.
- **Writes**:
  - `08_results/timesfm_C1_mcp_europe_predictions.parquet`
  - `08_results/timesfm_C1_mcp_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 146 · Funcionales ~120 · Comentarios/blank ~26 · `print(`: ~6.

## Code smells
- Imports comprimidos.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar Ridge + StandardScaler.
