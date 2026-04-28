# Audit — 06_models_foundation/02_timesfm_C1.py

## Propósito
TimesFM 2.5 sobre IPC España con condición C1_mcp. Implementa el patrón Fix1+Fix2: el base TimesFM ve el contexto IPC completo (igual que C0) y un Ridge externo ajustado SOLO sobre `df.loc['2015':origin]` añade una corrección residual basada en señales MCP. Las covariables del Ridge son: `gdelt_avg_tone`, `gdelt_tone_ma3`, `gdelt_tone_ma6`, `bce_shock_score`, `bce_tone_numeric`, `bce_cumstance`, `ine_surprise_score`, `ine_inflacion`, `signal_available`. Requiere `StandardScaler` antes de Ridge.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`.
- **Writes**:
  - `08_results/timesfm_C1_predictions.parquet`
  - `08_results/timesfm_C1_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 289 · Funcionales ~225 · Comentarios/blank ~64 · `print(`: ~15.

## Code smells
- Docstring extenso en español describiendo Fix1/Fix2 — buen contexto.
- Mezcla de TimesFM + Ridge + StandardScaler — flujo complejo pero documentado.
- `XREG_COVS` hardcoded (lista canónica del TFG).
- `SIGNAL_START = "2015-01-01"` constante crítica.

## Riesgo de refactor
**ALTO**. Lógica compleja con leakage prevention crítica (shift+1 en exógenas, StandardScaler antes de Ridge). PROJECT_CONTEXT memoriza explícitamente que sin StandardScaler la corrección es de +1.11 pp constantes.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios manteniendo la explicación de Fix1/Fix2.
3. NO tocar: `XREG_COVS`, `SIGNAL_START`, lógica del Ridge, normalización.
4. NO cambiar esquemas de salida.
