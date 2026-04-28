# Audit — 06_models_foundation/26_timesfm_C1_full_europe.py

## Propósito
TimesFM HICP Eurozona C1_full (C1_inst + C1_mcp combinados). ★★ del TFG: el mejor modelo global, rompe la barrera MAE<2.0 a h=12 (1.995). Variante final con todas las señales.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_europe.parquet`.
- **Writes**:
  - `08_results/timesfm_C1_full_europe_predictions.parquet`
  - `08_results/timesfm_C1_full_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 149 · Funcionales ~125 · Comentarios/blank ~24 · `print(`: ~6.

## Code smells
- Imports comprimidos.

## Riesgo de refactor
**MUY ALTO**. Modelo ★★ del TFG. Bit-exactitud absoluta.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar Ridge + StandardScaler ni `XREG_COVS` (combinación inst+mcp).
5. Validación post-refactor: comparar JSON contra baseline_pre_refactor antes de continuar.
