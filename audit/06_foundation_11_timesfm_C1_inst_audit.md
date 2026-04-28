# Audit — 06_models_foundation/11_timesfm_C1_inst.py

## Propósito
TimesFM 2.5 IPC España con condición C1_inst (3 covariables EPU Europe vía Ridge externo + StandardScaler). Estilo compacto, ~119 LOC. ★ del TFG: el mejor TimesFM España.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`.
- **Writes**:
  - `08_results/timesfm_C1_inst_predictions.parquet`
  - `08_results/timesfm_C1_inst_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 119 · Funcionales ~100 · Comentarios/blank ~19 · `print(`: ~5.

## Code smells
- Imports comprimidos.
- Docstring corto en español.
- Variante compacta del patrón Fix1+Fix2 (Ridge + StandardScaler).

## Riesgo de refactor
**ALTO**. Es un modelo ★ destacado en PROJECT_CONTEXT (`timesfm_C1_inst` mejora ARIMA en h=1).

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar Ridge + StandardScaler ni `XREG_COVS`.
