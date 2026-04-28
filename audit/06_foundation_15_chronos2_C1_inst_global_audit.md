# Audit — 06_models_foundation/15_chronos2_C1_inst_global.py

## Propósito
Chronos-2 sobre CPI Global condición C1_inst con 3 covariables top: `imf_comm_ma3` (corr 0.586), `brent_log_ma3` (0.456), `gscpi_ma3` (0.324). ★★ del TFG: el único modelo MASE<1.0 a h=12 (Global). Estilo compacto.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_global_institutional.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_inst_global_predictions.parquet`
  - `08_results/chronos2_C1_inst_global_metrics.json` (CONTRATO).

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 163 · Funcionales ~135 · Comentarios/blank ~28 · `print(`: ~7.

## Code smells
- Imports comprimidos.
- `SUBPERIODS` dict.

## Riesgo de refactor
**MUY ALTO**. Modelo ★★ del TFG. Bit-exactitud absolutamente crítica.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. NO tocar nada de la lógica numérica (`EXOG_COLS`, `Q_IDX`, `prepare_input`, `run_rolling`, `mase_scale`).
5. Validación post-refactor: comparar JSON contra baseline_pre_refactor antes de continuar.
