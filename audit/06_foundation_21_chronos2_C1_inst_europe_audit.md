# Audit — 06_models_foundation/21_chronos2_C1_inst_europe.py

## Propósito
Chronos-2 HICP Eurozona condición C1_inst (señales institucionales: DFR, Brent, TTF, EPU, ESI, breakeven, EUR/USD). Estilo compacto.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1_europe.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_inst_europe_predictions.parquet`
  - `08_results/chronos2_C1_inst_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 137 · Funcionales ~115 · Comentarios/blank ~22 · `print(`: ~6.

## Code smells
- Imports comprimidos.

## Riesgo de refactor
**ALTO**.

## Acciones FASE 3
1. Separar imports.
2. Logger.
3. Traducir docstring.
4. Mantener `EXOG_COLS`, `Q_IDX`, neutralización futura.
