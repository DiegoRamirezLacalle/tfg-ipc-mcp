# Audit — 06_models_foundation/09_chronos2_C1_inst.py

## Propósito
Chronos-2 sobre IPC España condición C1_inst con 3 covariables EPU Europe (`epu_europe_ma3`, `epu_europe_log`, `epu_europe_lag1` — corr 0.737/0.701/0.682). Estilo "compacto" (~114 LOC, imports comprimidos en una línea) — variante posterior, eficiente.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`.
- **Writes**:
  - `08_results/chronos2_C1_inst_predictions.parquet`
  - `08_results/chronos2_C1_inst_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 114 · Funcionales ~95 · Comentarios/blank ~19 · `print(`: ~5.

## Code smells
- Imports comprimidos (`import json, sys, warnings`) — anti-PEP8 pero el estilo elegido por Diego para los compactos.
- Funciones one-liner densas con `;` separator — funciona pero baja legibilidad.
- Docstring corto en español.
- `SUBPERIODS` dict sin uso aparente (hay que verificar).

## Riesgo de refactor
**ALTO**. Genera `chronos2_C1_inst_metrics.json` referenciado en PROJECT_CONTEXT.

## Acciones FASE 3
1. Separar imports a líneas individuales (PEP8) sin cambiar la lógica.
2. Logger.
3. Traducir docstring.
4. Verificar uso de `SUBPERIODS` y eliminar si no se usa.
5. NO cambiar `EXOG_COLS`, `Q_IDX`, neutralización futura.
