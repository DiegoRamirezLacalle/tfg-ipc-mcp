# Audit — 01_etl/10_ingest_institutional_signals_global.py

## Propósito
Descarga 8 señales institucionales globales (FRED: GEPU, IMF Commodity, DXY, VIX, USG10Y, FEDFUNDS; NY Fed: GSCPI; Caldara-Iacoviello: GPR), reutiliza `brent_log` y `dfr` de parquets ya existentes, genera derivadas `_ma3`, `_lag1`, `_diff` con `shift +1` y exporta `features_c1_global_institutional.parquet`. Imprime correlaciones por sub-período y guarda lista de columnas seleccionadas (>=0.2 |corr|) en `c1_global_inst_selected_cols.json`.

## Inputs / Outputs
- **Reads**: HTTPs (FRED CSV, NY Fed Excel, matteoiacoviello.com Excel), `data/processed/energy_prices_monthly.parquet`, `data/processed/ecb_rates_monthly.parquet`, `data/processed/cpi_global_monthly.parquet`.
- **Writes**:
  - `data/processed/features_c1_global_institutional.parquet`
  - `data/processed/c1_global_inst_selected_cols.json` (PROJECT_CONTEXT lo localiza en `08_results/`; aclarar antes de mover)

## Dependencias internas
- `from shared.constants import DATE_TRAIN_END, DATE_TEST_END` — importadas pero **no usadas** en el código.
- Ninguna otra.

## Métricas
- LOC totales: 324
- Funcionales: ~250
- Comentarios/docstrings/blank: ~74
- Debug prints: ~25.

## Code smells
- `DATE_TRAIN_END`, `DATE_TEST_END` importadas pero no se usan — eliminar import o aplicar.
- `import numpy as np` justificado (`np.nan`).
- `_derive`: aplica `shift(1)` ANTES de calcular `rolling/lag/diff`, así `_lag1` es lag-2 efectivo. **Mismo patrón que `09_ingest_institutional_signals.py`** — conviene unificar pero NO cambiar la fórmula numérica.
- `download_gpr` con fallback de URLs y, si todas fallan, devuelve `pd.Series(0.0, ...)`: silenciar fallos críticos así puede esconder regresiones — aceptable para un dataset opcional, pero merece logger.warning.
- Mezcla print + tablas formateadas manuales.
- `import json` dentro de `main` (perezoso) en lugar de top-level.
- `OUTPUT_PATH` apunta a `data/processed/` pero `c1_global_inst_selected_cols.json` (según PROJECT_CONTEXT) vive en `08_results/`. Aclarar la duplicidad.

## Riesgo de refactor
**ALTO**. Genera el dataset principal del experimento Global C1_inst. Es el input de Chronos-2 C1_inst Global (★★ del TFG, único modelo MASE<1.0 a h=12). Bit-exactitud absolutamente crítica.

## Acciones propuestas (FASE 3)
1. Eliminar import muerto `DATE_TRAIN_END`, `DATE_TEST_END`.
2. Subir `import json` al top-level.
3. Traducir docstrings/comentarios.
4. Logger en lugar de prints.
5. NO tocar `_derive`, `_fred`, `download_gscpi`, `download_gpr`, ni el orden de columnas.
6. Mantener fallback de GPR=0 (es el comportamiento que generó los resultados publicados). Convertir el aviso a `logger.warning`.
