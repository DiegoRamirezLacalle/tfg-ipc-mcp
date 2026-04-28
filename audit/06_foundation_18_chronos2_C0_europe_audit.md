# Audit — 06_models_foundation/18_chronos2_C0_europe.py

## Propósito
Chronos-2 sobre HICP Eurozona condición C0 (sin exógenas). Rolling-origin 48 orígenes con 21 cuantiles (p10/p50/p90).

## Inputs / Outputs
- **Reads**: `data/processed/hicp_europe_index.parquet` (con `df.set_index('date')` necesario).
- **Writes**:
  - `08_results/chronos2_C0_europe_predictions.parquet`
  - `08_results/chronos2_C0_europe_metrics.json`

## Dependencias internas
- `shared.constants`.

## Métricas
- LOC: 219 · Funcionales ~170 · Comentarios/blank ~49 · `print(`: ~12.

## Code smells
- Docstring/comentarios en español.
- Patrón paralelo a `03_chronos2_C0.py` adaptado a HICP.
- `Q_IDX` magic numbers.

## Riesgo de refactor
**ALTO**. Output `chronos2_C0_europe_predictions.parquet` es leído por `13_build_features_c1_europe.py` (correlaciones residuos). Bit-exactitud crítica.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener `Q_IDX`, `CHRONOS_MODEL_ID`, manejo del índice del HICP.
4. NO cambiar esquemas (especialmente el parquet de predicciones).
