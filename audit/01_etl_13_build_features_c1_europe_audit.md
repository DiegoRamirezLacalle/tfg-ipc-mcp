# Audit — 01_etl/13_build_features_c1_europe.py

## Propósito
Ensambla `features_c1_europe.parquet` (15 columnas, 276 filas) combinando: HICP target + ECB rates (DFR, MRR, dfr_ma3) + energy (brent_ma3, ttf_ma3) + institutional (epu_europe_ma3) + europe_signals (ESI, breakeven_5y_lag1, eurusd_ma3) + news/MCP (bce_*, gdelt_tone_ma6, signal_available). Reporta correlaciones con `hicp(t+1)` y, si existe, correlaciones de señales MCP contra los residuos del modelo `chronos2_C0_europe`.

## Inputs / Outputs
- **Reads**:
  - `data/processed/hicp_europe_index.parquet`
  - `data/processed/ecb_rates_monthly.parquet`
  - `data/processed/energy_prices_monthly.parquet`
  - `data/processed/institutional_signals_monthly.parquet`
  - `data/processed/europe_signals_monthly.parquet`
  - `data/processed/news_signals.parquet`
  - `08_results/chronos2_C0_europe_predictions.parquet` (opcional, si existe)
- **Writes**: `data/processed/features_c1_europe.parquet`.

## Dependencias internas
- `import sys` no usado.

## Métricas
- LOC totales: 183
- Funcionales: ~145
- Comentarios/docstrings/blank: ~38
- Debug prints: ~30 (la mayoría reportes).

## Code smells
- `import sys` muerto.
- Docstring/comentarios en español.
- Aplica `shift(1)` ad-hoc por columna (`dfr_s`, `dfr_ma3`, `mrr.shift(1)`) en lugar de hacerlo de forma uniforme. Las columnas MCP NO llevan shift, con un comentario explicando que el leakage está controlado en origen — riesgo moderado de leakage si esto cambia.
- Múltiples `reindex(idx).ffill().bfill()` en cada bloque (posible duplicación si más de uno hace lo mismo) — aceptable.
- Hardcoded `mcp_cols = [...]` definida pero solo se usan 4 elementos individualmente; la variable `mcp_cols` no se referencia. Eliminar.
- Lectura condicional de `chronos2_C0_europe_predictions.parquet` mezcla preparación de features con análisis de residuos — apropiado mantenerlo (es informativo para el TFG) pero merece pasar a logger.

## Riesgo de refactor
**ALTO**. `features_c1_europe.parquet` es input crítico de TODOS los modelos C1 Europa, incluido `timesfm_C1_full_europe` (★★ del TFG). Bit-exactitud requerida en orden de columnas y patrón de shift por columna.

## Acciones propuestas (FASE 3)
1. Eliminar `import sys` y `mcp_cols` no usados.
2. Traducir docstrings/comentarios.
3. Sustituir prints por logger.
4. Mantener EXACTO: orden de columnas, shifts, ffill/bfill por columna.
5. Documentar en código por qué las columnas MCP no llevan shift adicional (leakage controlado en origen).
