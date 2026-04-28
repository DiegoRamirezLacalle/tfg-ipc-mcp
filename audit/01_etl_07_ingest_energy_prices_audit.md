# Audit — 01_etl/07_ingest_energy_prices.py

## Propósito
Descarga Brent (BZ=F), WTI (CL=F), TTF (TTF=F) y Henry Hub (NG=F) de Yahoo Finance, construye series Brent y TTF completas hasta 2002 usando WTI/HH como proxies escalados pre-2007/2017, aplica transformaciones (`_log`, `_ret`, `_ma3`, `_lag1`) y un `shift +1` global anti-leakage. Exporta `energy_prices_monthly.parquet`.

## Inputs / Outputs
- **Reads**: Yahoo Finance API.
- **Writes**: `data/processed/energy_prices_monthly.parquet`.

## Dependencias internas
- Ninguna (yfinance, pandas, numpy).

## Métricas
- LOC totales: 150
- Funcionales: ~110
- Comentarios/docstrings/blank: ~40
- Debug prints: ~14.

## Code smells
- Docstrings/comentarios en español.
- Mezcla aleatoria de `print` y bloques separadores; debe pasar a logger.
- `transform_series` usa `.clip(lower=0.01)` para evitar `log(0)`: razonable pero opaco; merece comentar el porqué.
- `build_proxy`: si `overlap < 12` usa `ratio = 1.0` con un print de aviso — fallback peligroso si el proxy y la serie principal tienen escalas distintas, pero documentado.
- `df = df.shift(1)` global garantiza no-leakage; mantener exacto.
- `target_idx = pd.date_range("2002-01-01", "2025-06-01", freq="MS")` — `2025-06` futuro, posible artefacto.

## Riesgo de refactor
**MEDIO**. `energy_prices_monthly.parquet` lo consumen ETLs downstream (10_global, 13_europe) y se usa en condiciones C1_energy/C1_macro. Bit-exactitud crítica en `transform_series` y en el orden de las columnas finales.

## Acciones propuestas (FASE 3)
1. Traducir docstrings/comentarios.
2. Sustituir prints por logger.
3. Mantener exactos: `transform_series`, `build_proxy`, el shift global, el orden de las columnas.
4. Considerar mover `DATE_END` a `shared/constants.py` o documentar por qué llega a 2025-07.
