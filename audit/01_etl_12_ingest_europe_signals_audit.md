# Audit — 01_etl/12_ingest_europe_signals.py

## Propósito
Descarga 3 series de FRED (`ESIEZM` con fallback `BSCICP03EZM665S`, `T5YIE`, `DEXUSEU`), las pasa a mensual (mean / last según el caso), y construye `raw / ma3 / lag1 / diff` con `shift +1` para evitar leakage. Exporta `europe_signals_monthly.parquet` y un CSV raw con las series mensuales.

## Inputs / Outputs
- **Reads**: FRED API vía `pandas_datareader.get_data_fred`.
- **Writes**:
  - `data/raw/europe_signals_raw.csv`
  - `data/processed/europe_signals_monthly.parquet`

## Dependencias internas
- Manipula `sys.path` para añadir el monorepo, sin usar imports posteriores. Eliminar.

## Métricas
- LOC totales: 177
- Funcionales: ~140
- Comentarios/docstrings/blank: ~37
- Debug prints: ~15.

## Code smells
- `sys.path.insert` sin uso real.
- `import pandas_datareader` perezoso DENTRO de `fetch_fred` — repetido dos veces (en try y en fallback). Subir a top-level.
- Docstrings/comentarios en español.
- Si una serie falla, `make_features` se sustituye por `pd.DataFrame(np.nan, ...)` con columnas hardcoded — fragil si se renombra alguna columna en el futuro. Aceptable para este TFG.
- `make_features` nota: aplica `shift(1)` al raw y compone `_lag1 = raw.shift(2)` (lag1 sobre shifted = lag2 del raw). Mismo patrón opaco que `09_ingest_institutional_signals.py` y `10_ingest_institutional_signals_global.py`. NO cambiar la fórmula.

## Riesgo de refactor
**MEDIO**. Output consumido por `13_build_features_c1_europe.py` (clave para todos los modelos Europa C1). Bit-exactitud crítica.

## Acciones propuestas (FASE 3)
1. Subir `import pandas_datareader as pdr` al top-level.
2. Eliminar `sys.path.insert` no usado.
3. Traducir docstrings/comentarios.
4. Sustituir prints por logger.
5. NO cambiar `make_features` ni el orden de columnas.
