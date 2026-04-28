# Audit — 01_etl/11_ingest_hicp_europe.py

## Propósito
Descarga el HICP Eurozone (índice nivel base 2015=100) desde la ECB SDMX, parsea, filtra al rango `2002-01..2024-12`, valida (sin NaN, sin gaps, media 2015 ~100, pico 2022-23) y exporta CSV raw + parquet processed + snapshot + plot de control.

## Inputs / Outputs
- **Reads**: `https://data-api.ecb.europa.eu/.../ICP/M.U2.N.000000.4.INX?format=csvdata`.
- **Writes**:
  - `data/raw/hicp_europe_raw.csv`
  - `data/processed/hicp_europe_index.parquet`
  - `data/processed/hicp_europe_check.png`
  - `data/snapshots/hicp_europe_v1_<YYYYMM>.parquet`

## Dependencias internas
- Ninguna.

## Métricas
- LOC totales: 228
- Funcionales: ~165
- Comentarios/docstrings/blank: ~63
- Debug prints: ~15. No hay código muerto detectable.

## Code smells
- Docstrings/comentarios en español con tildes — sin riesgo de mojibake si UTF-8.
- Funciones bien separadas (`download_raw`, `parse_series`, `filter_range`, `verify`, `plot_series`, `save`, `main`).
- `save()` se llama pero `main()` también escribe `RAW_CSV` antes de filtrar — duplicación: `df_all` se exporta primero como raw, y luego `save(df)` lo re-escribe como raw filtrado. Bug menor: el CSV raw final solo contiene el rango filtrado, no todo el histórico. Confirmar comportamiento esperado.
- `print` en lugar de logger.
- El parquet final escribe `index=False` y con `date` como columna, lo que coincide con la nota crítica de PROJECT_CONTEXT (`hicp_europe_index.parquet` tiene índice entero, hay que `df.set_index('date')` después).

## Riesgo de refactor
**MEDIO**. Producto único (`hicp_europe_index.parquet`) consumido por todo el pipeline Europa. Bit-exactitud crítica en el filtro de rango y el guardado con `index=False`.

## Acciones propuestas (FASE 3)
1. Resolver el doble write de `RAW_CSV`: dejar solo uno (el raw completo o el filtrado, según convención).
2. Traducir docstrings/comentarios.
3. Sustituir prints por logger.
4. Mantener `index=False` y nombres de columnas.
