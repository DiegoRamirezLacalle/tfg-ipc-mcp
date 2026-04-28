# Audit — 01_etl/01_ingest_cpi_global.py

## Propósito
Descarga el Excel del World Bank Global Inflation, calcula la tasa global de inflación (mediana transversal de pct_change(12) sobre el índice por país) y exporta `cpi_global_monthly.parquet`, snapshot versionado y plot.

## Inputs / Outputs
- **Reads**: `https://thedocs.worldbank.org/.../Inflation-data.xlsx` (HTTP).
- **Writes**:
  - `data/raw/cpi_global_raw.xlsx`
  - `data/processed/cpi_global_monthly.parquet`
  - `data/snapshots/cpi_global_v1_<YYYYMM>.parquet`
  - `data/processed/cpi_global_monthly_check.png`

## Dependencias internas
- Ninguna (no importa de `shared/`).

## Métricas
- LOC totales: 270
- Funcionales: ~210
- Comentarios/docstrings/blank: ~60
- Debug prints: ~20 sentencias `print(...)`. No hay código comentado muerto detectable.

## Code smells
- Toda la documentación en español sin tildes (mojibake-friendly), mezcla de `print` con bloques decorativos `=`/`─` que hacen ruido.
- Reconfiguración de `sys.stdout` en runtime (`sys.stdout.reconfigure(encoding="utf-8")`) para evitar errores en Windows: aceptable como hack pero merece ir a `shared/` si se repite.
- `warnings.filterwarnings("ignore")` global — silencia todo, incluyendo posibles avisos relevantes.
- `download_excel`, `load_hcpi_m`, `compute_global_rate`, `print_stats`, `plot_series`, `main`: cada una una responsabilidad clara, ya está bien factorizado.
- `print_stats` mezcla cálculo y output: aceptable a este tamaño.
- `plot_series`: 50+ líneas con strings en español (`"Crisis financiera"`, `"Inflacion global (YoY, %)"`).
- Logger no se usa — todos los `print` deberían pasar a `shared/logger.py`.

## Riesgo de refactor
**BAJO**. El script es leaf (no es importado por nadie); solo produce outputs. La paridad numérica solo afecta a `cpi_global_monthly.parquet`, que sí consume el resto del pipeline. Bit-exactitud → mantener tal cual `pct_change(periods=12, axis=1) * 100` y la mediana skipna.

## Acciones propuestas (FASE 3)
1. Traducir docstrings/comentarios a inglés. Mantener strings de plot legibles.
2. Sustituir `print` por `logger = get_logger(__name__)` y `logger.info(...)`.
3. Extraer constantes de fechas a top-level (ya lo está) y dejar `main()` lineal.
4. No tocar `compute_global_rate` ni `load_hcpi_m` (paridad).
