# Audit — 01_etl/09_ingest_institutional_signals.py

## Propósito
Descarga EPU Spain + EPU Europe (policyuncertainty.com, Excel) y ESI Spain (Eurostat SDMX), aplica derivadas (`_log`, `_diff`, `_ma3`, `_lag1`) con `shift +1` global, calcula correlaciones con `IPC(t+1)` (full / 2015+ / 2002+) y exporta `institutional_signals_monthly.parquet`. Implementa un "gate check" que avisa si todas las correlaciones son < 0.3.

## Inputs / Outputs
- **Reads**: HTTPs (`policyuncertainty.com`, `ec.europa.eu/eurostat`), `data/processed/ipc_spain_index.parquet`.
- **Writes**: `data/processed/institutional_signals_monthly.parquet`.

## Dependencias internas
- Manipula `sys.path` para añadir el monorepo, pero no llega a importar de `shared/`. Limpieza recomendable.

## Métricas
- LOC totales: 235
- Funcionales: ~175
- Comentarios/docstrings/blank: ~60
- Debug prints: ~20.

## Code smells
- `sys.path.insert(0, str(MONOREPO))` sin uso real de imports `shared/`. Eliminar.
- `import numpy as np` se usa solo para `np.log`; OK pero verificar.
- `warnings.filterwarnings("ignore")` global.
- `transform`: aplica `shift(1)` después de calcular `_log/_diff/_ma3/_lag1`. La columna `_lag1` queda como `s.shift(1).shift(1) = s.shift(2)` efectivo — comportamiento intencionado pero opaco; merece comentario explícito (NO cambiar, paridad).
- Strings de progreso decoradas con `=`/`-` y mezcla print+formateo manual de tablas.
- "Gate check" textual: aceptable como side effect, pero merece pasar a logger.

## Riesgo de refactor
**MEDIO**. `institutional_signals_monthly.parquet` lo consume `10_merge_institutional_features.py` y `13_build_features_c1_europe.py`. Si la lógica de derivadas o el shift cambian, los modelos C1 cambian.

## Acciones propuestas (FASE 3)
1. Eliminar la manipulación de `sys.path` que no se usa.
2. Traducir docstrings/comentarios.
3. Sustituir prints por logger.
4. Documentar explícitamente en `transform` que `_lag1` es lag-2 efectivo tras el shift global. NO cambiar la fórmula.
5. Mantener formato del parquet (con columna `date`).
