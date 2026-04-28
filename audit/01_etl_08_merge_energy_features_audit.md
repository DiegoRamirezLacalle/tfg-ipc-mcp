# Audit — 01_etl/08_merge_energy_features.py

## Propósito
Mergea `energy_prices_monthly.parquet` (8 columnas) en `features_c1.parquet`, hace forward-fill / back-fill, y reporta correlaciones con `IPC(t+1)` (rango completo y desde 2015) clasificándolas en fuerte/media/débil con una recomendación textual.

## Inputs / Outputs
- **Reads**: `data/processed/features_c1.parquet`, `data/processed/energy_prices_monthly.parquet`.
- **Writes**: `data/processed/features_c1.parquet` (sobreescribe en sitio).

## Dependencias internas
- Ninguna.

## Métricas
- LOC totales: 147
- Funcionales: ~115
- Comentarios/docstrings/blank: ~32
- Debug prints: ~25 (gran parte son tablas de correlaciones).

## Code smells
- `import numpy as np` no usado (verificar) — solo se usa pandas en este archivo.
- Docstring/comentarios en español.
- Mezcla "merge" + "report" en el mismo `main`: la mitad inferior (correlaciones, ventana 2015+, recomendación) es informativa y podría aislarse en un script `08_report_energy_corr.py` o detrás de un flag, pero también es razonable mantenerlo.
- Sobreescribe el parquet de entrada — peligroso si se ejecuta dos veces (la segunda añade y la condición `if existing` lo mitiga, OK).
- Heurística de strings hardcoded (">>> LANZAR modelos C1 con energia: ...") es ruido para producción; convertir a JSON-summary o eliminar.

## Riesgo de refactor
**ALTO**. Modifica `features_c1.parquet` in-place — si la lógica del merge cambia (orden de columnas, ffill antes/después), todos los modelos C1 cambian de input. Bit-exactitud requerida.

## Acciones propuestas (FASE 3)
1. Confirmar `numpy` no usado y eliminarlo.
2. Traducir docstrings/comentarios.
3. Sustituir prints por logger.
4. Mantener orden de operaciones EXACTO: drop columnas existentes → merge → ffill → bfill → save.
5. Considerar: mover el bloque "RECOMENDACION" a un script aparte o detrás de un flag `--report`.
