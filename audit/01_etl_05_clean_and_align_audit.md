# Audit — 01_etl/05_clean_and_align.py

## Propósito
Lee el Excel del INE (`data/raw/IPC.xlsx`), extrae el índice general y los 13 grupos ECOICOP, valida continuidad mensual y exporta `ipc_spain_index.parquet` + snapshot versionado.

## Inputs / Outputs
- **Reads**: `data/raw/IPC.xlsx`.
- **Writes**:
  - `data/processed/ipc_spain_index.parquet`
  - `data/snapshots/ipc_spain_index_<tag>.parquet`

## Dependencias internas
- Ninguna (no usa `shared/`).

## Métricas
- LOC totales: 154
- Funcionales: ~110
- Comentarios/docstrings/blank: ~44
- Debug prints: ~12. Sin código muerto.

## Code smells
- `import sys` no usado.
- Docstrings y comentarios en español.
- Constantes mágicas (`ROW_DATES = 7`, `COL_INDEX_END = 291`) bien comentadas pero específicas del formato del Excel del INE; aceptable.
- `print` directos para info y warnings; debe pasar a logger.
- `ECOICOP_NAMES` con claves en español (correcto si los notebooks downstream esperan esos nombres — no tocar).
- `validate_monthly_continuity` devuelve `list[str]`; main lo imprime: separación limpia.

## Riesgo de refactor
**MEDIO**. Es la fuente de `ipc_spain_index.parquet`, consumido por casi todos los modelos de España. Cualquier cambio en parsing o en orden de columnas rompe scripts. Bit-exactitud crítica en `load_and_clean`.

## Acciones propuestas (FASE 3)
1. Eliminar `import sys`.
2. Traducir docstrings/comentarios a inglés.
3. Sustituir `print` por logger.
4. Mantener nombres en español de columnas ECOICOP (downstream depende).
5. No tocar constantes ni la lógica de `parse_ine_date` y `load_and_clean`.
