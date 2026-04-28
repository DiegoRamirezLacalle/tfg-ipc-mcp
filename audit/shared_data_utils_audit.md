# Audit — shared/data_utils.py

## Propósito
Helpers genéricos de datos: parseo de columna `date` a `DatetimeIndex` mensual, resampling, alineación y un split train/val/test por fechas, más un `freeze_snapshot` que serializa parquet versionado.

## Inputs / Outputs
- **Reads**: nada (helpers puros).
- **Writes**: `freeze_snapshot` escribe parquet en una ruta arbitraria.

## Dependencias internas
- Ninguna; solo `pandas`. `freeze_snapshot` hace `import os` interno.

## Métricas
- LOC totales: 44
- Funcionales: ~25
- Comentarios/docstrings/blank: ~19
- Sin debug.

## Code smells
- Docstrings y docstring de módulo en español.
- `import os` perezoso dentro de `freeze_snapshot` en lugar de top-level.
- `train_val_test_split`: `iloc[1:]` para excluir el corte funciona pero es frágil si `train_end` no está en el índice. La metodología real del TFG (rolling-origin) no usa este helper, así que probablemente sea código muerto a verificar con grep.
- `parse_monthly_index` usa `to_period("M").to_timestamp("MS")` — correcto pero no documenta que sobreescribe el índice incluso si ya estaba en MS.

## Riesgo de refactor
**BAJO-MEDIO**. Pocos consumers (a confirmar con grep) y la lógica es simple. Si `train_val_test_split` se usa, hay que respetar bit-exacto el comportamiento de los `iloc[1:]`.

## Acciones propuestas (FASE 3)
1. Traducir docstrings a inglés.
2. Mover `import os` al top.
3. Buscar usos reales de cada helper. Eliminar los no usados (probablemente `train_val_test_split` y `freeze_snapshot`).
4. Mantener firma y comportamiento de las funciones que sigan vivas.
