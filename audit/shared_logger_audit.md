# Audit — shared/logger.py

## Propósito
Factory `get_logger(name, level, log_file)` que devuelve un `logging.Logger` configurado con formato estándar, handler de stdout y opcional handler de fichero.

## Inputs / Outputs
- **Reads**: nada.
- **Writes**: si `log_file` se pasa, crea directorio padre y escribe log en UTF-8.

## Dependencias internas
- Ninguna.

## Métricas
- LOC totales: 33
- Funcionales: ~22
- Comentarios/docstrings/blank: ~11
- Sin debug.

## Code smells
- Comentario `# ya configurado` y docstring de módulo en español; comentarios `# consola` / `# fichero opcional` también en español.
- No tiene docstring de función (solo el del módulo).
- Idempotente y limpio en lo demás.

## Riesgo de refactor
**BAJO**. Es la pieza que el plan exige usar para sustituir `print` en TODA la base. Es la base del refactor de logging.

## Acciones propuestas (FASE 3)
1. Traducir comentarios a inglés.
2. Añadir docstring a `get_logger` (un párrafo breve).
3. Mantener firma y nombres exactos: muchos sitios la van a importar pronto.
