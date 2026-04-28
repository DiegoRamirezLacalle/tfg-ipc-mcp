# Audit — 05_mcp_pipeline/fix_signals.py

## Propósito
Repara las señales BCE e INE en MongoDB:
1. INE: parsea JSON de gráficos del body para extraer tasas IPC reales (sin LLM).
2. BCE: inserta decisiones históricas conocidas del Consejo de Gobierno (2015-2024) con señales precisas y datos públicos bien documentados.
3. Limpia documentos basura (páginas de navegación BCE, placeholders).

## Inputs / Outputs
- **Reads**: MongoDB `news_raw`.
- **Writes**: MongoDB `news_raw` (in-place fix), también puede leer/parsear cualquier JSON embebido.

## Dependencias internas
- Ninguna.

## Métricas
- LOC: 449 · Funcionales ~340 · Comentarios/blank ~109.

## Code smells
- Docstring/comentarios en español.
- "Decisiones históricas conocidas" embebidas en el código (tablas hardcoded de eventos BCE 2015-2024) — útil pero quebradizo si la fuente real cambia.
- `argparse` con flags `--ine`, `--bce`, `--clean`, `[default: todo]`.
- Sin dependencias `shared/`.

## Riesgo de refactor
**MEDIO**. Modifica MongoDB; si se ejecuta dos veces puede duplicar fixes (a verificar idempotencia).

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener tablas hardcoded de decisiones BCE EXACTAS (es la fuente de verdad para señales BCE pre-2024).
4. NO cambiar nombres de colecciones ni el contrato del API.
