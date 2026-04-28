# Audit — 05_mcp_pipeline/agent_extractor.py

## Propósito
Procesa textos RSS de comunicados oficiales (BCE, INE, BdE) con Ollama (modelo `qwen3:4b`) y devuelve señales estructuradas (`RSSSignals`) vía Pydantic + structured output. GDELT NO pasa por aquí (sus señales son cuantitativas).

## Inputs / Outputs
- **Reads**: 
  - `prompts/extraction_v1.txt` (template del prompt).
  - Texto pasado como argumento (típicamente desde MongoDB `news_raw`).
- **Writes**: nada directo (devuelve Pydantic models al caller).

## Dependencias internas
- Ninguna.

## Métricas
- LOC: 178 · Funcionales ~135 · Comentarios/blank ~43.

## Code smells
- Docstring/comentarios en español.
- Enums (`Decision`, `Tone`, `Topic`) con valores en español (`subida`, `bajada`, `hawkish`, etc.) — mantener si la pipeline downstream depende de esos strings.
- Modelo LLM `qwen3:4b` hardcoded — aceptable para reproducibilidad.
- `_THINK_RE` regex compilado — limpieza de tags de chain-of-thought de qwen3.

## Riesgo de refactor
**MEDIO-ALTO**. Si los enums cambian valores, las señales de MongoDB ya extraídas no encajan con el código nuevo. Bit-exactitud lógica (no numérica) requerida.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings (NO los valores de los enums).
3. Mantener `MODEL`, `_THINK_RE`, esquema Pydantic.
4. Confirmar si los valores de enum se usan exactamente en MongoDB / news_to_features.
