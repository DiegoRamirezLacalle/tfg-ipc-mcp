# Audit — 05_mcp_pipeline/news_to_features.py

## Propósito
Orquestador del pipeline MCP completo con 3 modos:
- `--acquire`: descarga GDELT + RSS y almacena en MongoDB
- `--process`: extrae señales LLM de comunicados RSS no procesados (vía `agent_extractor.py`)
- `--build-c1`: agrega a frecuencia mensual y exporta `news_signals.parquet`

Principio clave: separación total entre adquisición (Internet) y ejecución; control de leakage por `ingestion_timestamp < t` para cada origen.

## Inputs / Outputs
- **Reads**: MongoDB `news_raw`.
- **Writes**:
  - `data/processed/news_signals.parquet` (CONTRATO: leído por `13_build_features_c1_europe.py` y modelos C1 España).
  - `data/processed/features_exog.parquet` y `features_c1.parquet` (potencialmente).

## Dependencias internas
- `from constants import FREQ` (imports `shared/` mediante `sys.path.insert`).

## Métricas
- LOC: 419 · Funcionales ~320 · Comentarios/blank ~99.

## Code smells
- Docstring/comentarios en español.
- `sys.path.insert(0, str(PROJECT_ROOT.parent / "shared"))` — patrón distinto al del resto del repo (otros usan `MONOREPO`). Frágil si cambia la estructura.
- `from constants import FREQ` (vez de `from shared.constants import FREQ`) — consecuencia de sys.path manual.
- 3 modos en el mismo script; main de tipo CLI con argparse.
- MongoDB hardcoded.

## Riesgo de refactor
**ALTO**. Genera `news_signals.parquet`, contrato base de las señales MCP que consumen los modelos C1 de España y Europa. Bit-exactitud crítica.

## Acciones FASE 3
1. Normalizar el patrón de `sys.path` para que sea coherente con el resto: `sys.path.insert(0, str(PROJECT_ROOT.parent))` y luego `from shared.constants import FREQ`.
2. Logger.
3. Traducir docstring/comentarios.
4. Mantener esquema EXACTO de `news_signals.parquet` (date + columnas: gdelt_avg_tone, gdelt_goldstein, gdelt_n_articles, bce_shock_score, bce_uncertainty, bce_tone, ine_surprise_score, ine_topic, dominant_topic).
