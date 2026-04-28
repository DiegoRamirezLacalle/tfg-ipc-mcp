# Audit — 06_models_foundation/diagnostico_timegpt_c1.py

## Propósito
Script de diagnóstico/depuración para investigar el comportamiento de TimeGPT C1 (probablemente el incidente histórico de leakage que produjo MAE +534%). Genera análisis y plots auxiliares.

## Inputs / Outputs
- **Reads**: probables `08_results/timegpt_C1_*_predictions.parquet` y `features_c1.parquet`.
- **Writes**: figuras / informes de diagnóstico.

## Dependencias internas
- A confirmar (probablemente `shared.constants`).

## Métricas
- LOC: 546 (el más grande del módulo) · Funcionales ~410 · Comentarios/blank ~136 · `print(`: alto.

## Code smells
- Es script de diagnóstico, no de producción — alta probabilidad de código exploratorio, comentarios mixtos, plots experimentales.
- 546 líneas: posible dead code, código comentado, intentos descartados.

## Riesgo de refactor
**BAJO**. No produce métricas oficiales del TFG; es one-shot de investigación. Candidato a reubicar en `notebooks/` o eliminar tras la auditoría.

## Acciones FASE 3
1. Confirmar con el usuario si el script sigue siendo útil o si fue solo investigación de un incidente ya resuelto.
2. Si sigue: logger, traducir, eliminar código muerto.
3. Si no: archivar / eliminar.
