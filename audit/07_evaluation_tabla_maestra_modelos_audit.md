# Audit — 07_evaluation/tabla_maestra_modelos.py

## Propósito
Genera la tabla maestra HTML/CSV con todos los modelos del TFG agrupados por país (Global/España/Europa) × tier (C0/C1) × tipo de señal (energy / macro / institutional / mcp / full). Calcula MAE por horizonte y delta% vs el mejor baseline C0 por país. Soporta apertura automática en navegador.

## Inputs / Outputs
- **Reads**: 
  - `08_results/metrics_summary_final.json`
  - `08_results/rolling_metrics_global.json`
  - `08_results/rolling_metrics_C1_inst_global.json`
  - `08_results/deep_rolling_metrics_global.json`
  - `08_results/{modelo}_europe_metrics.json`
  - Varios `08_results/{modelo}_metrics.json`
- **Writes**:
  - `08_results/tabla_maestra.html`
  - `08_results/tabla_maestra.csv`

## Dependencias internas
- Ninguna directa (no usa `shared/`).

## Métricas
- LOC: 459 · Funcionales ~370 · Comentarios/blank ~89.

## Code smells
- Docstring/comentarios en español.
- Lista MASIVA hardcoded de modelos (`MODELS = [...]`) — frágil pero documentada.
- HTML inline (probablemente con CSS embebido) — aceptable para reporte.
- Mezcla de strings con tildes y sin tildes (`España`).

## Riesgo de refactor
**MEDIO-ALTO**. Reportes usados en la memoria del TFG. Si los nombres de archivos JSON cambian, este script rompe.

## Acciones FASE 3
1. Logger en lugar de prints (si los hay).
2. Traducir docstring/comentarios (NO tocar las strings de país que sean labels de tabla, podrían venir referenciados en LaTeX/notebooks).
3. Mantener `MODELS` y nombres de archivos exactos.
4. NO cambiar el formato HTML/CSV de salida.
