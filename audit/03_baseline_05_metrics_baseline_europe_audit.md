# Audit — 03_models_baseline/05_metrics_baseline_europe.py

## Propósito
Lee `rolling_metrics_europe.json` y produce un informe textual + JSON consolidado con tablas MAE/MASE por horizonte, delta SARIMAX vs SARIMA y mejor modelo por horizonte.

## Inputs / Outputs
- **Reads**: `08_results/rolling_metrics_europe.json`.
- **Writes**:
  - `08_results/baseline_europe_report.txt`
  - `08_results/baseline_europe_summary.json`

## Dependencias internas
- `import sys` no usado.

## Métricas
- LOC: 102 · Funcionales ~80 · Comentarios/blank ~22 · `print(`: 3.

## Code smells
- `import sys` muerto.
- Docstring/comentarios en español pero también en strings del informe (titles, labels).
- Texto del reporte construido linealmente en `lines` — legible y simple.

## Riesgo de refactor
**BAJO**. Solo lectura/reporte. Si se cambia el wording del informe no afecta a nadie programáticamente.

## Acciones FASE 3
1. Eliminar `import sys`.
2. Traducir docstring y strings del informe a inglés.
3. Logger para los 3 prints.
