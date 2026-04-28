# Audit — 03_models_baseline/05_metrics_baseline.py

## Propósito
Consolida los resultados de los scripts 01-04 de España: tabla de especificaciones, evaluación estática, evaluación rolling por horizonte, desglose por sub-período (pre-crisis / crisis / post-crisis), ranking por horizonte y plots. Produce un informe textual y un JSON resumen.

## Inputs / Outputs
- **Reads**:
  - `03_models_baseline/results/{arima,sarima,sarimax}_metrics.json`
  - `03_models_baseline/results/rolling_metrics.json`
  - `03_models_baseline/results/rolling_predictions.parquet`
- **Writes**:
  - `03_models_baseline/results/baseline_report.txt`
  - `03_models_baseline/results/baseline_summary.json`
  - `03_models_baseline/results/plots/rolling_mae_by_horizon.png`
  - `03_models_baseline/results/plots/error_by_period.png`
  - `03_models_baseline/results/plots/rolling_errors_h1_h12.png`

## Dependencias internas
- Manipula `sys.path` pero NO importa de `shared/` realmente.

## Métricas
- LOC: 373 · Funcionales ~290 · Comentarios/blank ~83 · `print(`: 13.

## Code smells
- `sys.path.insert` sin uso real — eliminar.
- Docstring/comentarios en español.
- 3 funciones `plot_*` con stack de hardcoded styles, colores y labels.
- Archivo grande: 7 funciones en main + main lineal. Aceptable como reporte.
- Solo 13 prints — usa principalmente outputs a archivos.

## Riesgo de refactor
**MEDIO**. Output `baseline_summary.json` no parece ser consumido por nadie más, solo usado como reporte humano. Pero los plots son figuras del TFG.

## Acciones FASE 3
1. Eliminar `sys.path.insert` no usado.
2. Logger.
3. Traducir docstrings/comentarios y labels de plots a inglés (verificar si las figuras se referencian en la memoria del TFG con titles/labels concretos).
4. Mantener esquemas de outputs.
