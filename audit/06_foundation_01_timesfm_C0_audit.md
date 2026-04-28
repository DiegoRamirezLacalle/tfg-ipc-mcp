# Audit — 06_models_foundation/01_timesfm_C0.py

## Propósito
TimesFM 2.5 (200M, PyTorch) sobre IPC España, condición C0 (solo histórico). Rolling-origin 48 orígenes × horizontes [1,3,6,12]. Carga `google/timesfm-2.5-200m-pytorch` con `max_context=512` (16 patches × 32) y `max_horizon=12`.

## Inputs / Outputs
- **Reads**: `data/processed/features_exog.parquet` (`indice_general`).
- **Writes**: 
  - `08_results/timesfm_C0_predictions.parquet`
  - `08_results/timesfm_C0_metrics.json`

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_TEST_END).

## Métricas
- LOC: 191 · Funcionales ~150 · Comentarios/blank ~41 · `print(`: ~12.

## Code smells
- Docstring en español sin tildes.
- `import timesfm` perezoso dentro de `load_model` (correcto; evita carga lenta si no se llama main).
- 12 prints — pasar a logger.
- MAE/RMSE/MASE re-implementados.
- `tqdm` para progreso.

## Riesgo de refactor
**ALTO**. Salida `timesfm_C0_metrics.json` aparece en `metrics_summary_final.json`. Bit-exactitud crítica.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener `max_context=512`, `max_horizon=12`, model id, y la lógica de rolling.
4. NO cambiar esquema de outputs.
