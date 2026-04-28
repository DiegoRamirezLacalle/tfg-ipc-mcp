# Audit — 04_models_deep/01_lstm_univariate.py

## Propósito
LSTM univariante (NeuralForecast) sobre IPC España: ajusta un LSTM por horizonte (1,3,6,12) con `input_size=24, hidden=64, max_steps=500, seed=42`; evalúa estáticamente sobre validation y guarda `lstm_metrics.json`.

## Inputs / Outputs
- **Reads**: `data/processed/ipc_spain_index.parquet` (vía `_helpers.load_nf_format`).
- **Writes**: `04_models_deep/results/lstm_metrics.json`.

## Dependencias internas
- `_helpers.load_nf_format`, `_helpers.evaluate_forecast`, `_helpers.print_comparison`, `_helpers.RESULTS_DIR`.

## Métricas
- LOC: 99 · Funcionales ~70 · Comentarios/blank ~29 · `print(`: ~12.

## Code smells
- Docstring/comentarios en español.
- `import pandas as pd` posiblemente no usado.
- Re-entrena el modelo desde cero por cada horizonte — caro pero intencional (cada `h` requiere su propio modelo h-step).
- `enable_progress_bar=False` correcto para batch.

## Riesgo de refactor
**MEDIO**. Output `lstm_metrics.json` lo lee el `04_backtesting_rolling.py` (deep, mismo módulo) — no, lo lee `05_metrics_deep.py` indirectamente vía `deep_rolling_metrics.json`. Verificar.

## Acciones FASE 3
1. Verificar imports no usados.
2. Logger.
3. Traducir docstrings/comentarios.
4. Mantener hiperparámetros EXACTOS y `random_seed=42`.
