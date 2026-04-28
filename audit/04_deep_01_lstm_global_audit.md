# Audit — 04_models_deep/01_lstm_global.py

## Propósito
LSTM univariante para CPI Global. Mismos hiperparámetros que el LSTM España (`input_size=24, hidden=64, max_steps=500, seed=42`) pero sobre `cpi_global_rate`. Evaluación estática por horizonte.

## Inputs / Outputs
- **Reads**: vía `_helpers_global.load_nf_format_global`.
- **Writes**: `08_results/lstm_global_metrics.json`.

## Dependencias internas
- `_helpers_global.{RESULTS_DIR, load_nf_format_global, evaluate_forecast, print_comparison}`.

## Métricas
- LOC: 93 · Funcionales ~70 · Comentarios/blank ~23 · `print(`: ~12.

## Code smells
- Docstrings/comentarios en español.
- `sys.path.insert(0, str(Path(__file__).resolve().parent))` para que `_helpers_global` resuelva.
- Patrón idéntico a `01_lstm_univariate.py` con I/O y unique_id distintos — duplicación esperada por separación por serie.

## Riesgo de refactor
**MEDIO**. Output `lstm_global_metrics.json` referenciado en PROJECT_CONTEXT.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios.
3. Mantener hiperparámetros y seed.
