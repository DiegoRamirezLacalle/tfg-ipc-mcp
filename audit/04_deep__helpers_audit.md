# Audit — 04_models_deep/_helpers.py

## Propósito
Helpers compartidos para los modelos deep España: `load_nf_format()` carga IPC y devuelve formato long NeuralForecast (unique_id="IPC_ESP", ds, y) más un array `y_train_values` para MASE; `evaluate_forecast()` envuelve `shared.metrics`; `print_comparison()` imprime tabla punto a punto.

## Inputs / Outputs
- **Reads**: `data/processed/ipc_spain_index.parquet` (en `load_nf_format`).
- **Writes**: nada.

## Dependencias internas
- `shared.constants` (DATE_TRAIN_END, DATE_VAL_END), `shared.metrics`.

## Métricas
- LOC: 66 · Funcionales ~50 · Comentarios/blank ~16 · `print(`: 2 (en `print_comparison`).

## Code smells
- Docstrings en español.
- `import numpy as np` usado.
- `print_comparison` mezcla logging y formato — debería pasar a logger o eliminarse.
- `RESULTS_DIR = .../results` (local), inconsistente con Europa/Global que apuntan a `08_results/`. Aceptable si es intencional.

## Riesgo de refactor
**MEDIO**. Importado por `01_lstm_univariate.py`, `02_nbeats.py`, `03_nhits.py`, `04_backtesting_rolling.py`, `05_metrics_deep.py`. Cambiar firmas rompe ~5 scripts.

## Acciones FASE 3
1. Traducir docstrings.
2. Logger en `print_comparison` (o moverlo a un debug-only helper).
3. Mantener firmas de `load_nf_format`, `evaluate_forecast`, `print_comparison`.
