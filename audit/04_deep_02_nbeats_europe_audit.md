# Audit — 04_models_deep/02_nbeats_europe.py

## Propósito
N-BEATS para HICP Eurozona, evaluación estática por horizonte, output `nbeats_europe_metrics.json`. Compacto (49 líneas).

## Inputs / Outputs
- **Reads**: vía `_helpers_europe.load_nf_format_europe`.
- **Writes**: `08_results/nbeats_europe_metrics.json`.

## Dependencias internas
- `_helpers_europe.{load_nf_format_europe, evaluate_forecast, RESULTS_DIR}`.

## Métricas
- LOC: 49 · Funcionales ~40 · Comentarios/blank ~9 · `print(`: ~6.

## Code smells
- Imports comprimidos en una sola línea (anti-PEP8).
- `numpy`/`pandas` posiblemente no usados.
- Patrón idéntico al `01_lstm_univariate_europe.py`.

## Riesgo de refactor
**BAJO-MEDIO**. Output autocontenido, número publicado en PROJECT_CONTEXT.

## Acciones FASE 3
1. Separar imports y eliminar los no usados.
2. Logger.
3. Traducir docstring.
4. Mantener hiperparámetros.
