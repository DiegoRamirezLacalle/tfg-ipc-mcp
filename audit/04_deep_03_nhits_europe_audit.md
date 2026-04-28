# Audit — 04_models_deep/03_nhits_europe.py

## Propósito
N-HiTS para HICP Eurozona, evaluación estática por horizonte. Compacto (~48 líneas).

## Inputs / Outputs
- **Reads**: vía `_helpers_europe.load_nf_format_europe`.
- **Writes**: `08_results/nhits_europe_metrics.json`.

## Dependencias internas
- `_helpers_europe.{load_nf_format_europe, evaluate_forecast, RESULTS_DIR}`.

## Métricas
- LOC: 48 · Funcionales ~38 · Comentarios/blank ~10 · `print(`: ~6.

## Code smells
- Imports comprimidos en una línea.
- Docstring corto en español.
- Imports posiblemente innecesarios (`numpy`/`pandas`).

## Riesgo de refactor
**BAJO-MEDIO**. Output autocontenido.

## Acciones FASE 3
1. Separar imports y eliminar los no usados.
2. Logger.
3. Traducir docstring.
