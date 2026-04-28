# Audit — 04_models_deep/01_lstm_univariate_europe.py

## Propósito
LSTM univariante para HICP Eurozona. Versión compacta de 49 líneas con `max_steps=300` (en vez de 500), evaluación estática y `lstm_europe_metrics.json` como output.

## Inputs / Outputs
- **Reads**: vía `_helpers_europe.load_nf_format_europe`.
- **Writes**: `08_results/lstm_europe_metrics.json`.

## Dependencias internas
- `_helpers_europe.load_nf_format_europe`, `_helpers_europe.evaluate_forecast`, `_helpers_europe.RESULTS_DIR`.

## Métricas
- LOC: 49 · Funcionales ~40 · Comentarios/blank ~9 · `print(`: 6.

## Code smells
- Imports comprimidos en una línea (`import json, sys, warnings`) — anti-PEP8.
- `import numpy as np`, `import pandas as pd` posiblemente no usados.
- `sys.path.insert(0, str(Path(__file__).parent))` para importar `_helpers_europe` — patrón frágil pero funciona.
- Docstring brevísimo en español.
- 6 prints.

## Riesgo de refactor
**BAJO-MEDIO**. Output autocontenido.

## Acciones FASE 3
1. Separar imports y eliminar los no usados.
2. Logger.
3. Traducir docstring.
4. Mantener hiperparámetros y seed.
