# Audit — 06_models_foundation/05_timegpt_C0.py

## Propósito
TimeGPT (Nixtla cloud API) sobre IPC España condición C0. Rolling-origin 48 orígenes con `client.forecast(df, h=h, freq='MS')`. Carga `NIXTLA_API_KEY` desde `.env` en raíz del monorepo. Soporta `--test-run` (5 orígenes) y `--full` (48) para control de coste.

## Inputs / Outputs
- **Reads**:
  - `data/processed/ipc_spain_index.parquet`
  - `.env` (NIXTLA_API_KEY)
- **Writes**:
  - `08_results/timegpt_C0_predictions.parquet`
  - `08_results/timegpt_C0_metrics.json`

## Dependencias internas
- `shared.constants`. `dotenv` para `.env`.

## Métricas
- LOC: 256 · Funcionales ~195 · Comentarios/blank ~61 · `print(`: ~20.

## Code smells
- Docstring/comentarios en español.
- API key check con string mágico `"tu_api_key_aqui"`.
- `argparse` con `--test-run`/`--full` — buen patrón.
- Maneja errores de API con try/except.
- `SERIES_ID = "ipc_spain"` constante.

## Riesgo de refactor
**ALTO**. Resultados publicados, y cada llamada cuesta dinero — no se puede re-correr a la ligera.

## Acciones FASE 3
1. Logger.
2. Traducir docstring/comentarios.
3. Mantener `SERIES_ID`, formato Nixtla df, controles de coste.
4. NO cambiar esquemas.
