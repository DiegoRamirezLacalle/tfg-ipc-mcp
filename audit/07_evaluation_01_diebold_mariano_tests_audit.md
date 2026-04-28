# Audit — 07_evaluation/01_diebold_mariano_tests.py

## Propósito
Tests Diebold-Mariano (Harvey-Leybourne-Newbold) sobre IPC España: compara C0 vs C1, cada modelo vs naive, y por sub-período. Alinea errores por (origin, horizon, fc_date).

## Inputs / Outputs
- **Reads**:
  - `08_results/{modelo}_predictions.parquet` (16+ modelos foundation)
  - `03_models_baseline/results/rolling_predictions.parquet`
- **Writes**: `08_results/diebold_mariano_results.json` (PROJECT_CONTEXT lo refiere como `_final.json`).

## Dependencias internas
- `shared.metrics.diebold_mariano`.

## Métricas
- LOC: 351 · Funcionales ~270 · Comentarios/blank ~81 · `print(`: ~25.

## Code smells
- Docstrings/comentarios en español.
- Lista hardcoded de 16 modelos en `load_foundation_preds()` — frágil si se añaden modelos nuevos.
- `SUBPERIODS` con esquema A_2021/B_2022_shock/C_2023_2024 (consistente con los foundations C1).
- `print` para tablas — pasar a logger.
- Diferencia con PROJECT_CONTEXT: el contexto cita el output como `diebold_mariano_results_final.json`; este script escribe `diebold_mariano_results.json` — verificar si hay un script de wrap-up o si simplemente el archivo `_final.json` es producto de un rename manual.

## Riesgo de refactor
**ALTO**. Genera tabla DM oficial del TFG (referenciada en PROJECT_CONTEXT y notebooks de evaluación).

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios (no traducir labels de SUBPERIODS si los notebooks downstream los usan literalmente).
3. Verificar / aclarar nombre del JSON final vs `_final.json`.
4. Mantener lista de modelos.
5. NO cambiar esquema del JSON de salida.
