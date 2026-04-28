# Audit — 07_evaluation/05_diebold_mariano_europe.py

## Propósito
Tests Diebold-Mariano sobre HICP Eurozona. Compara: (1) C0 vs C1_inst/C1_mcp/C1_full por familia, (2) C1_inst vs C1_mcp, (3) C1_inst/C1_mcp vs C1_full, (4) cada foundation C0 vs SARIMA Europe, (5) cross-model C0.

## Inputs / Outputs
- **Reads**: `08_results/{modelo}_europe_predictions.parquet` (12 modelos: C0/C1_inst/C1_mcp/C1_full × 3 familias).
- **Writes**: `08_results/diebold_mariano_results_europe.json` (CONTRATO; cargado por notebooks).

## Dependencias internas
- `shared.metrics.diebold_mariano`.

## Métricas
- LOC: 226 · Funcionales ~175 · Comentarios/blank ~51 · `print(`: ~15.

## Code smells
- Docstrings/comentarios en español.
- Lista hardcoded de 12 modelos.
- `SUBPERIODS` Europa (pre_shock/shock/normalizacion) distintos a Spain (A_2021/B_2022_shock/C_2023_2024). Inconsistencia documentada pero compatible.

## Riesgo de refactor
**ALTO**. Output usado por notebooks Europa.

## Acciones FASE 3
1. Logger.
2. Traducir docstrings/comentarios (cuidado con labels de SUBPERIODS).
3. Mantener lista de modelos.
4. NO cambiar esquema del JSON de salida.
