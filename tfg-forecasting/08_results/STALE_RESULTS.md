# ⚠️ Stale results — pending re-run

Two methodological **code** fixes were applied on branch `refactor-clean` but the
stored metrics/predictions were **not** regenerated yet (deliberate: re-running
the foundation models needs the model weights + heavy CPU inference). Until the
scripts below are re-run, the JSON/parquet/table numbers for these runs reflect
the **old** code.

## Affected runs

| Run | Script | Fix applied | Stale output files |
|-----|--------|-------------|--------------------|
| `chronos2_C1` (Spain, C1_mcp) | `06_models_foundation/04_chronos2_C1.py` | #1 — no longer reads realised future ECB rates; carries forward last value known at origin | `chronos2_C1_metrics.json`, `chronos2_C1_predictions.parquet`, `chronos2_C1_subperiod_metrics.json` |
| `timesfm_C1` (Spain, C1_mcp) | `06_models_foundation/02_timesfm_C1.py` | #2 — Ridge now uses `StandardScaler` | `timesfm_C1_metrics.json`, `timesfm_C1_predictions.parquet` |
| `timesfm_C1_inst` (Spain) | `06_models_foundation/11_timesfm_C1_inst.py` | #2 — StandardScaler | `timesfm_C1_inst_metrics.json`, `timesfm_C1_inst_predictions.parquet` |
| `timesfm_C1_macro` (Spain) | `06_models_foundation/12_timesfm_C1_macro.py` | #2 — StandardScaler | `timesfm_C1_macro_metrics.json`, `timesfm_C1_macro_predictions.parquet` |
| `timesfm_C1_inst_global` (Global) | `06_models_foundation/16_timesfm_C1_inst_global.py` | #2 — StandardScaler | `timesfm_C1_inst_global_metrics.json`, `timesfm_C1_inst_global_predictions.parquet` |
| `timesfm_C1_mcp_europe` (Europe) | `06_models_foundation/25_timesfm_C1_mcp_europe.py` | #2 — StandardScaler | `timesfm_C1_mcp_europe_metrics.json`, `timesfm_C1_mcp_europe_predictions.parquet` |

## Downstream files that must be rebuilt afterwards

These consolidate the runs above, so they are stale too until rebuilt:

- `metrics_summary_final.json` (Spain master — includes `timesfm_C1`, `chronos2_C1`, `timesfm_C1_inst`, `timesfm_C1_macro`)
- `tabla_maestra.md` / `tabla_maestra.html`
- `diebold_mariano_results_final.json` (Spain DM — compares `chronos2_C1`, `timesfm_C1`, `timesfm_C1_inst`, …)
- `diebold_mariano_results_europe.json` (includes `timesfm_C1_mcp_europe`)
- Prose figures quoting these numbers: `README.md` and `PROJECT_CONTEXT.md`
  (e.g. the Spain "TimesFM C1_inst h=1 = 0.423 ★" highlight may shift after rescaling).

## Re-run recipe (from `tfg-forecasting/`)

```bash
# 1. Re-run the 6 fixed model scripts (needs TimesFM-2.5-200M + Chronos-2 weights)
python 06_models_foundation/04_chronos2_C1.py
python 06_models_foundation/02_timesfm_C1.py
python 06_models_foundation/11_timesfm_C1_inst.py
python 06_models_foundation/12_timesfm_C1_macro.py
python 06_models_foundation/16_timesfm_C1_inst_global.py
python 06_models_foundation/25_timesfm_C1_mcp_europe.py

# 2. Rebuild the consolidated tables / DM tests
python 07_evaluation/build_metrics_summary_final.py
python 07_evaluation/01_diebold_mariano_tests.py
python 07_evaluation/05_diebold_mariano_europe.py
python 07_evaluation/tabla_maestra_modelos.py
# then re-execute the evaluation notebooks (02_compare_all_models, 03_evaluation_global,
# 04_evaluation_europe) to refresh figures, and update the numbers in README.md / PROJECT_CONTEXT.md

# 3. Validate integrity
python tests/check_artifacts_and_leakage.py
```

Delete this file once the re-run is done and the numbers are refreshed.
