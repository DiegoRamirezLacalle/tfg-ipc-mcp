# Refactor Plan — tfg-forecasting

Plan de orden, esfuerzo y riesgo para el refactor en FASE 3. Este documento agrega los 103 audits en una hoja de ruta ejecutable por bloques.

---

## 1. Resumen ejecutivo

- **Total scripts auditados**: 103 (incluidos 8 stubs vacíos sin código).
- **Total LOC**: ~17.700.
- **Reglas globales del refactor** (FASE 3):
  - Funcionalidad bit-exacta (mismo input → mismo output numérico).
  - Comentarios y docstrings en inglés.
  - Sin `print(...)` de debug; usar `shared.logger.get_logger`.
  - Sin código muerto comentado.
  - Funciones con responsabilidad única, nombres descriptivos.
  - NO cambiar nombres de scripts ni esquemas de outputs (parquet/JSON).
- **Reglas metodológicas no negociables** (recordatorio de PROJECT_CONTEXT y memoria):
  - `shift +1` en exógenas para anti-leakage.
  - `StandardScaler` antes de Ridge en correctores externos (TimesFM C1).
  - Splits por fecha (DATE_TRAIN_END, DATE_TEST_END).
  - MASE scale fija sobre 2002-2020.
  - HORIZONS = [1, 3, 6, 12] y 48 orígenes rolling 2021-01 → 2024-12.

---

## 2. Orden de refactor (de menos a más dependencias)

El orden está pensado para minimizar el blast radius: empezamos por `shared/` (sin dependientes), seguimos por scripts ETL terminales y subimos por módulos hacia los consumidores finales (notebooks de evaluación).

### Bloque 0 — Limpieza de stubs vacíos (fast wins)

Eliminar tras grep de seguridad. **Esfuerzo: muy bajo (~30 min)**.

- `01_etl/01_ingest_ipc_spain.py` (vacío, suplantado por `05_clean_and_align.py`)
- `01_etl/02_ingest_hicp_ecb.py` (vacío, suplantado por `11_ingest_hicp_europe.py`)
- `01_etl/04_ingest_gdelt.py` (vacío)
- `04_models_deep/02_lstm_multivariate.py` (vacío)
- `06_models_foundation/01_timegpt_C0_no_context.py` (vacío)
- `06_models_foundation/02_timegpt_C1_with_context.py` (vacío)
- `06_models_foundation/03_timesfm_C0.py` (vacío)
- `06_models_foundation/04_timesfm_C1_residual.py` (vacío)
- `06_models_foundation/05_backtesting_rolling.py` (vacío)

### Bloque 1 — `shared/` (foundation)

Pieza base: la migración de prints a logger en el resto del repo depende de que `shared/logger.py` esté listo y pulido. **Esfuerzo: bajo (~1-2 h)**.

Orden:
1. `shared/logger.py` — pulir, traducir, añadir docstring a `get_logger`.
2. `shared/constants.py` — traducir; verificar y posiblemente eliminar constantes no usadas (`SERIES_HICP_EA`, `SERIES_ECB_RATE`, `ALL_SERIES`, `DATE_VAL_END`, `FORECAST_HORIZON`).
3. `shared/data_utils.py` — traducir; mover `import os` al top; verificar uso de `train_val_test_split` y `freeze_snapshot` (probables candidatos a eliminación).
4. `shared/metrics.py` — traducir; verificar consumidores reales de `mase` y `diebold_mariano` antes de tocar fórmulas.

**Riesgo**: bajo. Pero cualquier cambio en `shared.metrics` puede afectar `01_arima_auto*.py`, `_helpers*.py` (deep) y eval scripts. Mantener firmas y números.

### Bloque 2 — `01_etl/` (productores de datos)

Scripts ETL terminales (no importan otros del proyecto). **Esfuerzo: medio (~6-8 h en total)**.

Orden recomendado (de "leaf" a "ensambladores"):
1. `03_ingest_ecb_rates.py` (74 LOC) — bajo riesgo, solo lee CSVs.
2. `07_ingest_energy_prices.py` (150 LOC) — riesgo medio.
3. `09_ingest_institutional_signals.py` (235 LOC).
4. `11_ingest_hicp_europe.py` (228 LOC).
5. `12_ingest_europe_signals.py` (177 LOC).
6. `01_ingest_cpi_global.py` (270 LOC).
7. `05_clean_and_align.py` (154 LOC) — riesgo medio (output base de España).
8. `06_feature_engineering_exog.py` (63 LOC).
9. `08_merge_energy_features.py` (147 LOC) — riesgo ALTO (sobreescribe `features_c1.parquet`).
10. `10_ingest_institutional_signals_global.py` (324 LOC) — riesgo ALTO (entrada del modelo ★★).
11. `10_merge_institutional_features.py` (69 LOC) — riesgo medio-alto.
12. `13_build_features_c1_europe.py` (183 LOC) — riesgo ALTO (entrada del modelo ★★).

Validación: tras cada script ETL, comparar bit-a-bit el parquet generado con el de `data/processed/` actual antes del refactor (usar `pd.testing.assert_frame_equal`).

### Bloque 3 — `05_mcp_pipeline/` (productor de news_signals.parquet)

Pipeline MCP que produce señales que consume `13_build_features_c1_europe.py` y modelos C1 España. **Esfuerzo: medio (~6 h)**.

Orden:
1. `mcp_server.py` — solo logger + traducción (mantener tools API).
2. `mcp_client.py` — solo logger + traducción.
3. `agent_extractor.py` — solo logger + traducción (NO tocar enums).
4. `fetch_rss_historical.py` (887 LOC, el más grande) — solo logger + traducción.
5. `fix_signals.py` — solo logger + traducción.
6. `news_to_features.py` — normalizar el patrón `sys.path` para que coincida con el resto del repo, logger, traducir; mantener esquema EXACTO de `news_signals.parquet`.

Validación: regenerar `news_signals.parquet` desde MongoDB y comparar contra el actual.

### Bloque 4 — `03_models_baseline/` (baselines España/Europa/Global)

19 scripts. **Esfuerzo: medio-alto (~10-12 h)**.

Orden:
1. Modelos estáticos primero (más simples):
   - `02_sarima_europe.py` (84) — el más limpio.
   - `03_sarimax_europe.py` (111).
   - `01_arima_auto.py` (170).
   - `01_arima_auto_europe.py` (127) — produce JSON consumido por el rolling Europa.
   - `01_arima_auto_global.py` (204).
   - `02_sarima_global.py` (227).
   - `02_sarima_seasonal.py` (211).
   - `03_sarimax_with_exog.py` (217).
   - `03_sarimax_global.py` (339) — descarga FEDFUNDS.
2. Rolling backtesting (núcleo del baseline):
   - `04_backtesting_rolling.py` (270) — España, riesgo ALTO.
   - `04_backtesting_rolling_europe.py` (236) — riesgo ALTO.
   - `04_backtesting_rolling_global.py` (303) — riesgo ALTO.
3. AutoARIMA dinámico (paralelo al rolling):
   - `07_autoarima_spain.py` (211) — riesgo ALTO.
   - `07_autoarima_europe.py` (212) — riesgo ALTO.
   - `07_autoarima_global.py` (213) — riesgo ALTO.
4. SARIMAX C1 institutional global:
   - `06_sarimax_global_institutional.py` (248) — riesgo ALTO.
5. Reportes (consumidores):
   - `05_metrics_baseline_europe.py` (102).
   - `05_metrics_baseline.py` (373).
   - `05_metrics_baseline_global.py` (427).

Validación: tras cada rolling, diff `rolling_metrics*.json` y `rolling_predictions*.parquet` contra `baseline_pre_refactor/`.

### Bloque 5 — `04_models_deep/` (LSTM/N-BEATS/N-HiTS)

17 scripts. **Esfuerzo: medio (~6-8 h)**.

Orden:
1. Helpers primero (los importan los demás):
   - `_helpers.py` (66).
   - `_helpers_europe.py` (44).
   - `_helpers_global.py` (66).
2. Modelos individuales (estáticos):
   - `01_lstm_univariate_europe.py` (49) y `02_nbeats_europe.py` (49) y `03_nhits_europe.py` (48) — los compactos.
   - `01_lstm_univariate.py` (99) y `02_nbeats.py` (105) y `03_nhits.py` (98).
   - `01_lstm_global.py` (93) y `02_nbeats_global.py` (91) y `03_nhits_global.py` (91).
3. Rolling backtesting:
   - `04_backtesting_rolling.py` (230) — riesgo ALTO.
   - `04_backtesting_rolling_deep_europe.py` (235) — riesgo ALTO.
   - `04_backtesting_rolling_deep_global.py` (224) — riesgo ALTO.
4. Reportes:
   - `05_metrics_deep.py` (200).
   - `05_metrics_deep_global.py` (189).

### Bloque 6 — `06_models_foundation/` (~33 scripts no vacíos)

Bloque más grande del refactor. **Esfuerzo: alto (~14-18 h)**.

Orden por familia × condición:

**Spain (16 scripts)**:
1. C0 puros: `01_timesfm_C0.py`, `03_chronos2_C0.py`, `05_timegpt_C0.py`.
2. Compactos C1_inst/C1_macro (todos ~115 LOC): `09_chronos2_C1_inst`, `10_chronos2_C1_macro`, `11_timesfm_C1_inst` ★, `12_timesfm_C1_macro`, `13_timegpt_C1_inst`, `14_timegpt_C1_macro`.
3. C1 con energía (~300+ LOC):
   - `04_chronos2_C1.py`, `05_chronos2_C1_energy.py`, `08_chronos2_C1_energy_only.py`.
   - `02_timesfm_C1.py` — riesgo ALTO (Ridge + StandardScaler).
   - `06_timegpt_C1.py`, `07_timegpt_C1_energy.py`, `07_timegpt_C1_energy_only.py`.

**Global (3 scripts)**:
4. `15_chronos2_C1_inst_global.py` ★★ — MUY ALTO riesgo (modelo principal del TFG).
5. `16_timesfm_C1_inst_global.py`.
6. `17_timegpt_C1_inst_global.py`.

**Europe (12 scripts)**:
7. C0: `18_chronos2_C0_europe.py`, `19_timesfm_C0_europe.py`, `20_timegpt_C0_europe.py`.
8. C1_inst: `21_chronos2_C1_inst_europe.py`, `24_timesfm_C1_inst_europe.py`, `27_timegpt_C1_inst_europe.py`.
9. C1_mcp: `22_chronos2_C1_mcp_europe.py`, `25_timesfm_C1_mcp_europe.py`, `28_timegpt_C1_mcp_europe.py`.
10. C1_full: `23_chronos2_C1_full_europe.py`, `26_timesfm_C1_full_europe.py` ★★, `29_timegpt_C1_full_europe.py`.

**Diagnósticos** (último; consultar al usuario si conservar):
11. `diagnostico_timegpt_c1.py` (546) y `diagnostico_timegpt_c1_part2.py` (258).

### Bloque 7 — `07_evaluation/` (master de evaluación)

4 scripts; cierran el pipeline. **Esfuerzo: bajo-medio (~3-4 h)**.

Orden:
1. `01_diebold_mariano_tests.py` (351) — Spain DM.
2. `05_diebold_mariano_europe.py` (226) — Europe DM.
3. `build_metrics_summary_final.py` (182) — MASTER JSON; riesgo MUY ALTO.
4. `tabla_maestra_modelos.py` (459) — reporte HTML/CSV final.

Validación final: después de este bloque, diff `metrics_summary_final.json` contra `baseline_pre_refactor/metrics_summary_final.json` clave por clave.

---

## 3. Estimación de esfuerzo

| Bloque | Scripts | LOC aprox. | Esfuerzo aprox. |
|---|---|---|---|
| 0. Limpieza stubs | 9 | 0 | 0.5 h |
| 1. shared/ | 4 | 178 | 1-2 h |
| 2. 01_etl/ | 12 (sin stubs) | 2074 | 6-8 h |
| 3. 05_mcp_pipeline/ | 6 | 2503 | 6 h |
| 4. 03_models_baseline/ | 19 | 4285 | 10-12 h |
| 5. 04_models_deep/ | 16 (sin stub) | 1977 | 6-8 h |
| 6. 06_models_foundation/ | 33 (sin stubs) | 7025 | 14-18 h |
| 7. 07_evaluation/ | 4 | 1218 | 3-4 h |
| **Total** | **103** | **~19260** | **~46-58 h** |

El esfuerzo de "logger + traducción + verificación bit-exacta" por script es aproximadamente **15-30 min para scripts <200 LOC**, **45-90 min para scripts 300-500 LOC**.

---

## 4. Scripts de alto riesgo (verificación especial)

Scripts cuyos outputs se publican en la memoria del TFG y/o son consumidos por múltiples downstream. Para estos, la validación post-refactor (diff bit-exacto contra `baseline_pre_refactor/`) es OBLIGATORIA antes de seguir al siguiente.

### Riesgo MUY ALTO (★★ del TFG)
- `06_models_foundation/15_chronos2_C1_inst_global.py` — único MASE<1.0 a h=12.
- `06_models_foundation/26_timesfm_C1_full_europe.py` — rompe MAE<2.0 a h=12 Europa.
- `07_evaluation/build_metrics_summary_final.py` — JSON master del TFG.

### Riesgo ALTO

**ETL críticos**:
- `01_etl/08_merge_energy_features.py` (sobreescribe `features_c1.parquet`).
- `01_etl/10_ingest_institutional_signals_global.py` (input del ★★ Global).
- `01_etl/13_build_features_c1_europe.py` (input del ★★ Europe).
- `05_mcp_pipeline/news_to_features.py` (genera `news_signals.parquet`).

**Baselines críticos**:
- `03_models_baseline/04_backtesting_rolling.py` (Spain rolling baseline).
- `03_models_baseline/04_backtesting_rolling_europe.py`.
- `03_models_baseline/04_backtesting_rolling_global.py`.
- `03_models_baseline/06_sarimax_global_institutional.py`.
- `03_models_baseline/07_autoarima_{spain,europe,global}.py` (3 scripts).
- `03_models_baseline/03_sarimax_global.py` (descarga FEDFUNDS).

**Deep críticos**:
- `04_models_deep/04_backtesting_rolling*.py` (3 scripts).

**Foundation con corrección Ridge** (sensibilidad a leakage / StandardScaler):
- `06_models_foundation/02_timesfm_C1.py`.
- `06_models_foundation/11_timesfm_C1_inst.py` ★.
- `06_models_foundation/12_timesfm_C1_macro.py`.
- `06_models_foundation/16_timesfm_C1_inst_global.py`.
- `06_models_foundation/24_timesfm_C1_inst_europe.py`.
- `06_models_foundation/25_timesfm_C1_mcp_europe.py`.

**Foundation con cuantiles / future_covariates** (cualquier cambio en la neutralización rompe paridad):
- Todos los `06_models_foundation/{04,05,08,09,10,21,22,23}_chronos2_*.py`.
- Todos los `06_models_foundation/{06,07,13,14,17,27,28,29}_timegpt_*.py`.

**TimeGPT con `.env` / API de pago** (no se puede re-correr libremente):
- `06_models_foundation/{05,06,07,07_energy_only,13,14,17,20,27,28,29}_timegpt_*.py`.

**Evaluación**:
- `07_evaluation/01_diebold_mariano_tests.py`.
- `07_evaluation/05_diebold_mariano_europe.py`.

### Patrones a no romper bajo ningún concepto

1. `_lag1` calculado tras un `shift(1)` global (es lag-2 efectivo). Aparece en `09_ingest_institutional_signals.py`, `10_ingest_institutional_signals_global.py`, `12_ingest_europe_signals.py`. **NO refactorizar la fórmula** aunque parezca "más limpio".
2. `MCP_NEUTRAL_COLS` con neutralización a 0 en `future_covariates` y `signal_available` neutralizado a 1. Aparece en todos los Chronos-2 C1 con MCP. **NO cambiar**.
3. `mase_scale = mean(|y[t] - y[t-12]|)` calculada sobre el train inicial (2002-2020). Recalculada en cada script de rolling — NO unificar para evitar discrepancias.
4. `Q_IDX = {"p10": 2, "p50": 10, "p90": 18}` para Chronos-2.
5. TimesFM `max_context=512`, `max_horizon=12`, `per_core_batch_size=1`.

---

## 5. Validación entre bloques

Tras cada script refactorizado:

```bash
# 1. Ejecutar el script
python tfg-forecasting/<bloque>/<script>.py

# 2. Comparar el output JSON o parquet contra baseline_pre_refactor/
python -c "
import pandas as pd, json
# para JSON:
old = json.load(open('tfg-forecasting/08_results/baseline_pre_refactor/<file>.json'))
new = json.load(open('tfg-forecasting/08_results/<file>.json'))
assert old == new, 'JSON differs'

# para parquet:
old = pd.read_parquet('.../baseline_pre_refactor/<file>.parquet')
new = pd.read_parquet('.../<file>.parquet')
pd.testing.assert_frame_equal(old, new)
"
```

Si la comparación falla:
1. NO hacer commit.
2. Identificar la divergencia (diff línea a línea sobre los JSON, columna a columna sobre los parquets).
3. Causas comunes a revisar: orden de operaciones de `shift`, `ffill` antes/después, `astype(float)` redondeos, omisión de `set_index`/`set_index('date')`, cambio en `random_seed`, semilla de torch.
4. Revertir cambios y reintentar con corrección puntual.

---

## 6. Recomendaciones operativas

- **Trabajar un script a la vez** y commitear por script (`git commit -m "refactor: clean <script_name>"`).
- **Logger central**: importar siempre con `logger = get_logger(__name__)`.
- **Mantener `random_seed=42` y `enable_progress_bar=False`** en todos los scripts de NeuralForecast — no los cambies aunque parezca obvio.
- Para scripts con LLM (`agent_extractor.py`) o API de pago (`timegpt_*.py`): si la lógica no cambia, no es necesario re-ejecutar el modelo; basta validar que el script sigue corriendo (smoke test con `--test-run` cuando aplique).
- **`__pycache__/` y `lightning_logs/`** pueden quedar fuera del refactor; son artefactos.
- Consultar antes de eliminar: `features_exog.parquet` (¿tiene consumidores vivos?), `02_sarima_seasonal.py` (¿se usa o suplantado por `01_arima_auto.py`?), los dos `diagnostico_timegpt_c1*.py`.

---

## 7. Salidas de la auditoría

Estructura del directorio `audit/`:
- 4 audits `shared/` (`shared_*_audit.md`)
- 15 audits `01_etl/` (`01_etl_*_audit.md`)
- 19 audits `03_models_baseline/` (`03_baseline_*_audit.md`)
- 17 audits `04_models_deep/` (`04_deep_*_audit.md`)
- 6 audits `05_mcp_pipeline/` (`05_mcp_*_audit.md`)
- 38 audits `06_models_foundation/` (`06_foundation_*_audit.md`)
- 4 audits `07_evaluation/` (`07_evaluation_*_audit.md`)
- Este `REFACTOR_PLAN.md`

Total: **103 audits + plan**. Listo para revisión.
