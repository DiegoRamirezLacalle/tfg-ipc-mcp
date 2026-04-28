# PROJECT_CONTEXT.md
> Memoria externa del proyecto para sesiones de Claude Code sin contexto previo.
> Última actualización: 2026-04-27

---

## 1. Resumen del TFG

**Título**: *Evaluación de Foundation Models de Series Temporales para la Predicción de Inflación con Señales MCP*

**Hipótesis central**: ¿Mejoran los foundation models de series temporales (Chronos-2, TimesFM, TimeGPT) la predicción de inflación frente a los baselines estadísticos clásicos (ARIMA/SARIMA)? ¿Las señales exógenas construidas con un pipeline MCP añaden valor predictivo?

**Autor**: Diego Ramirez · `analacallealvarez@gmail.com`

### Dos partes del repositorio

| Carpeta | Qué es |
|---|---|
| `tfg-forecasting/` | Investigación científica: datos, modelos, evaluación, resultados |
| `tfg-arquitectura/` | Plataforma web: FastAPI backend + React frontend + Docker |

---

## 2. Arquitectura del repositorio

```
tfg-ipc-mcp/
├── PROJECT_CONTEXT.md          ← este archivo
├── pyproject.toml              ← dependencias Python del monorepo
├── docker-compose.yml          ← orquestación de servicios web
├── shared/                     ← utilidades compartidas entre scripts
│   ├── constants.py            ← HORIZONS=[1,3,6,12], rutas de datos, etc.
│   ├── data_utils.py           ← funciones de carga de parquets
│   ├── metrics.py              ← MAE, RMSE, MASE, naive_scale
│   └── logger.py               ← logger estándar
│
├── tfg-forecasting/
│   ├── data/
│   │   ├── raw/                ← CSVs descargados (INE, ECB, FRED, etc.)
│   │   └── processed/          ← parquets listos para modelos (ver §4)
│   │
│   ├── 01_etl/                 ← scripts de ingesta y feature engineering
│   ├── 02_eda/                 ← notebooks de análisis exploratorio
│   ├── 03_models_baseline/     ← ARIMA, SARIMA, SARIMAX, AutoARIMA
│   ├── 04_models_deep/         ← LSTM, N-BEATS, N-HiTS (NeuralForecast)
│   ├── 05_mcp_pipeline/        ← pipeline MCP para España (GDELT + BCE)
│   ├── 05_mcp_pipeline_global/ ← pipeline MCP para Global (FOMC + BLS)
│   ├── 06_models_foundation/   ← Chronos-2, TimesFM, TimeGPT (30 scripts)
│   ├── 07_evaluation/          ← notebooks de evaluación y comparativa
│   ├── 08_results/             ← todos los JSON de métricas y figuras PNG
│   ├── configs/                ← configs YAML (no usado aún en producción)
│   └── lightning_logs/         ← logs PyTorch Lightning (~263 versiones)
│
└── tfg-arquitectura/
    ├── backend/                ← FastAPI (main.py, config.py, app/)
    ├── frontend/               ← React (package.json, src/)
    ├── gateway/                ← nginx / reverse proxy
    ├── db/                     ← esquemas de base de datos
    └── infra/                  ← IaC (Terraform/Docker)
```

### Módulo 01_etl — Scripts de ingesta

| Script | Qué hace |
|---|---|
| `01_ingest_ipc_spain.py` | Descarga IPC España (INE) → `ipc_spain_index.parquet` |
| `01_ingest_cpi_global.py` | Descarga CPI World Bank → `cpi_global_monthly.parquet` |
| `02_ingest_hicp_ecb.py` | Descarga HICP Eurozona (ECB SDW) → `hicp_europe_index.parquet` |
| `03_ingest_ecb_rates.py` | Tipos BCE (DFR, MRR) → `ecb_rates_monthly.parquet` |
| `04_ingest_gdelt.py` | GDELT BigQuery → `news_signals.parquet` |
| `05_clean_and_align.py` | Alineación temporal a freq mensual |
| `06_feature_engineering_exog.py` | Construcción de lags/MAs/diffs → `features_exog.parquet` |
| `07_ingest_energy_prices.py` | Brent + TTF gas → `energy_prices_monthly.parquet` |
| `08_merge_energy_features.py` | Fusiona energía en features_c1 |
| `09_ingest_institutional_signals.py` | EPU Europe + señales ECB (Spain/Europe) |
| `10_ingest_institutional_signals_global.py` | GEPU, GSCPI, VIX, DXY, USG10Y, FEDFUNDS → `institutional_signals_monthly.parquet` |
| `10_merge_institutional_features.py` | Fusiona institucionales en features_c1 |
| `11_ingest_hicp_europe.py` | Actualización HICP |
| `12_ingest_europe_signals.py` | ESI Eurozona, breakeven 5y, EUR/USD → `europe_signals_monthly.parquet` |
| `13_build_features_c1_europe.py` | Ensambla `features_c1_europe.parquet` |

### Módulo 07_evaluation — Notebooks de evaluación

| Notebook | Contenido |
|---|---|
| `02_compare_all_models.ipynb` | Evaluación completa España: ranking, perfiles MAE, heatmap ΔC1, DM tests |
| `03_evaluation_global.ipynb` | Evaluación IPC Global con AutoARIMA |
| `03_regime_analysis.ipynb` | Análisis por regímenes (pre-pandemia, COVID, shock energético, post-shock) |
| `04_ablation_context_type.ipynb` | Ablación C0 vs C1_inst vs C1_mcp vs C1_full |
| `04_evaluation_europe.ipynb` | Evaluación HICP Europa con AutoARIMA |
| `05_spain_vs_global_vs_europe.ipynb` | Comparativa transversal de las 3 series |
| `01_diebold_mariano_tests.py` | DM tests España → `diebold_mariano_results_final.json` |
| `05_diebold_mariano_europe.py` | DM tests Europa → `diebold_mariano_results_europe.json` |
| `build_metrics_summary_final.py` | Construye `metrics_summary_final.json` (master Spain) |
| `tabla_maestra_modelos.py` | Tabla LaTeX de todos los modelos |

---

## 3. Decisiones metodológicas clave

### Splits temporales

```
Entrenamiento inicial (train): 2002-01 a 2020-12  (228 obs)
Validación (implícita en backtesting): 2021-01 a 2022-12
Test (rolling-origin): 2021-01 a 2024-12  (48 orígenes)
```

El split se hace por **fecha**, no por índice numérico. El entrenamiento inicial se usa para:
- Calcular la escala MASE (MAE del naive lag-12 sobre 2002-2020)
- Seleccionar órdenes ARIMA/SARIMA fijos (1 sola vez, sin re-selección)
- Entrenar los modelos deep learning en ventana de 24 meses

### Backtesting rolling-origin expandido

```
Orígenes: 48 (mensualmente desde 2021-01-01 hasta 2024-12-01)
Horizontes: h = 1, 3, 6, 12 meses
Ventana: EXPANDIDA (cada origen añade 1 mes al histórico)
```

Para el origen `t`, el modelo se entrena sobre `[2002-01, t]` y predice `[t+1, t+h]`.
Los foundation models reciben la serie completa hasta `t` sin re-entrenamiento.

### Métricas

```python
MAE   = mean(|y_pred - y_true|)
RMSE  = sqrt(mean((y_pred - y_true)^2))
MASE  = MAE / naive_scale
naive_scale = mean(|y_t - y_{t-12}|)  # calculado sobre 2002-2020

# Escala MASE por serie:
# España : 1.4051 pp
# Global : 1.1720 pp
# Europa : 1.4558 pp
```

MASE < 1 → el modelo bate al naive seasonal lag-12.

### Diebold-Mariano (DM tests)

- Test bilateral, corrección Harvey-Leybourne-Newbold (HLN)
- Null: ambos modelos tienen el mismo error cuadrático medio
- Se aplica por horizonte h y por subperiodo
- Subperiodos: 2021-completo, Shock-2022 (2022-01 a 2022-12), Post-shock (2023-2024)
- Resultados en `08_results/diebold_mariano_results_final.json` y `*_europe.json`

### Leakage prevention (shift +1 en exógenas)

**CRÍTICO**: todas las señales exógenas se shiftan +1 mes antes de pasarlas al modelo.

```python
# En todos los scripts de foundation con C1:
exog_shifted = exog_df.shift(1)  # la señal de t-1 predice t
```

Sin este shift, se usa información del futuro. Error detectado en sesiones previas con TimeGPT (MAE +534% sin StandardScaler + sin shift).

### Normalización de señales exógenas

**CRÍTICO**: las señales tienen escalas muy distintas (EPU std~65, diff(HICP) std~0.44).
La corrección Ridge **requiere** StandardScaler antes de ajustar:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# ridge.fit(X_train_scaled, residuals_train)
```

Sin normalización, el ridge produce una corrección constante de ~+1.11 pp.

---

## 4. Datasets y exógenas

### Series objetivo

| Archivo | Variable | Período | Obs | Fuente |
|---|---|---|---|---|
| `ipc_spain_index.parquet` | `indice_general` (+ 13 subíndices) | 2002-01 → 2026-01 | 289 | INE |
| `hicp_europe_index.parquet` | `hicp_index` | 2002-01 → 2024-12 | 276 | ECB SDW |
| `cpi_global_monthly.parquet` | `cpi_global_rate` | 2002-01 → 2024-12 | 276 | World Bank |

**Atención**: `hicp_europe_index.parquet` tiene índice entero (0..275) y `date` como columna normal (no como DatetimeIndex). Los scripts de Europe deben hacer `df = df.set_index('date')` antes de usar.

### Señales exógenas — España (`features_c1.parquet`, 282 obs × 34 cols)

| Grupo | Columnas clave |
|---|---|
| IPC objetivo | `indice_general` |
| Tipos BCE | `dfr`, `mrr`, `dfr_diff`, `dfr_lag3`, `dfr_lag6`, `dfr_lag12` |
| GDELT (tono noticias ECB) | `gdelt_avg_tone`, `gdelt_goldstein`, `gdelt_n_articles`, `gdelt_tone_ma3`, `gdelt_tone_ma6` |
| Señales BCE (MCP) | `bce_shock_score`, `bce_uncertainty`, `bce_tone`, `bce_tone_numeric`, `bce_cumstance` |
| Señales INE (MCP) | `ine_surprise_score`, `ine_topic`, `ine_inflacion`, `dominant_topic` |
| Energía | `brent_log`, `brent_ret`, `brent_ma3`, `brent_lag1`, `ttf_log`, `ttf_ret`, `ttf_ma3`, `ttf_lag1` |
| EPU Europe | `epu_europe_log`, `epu_europe_ma3`, `epu_europe_lag1` |
| Disponibilidad MCP | `signal_available` (0/1 por mes) |

### Señales exógenas — Global (`features_c1_global_institutional.parquet`, 276 obs × 31 cols)

Señales institucionales globales, todas con sufijos `_ma3`, `_lag1`, `_diff`:
`gepu` (Global EPU), `imf_comm` (IMF commodity), `dxy` (US Dollar Index),
`vix`, `usg10y` (US Treasury 10y), `fedfunds` (Fed Funds Rate),
`gscpi` (Global Supply Chain Pressure Index), `gpr` (Geopolitical Risk),
`brent_log`, `dfr` (ECB deposit rate). Más `cpi_global_rate` (target).

Selección final de columnas en `08_results/c1_global_inst_selected_cols.json`.

### Señales exógenas — Europa (`features_c1_europe.parquet`, 276 obs × 15 cols)

`hicp_index`, `dfr`, `dfr_ma3`, `mrr`, `brent_ma3`, `ttf_ma3`,
`epu_europe_ma3`, `esi_eurozone` (Economic Sentiment), `breakeven_5y_lag1`,
`eurusd_ma3`, `bce_shock_score`, `bce_tone_numeric`, `bce_cumstance`,
`gdelt_tone_ma6`, `signal_available`.

---

## 5. Modelos implementados

### Condiciones experimentales

| Condición | Descripción |
|---|---|
| **C0** | Univariante — solo la serie objetivo, sin señales exógenas |
| **C1_inst** | Con señales institucionales (tipos BCE, EPU, energía, indicadores macro) |
| **C1_mcp** | Con señales de texto MCP (GDELT tone, BCE press releases procesados) |
| **C1_full** | C1_inst + C1_mcp combinados (solo Europa) |
| **C1_energy** | Solo señales de energía (Brent, TTF) |
| **C1_macro** | Mix de señales macro (Brent + TTF + EPU) |

### Baselines estadísticos (03_models_baseline/)

| Modelo | Script | Series | Notas |
|---|---|---|---|
| Naive lag-12 | implícito en métricas | todas | benchmark de referencia |
| ARIMA | `01_arima_auto{_europe,_global}.py` | todas | órdenes fijos seleccionados por auto_arima 1× |
| ARIMA(1,1,1) | — | Global | variante fija para comparación |
| SARIMA | `02_sarima{_europe,_global}.py` | España/Europa | incluye componente estacional (1,0,1,12) |
| SARIMAX | `03_sarimax{_europe,_global}.py` | todas | SARIMA + exógenas institucionales |
| AutoARIMA | `07_autoarima_{spain,europe,global}.py` | todas | reselección de órdenes en cada origen |

**AutoARIMA**: usa `pmdarima.auto_arima(seasonal=True, m=12, stepwise=True, information_criterion='aic', max_p=3, max_q=3, max_P=2, max_Q=2)`. Re-ajusta órdenes en cada uno de los 48 orígenes rolling.

### Deep learning (04_models_deep/)

| Modelo | Script | Series | Librería |
|---|---|---|---|
| LSTM univariante | `01_lstm_univariate{_europe}.py` | España/Europa | PyTorch |
| LSTM global | `01_lstm_global.py` | Global | PyTorch |
| N-BEATS | `02_nbeats{_europe,_global}.py` | todas | NeuralForecast |
| N-HiTS | `03_nhits{_europe,_global}.py` | todas | NeuralForecast |

Helpers compartidos: `_helpers.py`, `_helpers_europe.py`, `_helpers_global.py`.
Logs en `lightning_logs/` (~263 versiones).

### Foundation models (06_models_foundation/)

#### España (métricas en `metrics_summary_final.json`)

| Modelo | Script | Condición |
|---|---|---|
| `timesfm_C0` | `01_timesfm_C0.py` / `03_timesfm_C0.py` | C0 |
| `timesfm_C1` | `02_timesfm_C1.py` | C1_mcp (GDELT) |
| `timesfm_C1_inst` | `11_timesfm_C1_inst.py` | C1_inst (EPU Europe) ★ |
| `timesfm_C1_macro` | `12_timesfm_C1_macro.py` | C1_macro |
| `chronos2_C0` | `03_chronos2_C0.py` | C0 |
| `chronos2_C1` | `04_chronos2_C1.py` | C1_mcp |
| `chronos2_C1_inst` | `09_chronos2_C1_inst.py` | C1_inst |
| `chronos2_C1_macro` | `10_chronos2_C1_macro.py` | C1_macro |
| `chronos2_C1_energy` | `05_chronos2_C1_energy.py` | C1_energy |
| `chronos2_C1_energy_only` | `08_chronos2_C1_energy_only.py` | solo energía |
| `timegpt_C0` | `05_timegpt_C0.py` | C0 |
| `timegpt_C1` | `06_timegpt_C1.py` | C1_mcp |
| `timegpt_C1_inst` | `13_timegpt_C1_inst.py` | C1_inst |
| `timegpt_C1_macro` | `14_timegpt_C1_macro.py` | C1_macro |
| `timegpt_C1_energy` | `07_timegpt_C1_energy.py` | C1_energy |
| `timegpt_C1_energy_only` | `07_timegpt_C1_energy_only.py` | solo energía |

#### Global (métricas en `rolling_metrics_global.json` + `*_global_metrics.json`)

| Modelo | Script | Condición |
|---|---|---|
| `chronos2_C1_inst_global` | `15_chronos2_C1_inst_global.py` | C1_inst ★★ |
| `timesfm_C1_inst_global` | `16_timesfm_C1_inst_global.py` | C1_inst |
| `timegpt_C1_inst_global` | `17_timegpt_C1_inst_global.py` | C1_inst |

#### Europa (métricas en `rolling_metrics_europe.json` + `*_europe_metrics.json`)

Cada familia tiene C0, C1_inst, C1_mcp, C1_full (scripts 18–29):

| Familia | C0 | C1_inst | C1_mcp | C1_full |
|---|---|---|---|---|
| Chronos-2 | `18_` | `21_` | `22_` | `23_` |
| TimesFM | `19_` | `24_` | `25_` | `26_` ★★ |
| TimeGPT | `20_` | `27_` | `28_` | `29_` |

---

## 6. Resultados principales

### España — IPC (MASE scale: 1.4051 pp)

| Modelo | MAE h=1 | MAE h=3 | MAE h=6 | MAE h=12 | MASE h=1 | MASE h=12 |
|---|---|---|---|---|---|---|
| Naive lag-12 | 3.626 | — | — | 6.588 | 2.580 | 4.689 |
| ARIMA | **0.478** | **0.672** | **0.966** | **1.541** | 0.340 | **1.097** |
| SARIMA | 0.442 | 0.724 | 1.008 | 1.595 | **0.314** | 1.135 |
| AutoARIMA | 0.456 | 0.761 | 1.138 | 1.866 | 0.325 | 1.328 |
| N-BEATS | **0.359** | **0.670** | 1.195 | 1.895 | **0.255** | 1.348 |
| TimesFM C0 | 0.436 | 0.785 | 1.129 | 1.864 | 0.311 | 1.326 |
| TimesFM C1_inst ★ | **0.423** | 0.706 | 1.046 | 1.816 | **0.301** | 1.292 |
| Chronos-2 C0 | 0.520 | — | — | 1.990 | 0.370 | 1.416 |
| TimeGPT C0 | 0.549 | — | — | 2.010 | 0.391 | 1.430 |

**Veredicto España**:
- ARIMA fijo es el mejor modelo a h≥3 y h=12. Ningún foundation supera ARIMA en el largo plazo.
- N-BEATS gana a h=1 (MAE=0.359), pero empeora mucho a h=12 (MAE=1.895, peor que ARIMA).
- `timesfm_C1_inst` mejora sobre ARIMA a h=1 (-11.5%), pero pierde a h≥3 (+5-18%).
- C1_mcp (GDELT) **degrada** sistemáticamente todos los modelos (+33% a +57%).
- La señal más informativa es EPU Europe (corr nivel=+0.737 con IPC), pero es correlación espuria de nivel, no predice cambios mes a mes.

### Global — CPI Mundial (MASE scale: 1.1720 pp)

| Modelo | MAE h=1 | MAE h=6 | MAE h=12 | MASE h=12 |
|---|---|---|---|---|
| ARIMA | 0.191 | 0.682 | 1.544 | 1.317 |
| AutoARIMA | **0.179** | 0.567 | **1.329** | **1.134** |
| Chronos-2 C1_inst ★★ | 0.200 | **0.591** | **1.143** | **0.976** |
| TimesFM C1_inst | 0.269 | 0.712 | 1.284 | 1.096 |
| TimeGPT C1_inst | 0.415 | 1.180 | 2.114 | 1.803 |

**Veredicto Global**:
- `chronos2_C1_inst_global` es el **único modelo con MASE < 1.0 a h=12** (0.976), bate al naive.
- Chronos-2 supera a ARIMA a partir de h=3 (h=3: -4.2%, h=6: -13.3%, h=12: **-26.0%**).
- A h=1 ARIMA sigue siendo mejor (+5.1% penalización de Chronos-2).
- AutoARIMA también bate a ARIMA fijo en Global (h=1: -6.3%, h=12: -13.9%) — caso donde la reselección dinámica ayuda.
- TimeGPT degrada severamente con señales (+77% peor que ARIMA a h=12).
- Señales C1_inst usadas: GEPU, FEDFUNDS, GSCPI, Brent, DFR (BCE).

### Europa — HICP Eurozona (MASE scale: 1.4558 pp)

| Modelo | MAE h=1 | MAE h=6 | MAE h=12 | MASE h=12 |
|---|---|---|---|---|
| SARIMA | 0.413 | 1.226 | 2.411 | 1.656 |
| AutoARIMA | 0.376 | 1.147 | 2.510 | 1.724 |
| TimesFM C0 | **0.353** | 1.048 | 2.014 | 1.384 |
| TimesFM C1_full ★★ | 0.436 | **0.995** | **1.995** | **1.370** |
| Chronos-2 C0 | 0.512 | — | 2.300 | 1.580 |

**Veredicto Europa**:
- `timesfm_C1_full_europe` es el mejor modelo global: rompe la barrera MAE < 2.0 a h=12 (1.995).
- Supera a SARIMA en h≥6 (h=6: **-18.8%**, h=12: **-17.3%**); pero pierde a h=1 (+5.6%).
- C1_full = C1_inst + C1_mcp. La combinación de señales institucionales + BCE/GDELT es clave.
- C1_inst solo: mejora modesta (-2 a -4%). Es el MCP (BCE press) quien añade valor incremental.
- Chronos-2 con cualquier condición C1 no mejora sobre C0 en Europa.
- AutoARIMA peor que SARIMA fijo a h=12 (+4.1%).
- Señales C1_full usadas: DFR, Brent, TTF, EPU Europe, ESI Eurozona, breakeven 5y, BCE tone (MCP), GDELT tone MA6.

### AutoARIMA — Hallazgo metodológico transversal

| Serie | h=1 vs ref | h=12 vs ref | Referencia |
|---|---|---|---|
| España | -4.6% vs ARIMA, +3.3% vs SARIMA | +21.1% vs ARIMA, +17.0% vs SARIMA | ARIMA/SARIMA |
| Global | **-6.3%** vs ARIMA | **-13.9%** vs ARIMA | ARIMA |
| Europa | **-8.9%** vs SARIMA | +4.1% vs SARIMA | SARIMA |

- En **Global**, AutoARIMA mejora sobre ARIMA fijo (series con más cambio estructural).
- En **España**, AutoARIMA sistemáticamente peor que ARIMA/SARIMA a h≥3.
- En **Europa**, AutoARIMA compite en h corto pero pierde a h=12.
- Causa en España: el ARIMA fijo (3,1,0)(1,0,1,12) seleccionado sobre 2002-2020 captura la dinámica estacional mejor que los órdenes variables por ventana.

---

## 7. Convenciones de código

### Nombrado de scripts

```
NN_descripcion_serie.py
│  │            └─ spain | europe | global (o vacío si es solo España)
│  └─ descripción corta en minúsculas con guiones_bajos
└─ 2 dígitos (numeración secuencial dentro del módulo)
```

Ejemplos: `07_autoarima_spain.py`, `15_chronos2_C1_inst_global.py`, `04_backtesting_rolling_europe.py`

### Outputs de métricas

Cada script de evaluación guarda en `08_results/`:

```
{modelo}_{condicion}_{serie}_metrics.json      # métricas rolling
{modelo}_{condicion}_{serie}_predictions.parquet  # predicciones raw
{modelo}_{condicion}_{serie}_orders.json       # (AutoARIMA) órdenes por origen
```

Estructura interna de los JSON de métricas:
```json
{
  "nombre_modelo": {
    "h1":  {"MAE": 0.1234, "RMSE": 0.1567, "MASE": 0.1054, "n_evals": 47},
    "h3":  {"MAE": ..., ...},
    "h6":  {"MAE": ..., ...},
    "h12": {"MAE": ..., ...}
  }
}
```

### Archivos consolidados de métricas

| Archivo | Contenido |
|---|---|
| `metrics_summary_final.json` | Master de España: 24 modelos (naive → auto_arima) |
| `rolling_metrics.json` *(en 03_models_baseline/results/)* | Baselines Spain rolling |
| `rolling_metrics_global.json` | Baselines Global: naive, arima, arima111, arimax, auto_arima |
| `rolling_metrics_europe.json` | Baselines Europa: naive, sarima, sarimax, auto_arima |
| `rolling_metrics_C1_inst_global.json` | SARIMAX global con señales inst. |
| `deep_rolling_metrics_global.json` | Deep models Global |
| `deep_rolling_metrics_europe.json` | Deep models Europa |
| `diebold_mariano_results_final.json` | DM tests España (lista de objetos) |
| `diebold_mariano_results_europe.json` | DM tests Europa |

Los notebooks de evaluación cargan los datos así:
```python
spain_raw  = json.load(open(RESULTS / 'metrics_summary_final.json'))
global_raw = {}
for src in ['rolling_metrics_global.json', 'rolling_metrics_C1_inst_global.json',
            'deep_rolling_metrics_global.json']:
    global_raw.update(json.load(open(RESULTS / src)))
europe_raw = {}
for src in ['rolling_metrics_europe.json', 'deep_rolling_metrics_europe.json']:
    d = json.load(open(RESULTS / src))
    for k, v in d.items():
        europe_raw[f'{k}_europe'] = v  # IMPORTANTE: sufijo _europe
```

### Imports desde shared/

Los scripts de `tfg-forecasting/` importan `shared/` añadiendo la raíz al path:
```python
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.metrics import compute_mae, compute_mase
```

### Patrón de backtesting rolling (todos los módulos)

```python
ORIGINS = pd.date_range('2021-01-01', '2024-12-01', freq='MS')  # 48 orígenes
HORIZONS = [1, 3, 6, 12]

for origin in ORIGINS:
    y_train = full_series[full_series.index <= origin]
    for h in HORIZONS:
        y_true = full_series[origin + pd.DateOffset(months=1):
                             origin + pd.DateOffset(months=h)]
        # fit & predict...
        # guardar: (origin, h, y_pred, y_true)
```

---

## 8. Entorno y dependencias clave

```
Python  >= 3.11
pandas  >= 2.2
numpy   >= 1.26
statsmodels >= 0.14        # ARIMA/SARIMA/DM tests
pmdarima >= 2.1.1          # auto_arima (AutoARIMA dinámico)
torch   >= 2.2             # LSTM
neuralforecast             # N-BEATS, N-HiTS
nixtla  >= 0.5             # TimeGPT (API: requiere NIXTLA_API_KEY)
timesfm                    # TimesFM (instalación aparte, ver requirements.txt)
anthropic >= 0.25          # pipeline MCP
mcp >= 1.0                 # MCP server/client
scikit-learn               # StandardScaler para señales exógenas
```

Variables de entorno necesarias:
- `NIXTLA_API_KEY` — para TimeGPT
- `ANTHROPIC_API_KEY` — para el pipeline MCP (agente extractor)

---

## 9. Estado actual del proyecto (2026-04-27)

### Completado ✅

**ETL y features**:
- [x] Ingesta y limpieza de las 3 series objetivo
- [x] Pipeline MCP para España (GDELT + BCE press releases)
- [x] Pipeline MCP para Global (FOMC + BLS press)
- [x] Señales institucionales globales (GEPU, FEDFUNDS, VIX, GSCPI, etc.)
- [x] Señales Europa (ECB DFR, Brent, TTF, EPU, ESI, breakeven 5y, EUR/USD)
- [x] Feature engineering: lags (1,3,6,12), MAs (3,6), diffs, log-transforms
- [x] EDA: estacionariedad, estacionalidad, ACF/PACF, análisis de correlaciones

**Modelos baseline**:
- [x] ARIMA, SARIMA, SARIMAX — España, Global, Europa
- [x] AutoARIMA rolling — España, Global, Europa
- [x] Backtesting rolling 48 orígenes para todos los baselines

**Modelos deep**:
- [x] LSTM univariante — España, Europa, Global
- [x] N-BEATS — España, Europa, Global
- [x] N-HiTS — España, Europa, Global

**Foundation models** (30 configuraciones en total):
- [x] TimesFM: C0, C1_mcp, C1_inst, C1_macro — España
- [x] Chronos-2: C0, C1_mcp, C1_inst, C1_macro, C1_energy, C1_energy_only — España
- [x] TimeGPT: C0, C1_mcp, C1_inst, C1_macro, C1_energy, C1_energy_only — España
- [x] Chronos-2, TimesFM, TimeGPT: C1_inst — Global
- [x] Chronos-2, TimesFM, TimeGPT: C0, C1_inst, C1_mcp, C1_full — Europa

**Evaluación**:
- [x] DM tests — España y Europa
- [x] Análisis por régimen (pre-pandemia, COVID, shock energético, post-shock)
- [x] Ablación tipo de contexto (C0 vs C1_inst vs C1_mcp vs C1_full)
- [x] Comparativa transversal 3 series (`05_spain_vs_global_vs_europe.ipynb`)
- [x] AutoARIMA incorporado en todos los notebooks de evaluación (con HALLAZGO 5)

**Figuras** (en `08_results/`):
- [x] `fig_MAIN_summary.png` — evaluación España
- [x] `fig_MAIN_comparison.png` — comparativa 3 series
- [x] `fig_comp1_difficulty.png`, `fig_comp2_foundation_vs_stat.png`, `fig_comp3_families.png`, `fig_comp4_c1_effect.png`

### Pendiente / En curso ⏳

**Memoria del TFG (redacción)**:
- [ ] Capítulo 3: Metodología (rolling-origin, métricas, DM tests)
- [ ] Capítulo 4: Resultados por serie (tablas LaTeX desde `tabla_maestra_modelos.py`)
- [ ] Capítulo 5: Discusión transversal (usar hallazgos de `05_spain_vs_global_vs_europe.ipynb`)
- [ ] Capítulo 6: Conclusiones

**Plataforma web (`tfg-arquitectura/`)**:
- [ ] Backend FastAPI: endpoints de predicción en tiempo real
- [ ] Frontend React: dashboard de inflación con visualizaciones
- [ ] Integración con pipeline MCP en producción
- [ ] Despliegue (Docker Compose + nginx)

**Posibles mejoras experimentales**:
- [ ] Diebold-Mariano tests para Global (solo existe para España y Europa)
- [ ] Ablación condición de contexto para Global (solo se tiene C1_inst)
- [ ] Evaluación de AutoARIMA con ventana deslizante (no expandida)

---

## 10. Decisiones de diseño relevantes

1. **AutoARIMA vs ARIMA fijo**: el tutor recomendó probar AutoARIMA dinámico. El resultado es mixto: mejora en Global (-14% h=12), empeora en España (+21% h=12). El ARIMA fijo (seleccionado 1× sobre el histórico completo) es más robusto para series con dinámica estacional estable.

2. **EPU Europe en España**: la correlación nivel=+0.737 con IPC es espuria (ambas crecen en 2022). La correlación de diferencias es -0.09. El modelo captura el régimen de precios alto, no la variación mes a mes. Resultado: C1_inst levemente mejor en shock-2022, neutro en otras épocas.

3. **MCP en Europa vs España**: el pipeline MCP (BCE press releases) añade valor en Europa porque los comunicados del BCE son directamente relevantes para la política monetaria de la Eurozona. En España, el IPC tiene su propia dinámica de componentes domésticos que los comunicados del BCE no capturan directamente.

4. **TimeGPT con exógenas**: TimeGPT es el más frágil ante señales extremas. En el período del shock energético 2022, las señales de precio se dispararon y TimeGPT hizo carry-forward de correcciones. Siempre peor que C0 cuando se añaden señales.

5. **Normalización en Ridge**: el corrector Ridge sobre residuos del foundation model requiere StandardScaler. Sin él, la señal EPU (std~65) domina y produce una corrección casi constante (+1.11 pp en todos los orígenes).

6. **Índice de hicp_europe_index.parquet**: usa índice entero 0..275, no DatetimeIndex. Todos los scripts de Europa deben hacer `df = df.set_index('date')` tras cargar.
