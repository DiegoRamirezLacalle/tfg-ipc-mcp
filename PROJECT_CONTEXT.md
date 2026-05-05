# PROJECT_CONTEXT.md
> External memory for Claude Code sessions without prior context.
> Last updated: 2026-05-04

---

## 1. Project Summary

**Title**: *Evaluation of Foundation Time-Series Models for Inflation Forecasting with MCP Signals*

**Central hypothesis**: Do foundation time-series models (Chronos-2, TimesFM, TimeGPT) improve inflation forecasting over classical statistical baselines (ARIMA/SARIMA)? Do exogenous signals built with an MCP pipeline add predictive value?

**Author**: Diego Ramirez · `analacallealvarez@gmail.com`

### Two parts of the repository

| Folder | What it is |
|--------|------------|
| `tfg-forecasting/` | Scientific research: data, models, evaluation, results |
| `tfg-arquitectura/` | Web platform: FastAPI backend + React frontend + Docker |

---

## 2. Repository Architecture

```
tfg-ipc-mcp/
├── PROJECT_CONTEXT.md          ← this file
├── README.md                   ← updated project overview
├── pyproject.toml              ← monorepo Python dependencies
├── docker-compose.yml          ← web services orchestration
├── shared/                     ← utilities shared across scripts
│   ├── constants.py            ← HORIZONS=[1,3,6,12], data paths, etc.
│   ├── data_utils.py           ← parquet loading helpers
│   ├── metrics.py              ← MAE, RMSE, MASE, naive_scale
│   └── logger.py               ← standard logger
│
├── tfg-forecasting/
│   ├── data/
│   │   ├── raw/                ← downloaded CSVs (INE, ECB, FRED, etc.)
│   │   └── processed/          ← parquets ready for models (see §4)
│   │
│   ├── 01_etl/                 ← 13 ingestion & feature-engineering scripts
│   ├── 02_eda/                 ← 13 notebooks (visual, stationarity, seasonality, ACF/PACF, regimes)
│   ├── 03_models_baseline/     ← ARIMA, SARIMA, SARIMAX, AutoARIMA
│   ├── 04_models_deep/         ← LSTM, N-BEATS, N-HiTS (NeuralForecast)
│   ├── 05_mcp_pipeline/        ← Spain MCP pipeline (GDELT + ECB press releases)
│   ├── 05_mcp_pipeline_global/ ← Global MCP pipeline (FOMC + BLS press releases)
│   ├── 06_models_foundation/   ← 29 scripts — Chronos-2, TimesFM, TimeGPT (C0/C1, 3 series)
│   ├── 07_evaluation/          ← evaluation notebooks + Diebold-Mariano tests
│   ├── 08_results/             ← JSON metrics, Parquet predictions, PNG figures
│   ├── configs/                ← YAML model configurations
│   └── lightning_logs/         ← PyTorch Lightning logs (~263 versions)
│
└── tfg-arquitectura/
    ├── backend/                ← FastAPI (main.py, config.py, app/)
    ├── frontend/               ← React (package.json, src/)
    ├── gateway/                ← nginx / reverse proxy
    ├── db/                     ← database schemas
    └── infra/                  ← IaC (Terraform/Docker)
```

### Module 01_etl — Ingestion scripts

| Script | What it does |
|--------|-------------|
| `01_ingest_cpi_global.py` | Downloads World Bank CPI → `cpi_global_monthly.parquet` |
| `03_ingest_ecb_rates.py` | ECB rates (DFR, MRR) → `ecb_rates_monthly.parquet` |
| `05_clean_and_align.py` | Temporal alignment to monthly frequency |
| `06_feature_engineering_exog.py` | Builds lags/MAs/diffs → `features_exog.parquet` |
| `07_ingest_energy_prices.py` | Brent + TTF gas → `energy_prices_monthly.parquet` |
| `08_merge_energy_features.py` | Merges energy into features_c1 |
| `09_ingest_institutional_signals.py` | EPU Europe + ECB signals (Spain/Europe) |
| `10_ingest_institutional_signals_global.py` | GEPU, GSCPI, VIX, DXY, USG10Y, FEDFUNDS → `institutional_signals_monthly.parquet` |
| `10_merge_institutional_features.py` | Merges institutional into features_c1 |
| `11_ingest_hicp_europe.py` | HICP update |
| `12_ingest_europe_signals.py` | ESI Eurozone, 5y breakeven, EUR/USD → `europe_signals_monthly.parquet` |
| `13_build_features_c1_europe.py` | Assembles `features_c1_europe.parquet` |

### Module 07_evaluation — Evaluation notebooks

| Notebook | Content |
|----------|---------|
| `02_compare_all_models.ipynb` | Full Spain evaluation: ranking, MAE profiles, ΔC1 heatmap, DM tests |
| `03_evaluation_global.ipynb` | Global CPI evaluation — MASE/MAE profiles, family comparison, AutoARIMA |
| `03_regime_analysis.ipynb` | Regime analysis (pre-pandemic, COVID, energy shock, post-shock) |
| `04_ablation_context_type.ipynb` | Ablation C0 vs C1_inst vs C1_mcp vs C1_full |
| `04_evaluation_europe.ipynb` | Europe HICP evaluation — MASE/MAE profiles, C1 ablation, AutoARIMA |
| `05_spain_vs_global_vs_europe.ipynb` | Cross-series synthesis — main thesis figure |
| `01_diebold_mariano_tests.py` | DM tests Spain → `diebold_mariano_results_final.json` |
| `05_diebold_mariano_europe.py` | DM tests Europe → `diebold_mariano_results_europe.json` |
| `build_metrics_summary_final.py` | Builds `metrics_summary_final.json` (Spain master) |
| `tabla_maestra_modelos.py` | LaTeX table of all models |

---

## 3. Critical Methodological Decisions

### Temporal splits

```
Initial training (train): 2002-01 to 2020-12  (228 obs)
Implicit validation: inside rolling backtesting window
Test (rolling-origin):   2021-01 to 2024-12  (48 origins)
```

Split is done **by date**, not by numeric index. The initial training period is used to:
- Compute the MASE scale (MAE of naive lag-12 over 2002–2020)
- Select fixed ARIMA/SARIMA orders (once, no reselection)
- Train deep learning models on the 24-month window

### Expanding rolling-origin backtesting

```
Origins : 48 (monthly from 2021-01-01 to 2024-12-01)
Horizons: h = 1, 3, 6, 12 months
Window  : EXPANDING (each origin adds 1 month to history)
```

For origin `t`, the model trains on `[2002-01, t]` and predicts `[t+1, t+h]`.
Foundation models receive the full series up to `t` without retraining.

### Metrics

```python
MAE   = mean(|y_pred - y_true|)
RMSE  = sqrt(mean((y_pred - y_true)^2))
MASE  = MAE / naive_scale
naive_scale = mean(|y_t - y_{t-12}|)  # computed over 2002-2020

# MASE scale by series:
# Spain  : 1.4051 pp
# Global : 1.1720 pp
# Europe : 1.4558 pp
```

MASE < 1 → the model beats the seasonal naive lag-12.

### Diebold-Mariano (DM) tests

- Two-sided test, Harvey-Leybourne-Newbold (HLN) correction
- Null: both models have the same mean squared error
- Applied per horizon h and per sub-period
- Sub-periods: full-2021, Shock-2022 (2022-01 to 2022-12), Post-shock (2023–2024)
- Results in `08_results/diebold_mariano_results_final.json` and `*_europe.json`

### Leakage prevention — shift +1 on exogenous signals

**CRITICAL**: all exogenous signals are shifted +1 month before being passed to the model.

```python
# In all foundation scripts with C1:
exog_shifted = exog_df.shift(1)  # signal at t-1 predicts t
```

Without this shift, future information leaks into the model. This bug was detected in earlier sessions with TimeGPT.

### Normalization of exogenous signals

**CRITICAL**: signals have very different scales (EPU std~65, diff(HICP) std~0.44).
Ridge correction **requires** StandardScaler before fitting:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# ridge.fit(X_train_scaled, residuals_train)
```

Without normalization, Ridge produces an almost-constant correction of ~+1.11 pp (TimesFM MAE +534% vs C0 in the diagnostic run that caught this).

---

## 4. Datasets and Exogenous Signals

### Target series

| File | Variable | Period | Obs | Source |
|------|----------|--------|-----|--------|
| `ipc_spain_index.parquet` | `indice_general` (+ 13 sub-indices) | 2002-01 → 2026-01 | 289 | INE |
| `hicp_europe_index.parquet` | `hicp_index` | 2002-01 → 2024-12 | 276 | ECB SDW |
| `cpi_global_monthly.parquet` | `cpi_global_rate` | 2002-01 → 2024-12 | 276 | World Bank |

**Important**: `hicp_europe_index.parquet` has an integer index (0..275) and `date` as a regular column (not a DatetimeIndex). All Europe scripts must do `df = df.set_index('date')` after loading.

### Exogenous signals — Spain (`features_c1.parquet`, 282 obs × 34 cols)

| Group | Key columns |
|-------|------------|
| CPI target | `indice_general` |
| ECB rates | `dfr`, `mrr`, `dfr_diff`, `dfr_lag3`, `dfr_lag6`, `dfr_lag12` |
| GDELT (ECB news tone) | `gdelt_avg_tone`, `gdelt_goldstein`, `gdelt_n_articles`, `gdelt_tone_ma3`, `gdelt_tone_ma6` |
| ECB signals (MCP) | `bce_shock_score`, `bce_uncertainty`, `bce_tone`, `bce_tone_numeric`, `bce_cumstance` |
| INE signals (MCP) | `ine_surprise_score`, `ine_topic`, `ine_inflacion`, `dominant_topic` |
| Energy | `brent_log`, `brent_ret`, `brent_ma3`, `brent_lag1`, `ttf_log`, `ttf_ret`, `ttf_ma3`, `ttf_lag1` |
| EPU Europe | `epu_europe_log`, `epu_europe_ma3`, `epu_europe_lag1` |
| MCP availability | `signal_available` (0/1 per month) |

### Exogenous signals — Global (`features_c1_global_institutional.parquet`, 276 obs × 31 cols)

Global institutional signals, all with `_ma3`, `_lag1`, `_diff` suffixes:
`gepu` (Global EPU), `imf_comm` (IMF commodity), `dxy` (US Dollar Index),
`vix`, `usg10y` (US Treasury 10y), `fedfunds` (Fed Funds Rate),
`gscpi` (Global Supply Chain Pressure Index), `gpr` (Geopolitical Risk),
`brent_log`, `dfr` (ECB deposit rate). Plus `cpi_global_rate` (target).

Final selected columns in `08_results/c1_global_inst_selected_cols.json`.

### Exogenous signals — Europe (`features_c1_europe.parquet`, 276 obs × 15 cols)

`hicp_index`, `dfr`, `dfr_ma3`, `mrr`, `brent_ma3`, `ttf_ma3`,
`epu_europe_ma3`, `esi_eurozone` (Economic Sentiment Indicator), `breakeven_5y_lag1`,
`eurusd_ma3`, `bce_shock_score`, `bce_tone_numeric`, `bce_cumstance`,
`gdelt_tone_ma6`, `signal_available`.

---

## 5. Models Implemented

### Experimental conditions

| Condition | Description |
|-----------|-------------|
| **C0** | Univariate — target series only, no exogenous signals |
| **C1_inst** | + Institutional signals (ECB rates, EPU, energy, macro indicators) |
| **C1_mcp** | + MCP text signals (GDELT tone, ECB press releases processed by Claude) |
| **C1_full** | C1_inst + C1_mcp combined (Europe only) |
| **C1_energy** | Energy signals only (Brent, TTF) |
| **C1_macro** | Mixed macro signals (Brent + TTF + EPU) |

### Statistical baselines (`03_models_baseline/`)

| Model | Script | Series | Notes |
|-------|--------|--------|-------|
| Naive lag-12 | implicit in metrics | all | reference benchmark |
| ARIMA | `01_arima_auto{_europe,_global}.py` | all | fixed orders selected by auto_arima once |
| ARIMA(1,1,1) | — | Global | fixed-order variant for comparison |
| SARIMA | `02_sarima{_europe,_global}.py` | Spain/Europe | includes seasonal component (1,0,1,12) |
| SARIMAX | `03_sarimax{_europe,_global}.py` | all | SARIMA + institutional exogenous |
| AutoARIMA | `07_autoarima_{spain,europe,global}.py` | all | order reselection at each origin |

**AutoARIMA**: uses `pmdarima.auto_arima(seasonal=True, m=12, stepwise=True, information_criterion='aic', max_p=3, max_q=3, max_P=2, max_Q=2)`. Refits orders at each of the 48 rolling origins.

### Deep learning (`04_models_deep/`)

| Model | Script | Series | Library |
|-------|--------|--------|---------|
| LSTM univariate | `01_lstm_univariate{_europe}.py` | Spain/Europe | PyTorch |
| LSTM global | `01_lstm_global.py` | Global | PyTorch |
| N-BEATS | `02_nbeats{_europe,_global}.py` | all | NeuralForecast |
| N-HiTS | `03_nhits{_europe,_global}.py` | all | NeuralForecast |

Shared helpers: `_helpers.py`, `_helpers_europe.py`, `_helpers_global.py`.
Logs in `lightning_logs/` (~263 versions).

### Foundation models (`06_models_foundation/`)

#### Spain (metrics in `metrics_summary_final.json`)

| Model key | Script | Condition |
|-----------|--------|-----------|
| `timesfm_C0` | `01_timesfm_C0.py` | C0 |
| `timesfm_C1` | `02_timesfm_C1.py` | C1_mcp (GDELT) |
| `timesfm_C1_inst` | `11_timesfm_C1_inst.py` | C1_inst (EPU Europe) ★ |
| `timesfm_C1_macro` | `12_timesfm_C1_macro.py` | C1_macro |
| `chronos2_C0` | `03_chronos2_C0.py` | C0 |
| `chronos2_C1` | `04_chronos2_C1.py` | C1_mcp |
| `chronos2_C1_inst` | `09_chronos2_C1_inst.py` | C1_inst |
| `chronos2_C1_macro` | `10_chronos2_C1_macro.py` | C1_macro |
| `chronos2_C1_energy` | `05_chronos2_C1_energy.py` | C1_energy |
| `chronos2_C1_energy_only` | `08_chronos2_C1_energy_only.py` | energy only |
| `timegpt_C0` | `05_timegpt_C0.py` | C0 |
| `timegpt_C1` | `06_timegpt_C1.py` | C1_mcp |
| `timegpt_C1_inst` | `13_timegpt_C1_inst.py` | C1_inst |
| `timegpt_C1_macro` | `14_timegpt_C1_macro.py` | C1_macro |
| `timegpt_C1_energy` | `07_timegpt_C1_energy.py` | C1_energy |
| `timegpt_C1_energy_only` | `07_timegpt_C1_energy_only.py` | energy only |

#### Global (metrics in `rolling_metrics_global.json` + `*_global_metrics.json`)

| Model key | Script | Condition |
|-----------|--------|-----------|
| `chronos2_C1_inst_global` | `15_chronos2_C1_inst_global.py` | C1_inst ★★ |
| `timesfm_C1_inst_global` | `16_timesfm_C1_inst_global.py` | C1_inst |
| `timegpt_C1_inst_global` | `17_timegpt_C1_inst_global.py` | C1_inst |

#### Europe (metrics in `rolling_metrics_europe.json` + `*_europe_metrics.json`)

Each family has C0, C1_inst, C1_mcp, C1_full (scripts 18–29):

| Family | C0 | C1_inst | C1_mcp | C1_full |
|--------|----|---------|--------|---------|
| Chronos-2 | `18_` | `21_` | `22_` | `23_` |
| TimesFM | `19_` | `24_` | `25_` | `26_` ★★ |
| TimeGPT | `20_` | `27_` | `28_` | `29_` |

---

## 6. Final Results

### Spain — CPI (MASE scale: 1.4051 pp)

| Model | MAE h=1 | MAE h=3 | MAE h=6 | MAE h=12 | MASE h=1 | MASE h=12 |
|-------|---------|---------|---------|----------|----------|----------|
| Naive lag-12 | 3.626 | — | — | 6.588 | 2.580 | 4.689 |
| ARIMA | **0.478** | **0.672** | **0.966** | **1.541** | 0.340 | **1.097** |
| SARIMA | 0.442 | 0.724 | 1.008 | 1.595 | **0.314** | 1.135 |
| AutoARIMA | 0.456 | 0.761 | 1.138 | 1.866 | 0.325 | 1.328 |
| N-BEATS | **0.359** | **0.670** | 1.195 | 1.895 | **0.255** | 1.348 |
| TimesFM C0 | 0.436 | 0.785 | 1.129 | 1.864 | 0.311 | 1.326 |
| TimesFM C1_inst ★ | **0.423** | 0.706 | 1.046 | 1.816 | **0.301** | 1.292 |
| Chronos-2 C0 | 0.520 | — | — | 1.990 | 0.370 | 1.416 |
| TimeGPT C0 | 0.549 | — | — | 2.010 | 0.391 | 1.430 |

**Spain verdict**:
- Fixed ARIMA is the best model at h≥3 and h=12. No foundation model beats ARIMA at long horizons.
- N-BEATS wins at h=1 (MAE=0.359), but deteriorates badly at h=12 (MAE=1.895, worse than ARIMA).
- `timesfm_C1_inst` improves over ARIMA at h=1 (−11.5%), but loses at h≥3 (+5–18%).
- C1_mcp (GDELT) **systematically degrades** all models (+33% to +57%).
- The most informative signal is EPU Europe (level correlation +0.737 with CPI), but this is spurious level correlation — it does not predict month-to-month changes.

### Global — World CPI (MASE scale: 1.1720 pp)

| Model | MAE h=1 | MAE h=6 | MAE h=12 | MASE h=12 |
|-------|---------|---------|----------|----------|
| ARIMA | 0.191 | 0.682 | 1.544 | 1.317 |
| AutoARIMA | **0.179** | 0.567 | **1.329** | **1.134** |
| Chronos-2 C1_inst ★★ | 0.200 | **0.591** | **1.143** | **0.976** |
| TimesFM C1_inst | 0.269 | 0.712 | 1.284 | 1.096 |
| TimeGPT C1_inst | 0.415 | 1.180 | 2.114 | 1.803 |

**Global verdict**:
- `chronos2_C1_inst_global` is the **only model with MASE < 1.0 at h=12** (0.976) — beats the seasonal naive.
- Chronos-2 beats ARIMA from h=3 onwards (h=3: −4.2%, h=6: −13.3%, h=12: **−26.0%**).
- At h=1 ARIMA is still better (+5.1% penalty for Chronos-2).
- AutoARIMA also beats fixed ARIMA in Global (h=1: −6.3%, h=12: −13.9%) — the one case where dynamic order selection helps.
- TimeGPT severely degrades with signals (+77% worse than ARIMA at h=12).
- C1_inst signals used: GEPU, FEDFUNDS, GSCPI, Brent, DFR (ECB).

### Europe — HICP Eurozone (MASE scale: 1.4558 pp)

| Model | MAE h=1 | MAE h=6 | MAE h=12 | MASE h=12 |
|-------|---------|---------|----------|----------|
| SARIMA | 0.413 | 1.226 | 2.411 | 1.656 |
| AutoARIMA | 0.376 | 1.147 | 2.510 | 1.724 |
| TimesFM C0 | **0.353** | 1.048 | 2.014 | 1.384 |
| TimesFM C1_full ★★ | 0.436 | **0.995** | **1.995** | **1.370** |
| Chronos-2 C0 | 0.512 | — | 2.300 | 1.580 |

**Europe verdict**:
- `timesfm_C1_full_europe` is the overall best model: first to break the MAE < 2.0 barrier at h=12 (1.995).
- Beats SARIMA at h≥6 (h=6: **−18.8%**, h=12: **−17.3%**); loses at h=1 (+5.6%).
- C1_full = C1_inst + C1_mcp. The combination of institutional signals + ECB/GDELT text is key.
- C1_inst alone: modest improvement (−2 to −4%). It is the MCP (ECB press) that adds incremental value.
- Chronos-2 with any C1 condition does not improve over C0 in Europe.
- AutoARIMA worse than fixed SARIMA at h=12 (+4.1%).
- C1_full signals: DFR, Brent, TTF, EPU Europe, ESI Eurozone, 5y breakeven, ECB tone (MCP), GDELT tone MA6.

### Cross-series summary (h=12)

| Series | Best statistical | MASE h=12 | Best foundation | MASE h=12 | C1 effect |
|--------|-----------------|-----------|-----------------|-----------|-----------|
| Spain CPI | ARIMA | 1.097 | TimesFM C1_inst | 1.292 | ~0% (neutral / slightly worse) |
| Global CPI | ARIMA | 1.317 | Chronos-2 C1_inst ★★ | **0.976** | −26% |
| Europe HICP | SARIMA | 1.656 | TimesFM C1_full ★★ | **1.370** | −17% |

### AutoARIMA — Cross-series methodological finding

| Series | h=1 vs ref | h=12 vs ref | Reference |
|--------|------------|-------------|-----------|
| Spain | −4.6% vs ARIMA | **+21.1%** vs ARIMA | ARIMA |
| Global | **−6.3%** vs ARIMA | **−13.9%** vs ARIMA | ARIMA |
| Europe | **−8.9%** vs SARIMA | +4.1% vs SARIMA | SARIMA |

- In **Global**, AutoARIMA improves on fixed ARIMA (series with more structural change).
- In **Spain**, AutoARIMA is systematically worse than ARIMA/SARIMA at h≥3.
- In **Europe**, AutoARIMA competes at short horizons but loses at h=12.
- Root cause for Spain: the fixed ARIMA (3,1,0)(1,0,1,12) selected on 2002–2020 captures seasonal dynamics better than orders that vary by rolling window.

---

## 7. Code Conventions

### Script naming

```
NN_description_series.py
│  │            └─ spain | europe | global (or empty for Spain-only)
│  └─ short description in lowercase with underscores
└─ 2-digit prefix (sequential within module)
```

Examples: `07_autoarima_spain.py`, `15_chronos2_C1_inst_global.py`, `04_backtesting_rolling_europe.py`

### Metric output format

Each evaluation script saves to `08_results/`:

```
{model}_{condition}_{series}_metrics.json        # rolling metrics
{model}_{condition}_{series}_predictions.parquet # raw predictions
{model}_{condition}_{series}_orders.json         # (AutoARIMA) orders per origin
```

Internal structure of metrics JSON:
```json
{
  "model_name": {
    "h1":  {"MAE": 0.1234, "RMSE": 0.1567, "MASE": 0.1054, "n_evals": 47},
    "h3":  {"MAE": ..., ...},
    "h6":  {"MAE": ..., ...},
    "h12": {"MAE": ..., ...}
  }
}
```

### Consolidated metrics files

| File | Content |
|------|---------|
| `metrics_summary_final.json` | Spain master: 24 models (naive → auto_arima) |
| `rolling_metrics_global.json` | Global baselines: naive, arima, arima111, arimax, auto_arima |
| `rolling_metrics_europe.json` | Europe baselines: naive, sarima, sarimax, auto_arima |
| `rolling_metrics_C1_inst_global.json` | SARIMAX global with institutional signals |
| `deep_rolling_metrics_global.json` | Deep models — Global |
| `deep_rolling_metrics_europe.json` | Deep models — Europe |
| `diebold_mariano_results_final.json` | DM tests Spain (list of objects) |
| `diebold_mariano_results_europe.json` | DM tests Europe |

How the evaluation notebooks load metrics:
```python
spain_raw  = json.load(open(RESULTS / 'metrics_summary_final.json'))

global_raw = {}
for src in ['rolling_metrics_global.json', 'rolling_metrics_C1_inst_global.json',
            'deep_rolling_metrics_global.json']:
    global_raw.update(json.load(open(RESULTS / src)))
# + individual *_global_metrics.json files for each foundation model

europe_raw = {}
for src in ['rolling_metrics_europe.json', 'deep_rolling_metrics_europe.json']:
    d = json.load(open(RESULTS / src))
    for k, v in d.items():
        europe_raw[f'{k}_europe'] = v   # IMPORTANT: _europe suffix added here
# + individual *_europe_metrics.json (no suffix needed — already in name)
```

### Rolling backtesting pattern (all modules)

```python
ORIGINS  = pd.date_range('2021-01-01', '2024-12-01', freq='MS')  # 48 origins
HORIZONS = [1, 3, 6, 12]

for origin in ORIGINS:
    y_train = full_series[full_series.index <= origin]
    for h in HORIZONS:
        y_true = full_series[origin + pd.DateOffset(months=1):
                             origin + pd.DateOffset(months=h)]
        # fit & predict...
        # save: (origin, h, y_pred, y_true)
```

### Importing from shared/

Scripts in `tfg-forecasting/` import `shared/` by adding root to path:
```python
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.metrics import compute_mae, compute_mase
```

---

## 8. Environment and Key Dependencies

```
Python       >= 3.11
pandas       >= 2.2
numpy        >= 1.26
statsmodels  >= 0.14        # ARIMA/SARIMA/DM tests
pmdarima     >= 2.1.1       # auto_arima (dynamic AutoARIMA)
torch        >= 2.2         # LSTM
neuralforecast              # N-BEATS, N-HiTS
nixtla       >= 0.5         # TimeGPT (API — requires NIXTLA_API_KEY)
timesfm                     # TimesFM (separate install, see requirements.txt)
anthropic    >= 0.25        # MCP pipeline
mcp          >= 1.0         # MCP server/client
scikit-learn                # StandardScaler for exogenous signals
```

Required environment variables:
- `NIXTLA_API_KEY` — for TimeGPT
- `ANTHROPIC_API_KEY` — for the MCP pipeline (Claude extractor agent)

---

## 9. Project Status (2026-05-04)

### Completed ✅

**ETL and features**:
- [x] Ingestion and cleaning of all 3 target series
- [x] MCP pipeline for Spain (GDELT + ECB press releases)
- [x] MCP pipeline for Global (FOMC + BLS press)
- [x] Global institutional signals (GEPU, FEDFUNDS, VIX, GSCPI, etc.)
- [x] Europe signals (ECB DFR, Brent, TTF, EPU, ESI, 5y breakeven, EUR/USD)
- [x] Feature engineering: lags (1,3,6,12), MAs (3,6), diffs, log-transforms
- [x] EDA: stationarity, seasonality, ACF/PACF, correlation analysis — all 3 series

**Statistical models**:
- [x] ARIMA, SARIMA, SARIMAX — Spain, Global, Europe
- [x] AutoARIMA rolling — Spain, Global, Europe
- [x] Rolling backtesting (48 origins) for all baselines

**Deep learning**:
- [x] LSTM univariate — Spain, Europe, Global
- [x] N-BEATS — Spain, Europe, Global
- [x] N-HiTS — Spain, Europe, Global

**Foundation models** (29 configurations total):
- [x] TimesFM: C0, C1_mcp, C1_inst, C1_macro — Spain
- [x] Chronos-2: C0, C1_mcp, C1_inst, C1_macro, C1_energy, C1_energy_only — Spain
- [x] TimeGPT: C0, C1_mcp, C1_inst, C1_macro, C1_energy, C1_energy_only — Spain
- [x] Chronos-2, TimesFM, TimeGPT: C1_inst — Global
- [x] Chronos-2, TimesFM, TimeGPT: C0, C1_inst, C1_mcp, C1_full — Europe

**Evaluation**:
- [x] DM tests — Spain and Europe
- [x] Regime analysis (pre-pandemic, COVID, energy shock, post-shock)
- [x] Context-type ablation (C0 vs C1_inst vs C1_mcp vs C1_full)
- [x] Cross-series comparison of all 3 series (`05_spain_vs_global_vs_europe.ipynb`)
- [x] AutoARIMA incorporated in all evaluation notebooks (Finding 5)

**Code refactoring (branch `refactor-clean`)**:
- [x] All Python scripts in `shared/`, `01_etl/`, `03_models_baseline/`, `04_models_deep/`, `05_mcp_pipeline{,_global}/`, `06_models_foundation/` translated to English and cleaned
- [x] All 13 notebooks in `02_eda/` translated and cleaned
- [x] All 4 evaluation notebooks in `07_evaluation/` translated and cleaned
- [x] README.md fully rewritten with final data and results

**Output figures** (`08_results/`):
- [x] `fig_MAIN_comparison.png` — 2×3 cross-series panel (main thesis figure)
- [x] `fig_MAIN_summary.png` — Spain evaluation summary
- [x] `fig_comp1_difficulty.png` — forecast difficulty by series
- [x] `fig_comp2_foundation_vs_stat.png` — foundation vs statistical MAE profiles
- [x] `fig_comp3_families.png` — family comparison (Chronos-2 / TimesFM / TimeGPT)
- [x] `fig_comp4_c1_effect.png` — C1 signal effect heatmap (Δ MAE %)

### Pending / In progress ⏳

**Thesis document (writing)**:
- [ ] Chapter 3: Methodology (rolling-origin, metrics, DM tests)
- [ ] Chapter 4: Results by series (LaTeX tables from `tabla_maestra_modelos.py`)
- [ ] Chapter 5: Cross-series discussion (use findings from `05_spain_vs_global_vs_europe.ipynb`)
- [ ] Chapter 6: Conclusions

**Web platform (`tfg-arquitectura/`)**:
- [ ] FastAPI backend: real-time forecasting endpoints
- [ ] React frontend: inflation dashboard with visualizations
- [ ] MCP pipeline integration in production
- [ ] Deployment (Docker Compose + nginx)

**Possible experimental additions**:
- [ ] Diebold-Mariano tests for Global (only exists for Spain and Europe)
- [ ] Context-type ablation for Global (currently only C1_inst exists)
- [ ] AutoARIMA evaluation with sliding window (not expanding)

---

## 10. Key Design Decisions

1. **AutoARIMA vs fixed ARIMA**: the thesis supervisor recommended testing dynamic AutoARIMA. The result is mixed: it improves for Global (−14% h=12), worsens for Spain (+21% h=12). The fixed ARIMA model (selected once on the full historical sample) is more robust for series with stable seasonal dynamics.

2. **EPU Europe in Spain**: the level correlation +0.737 with CPI is spurious (both grew in 2022). The correlation of first differences is −0.09. The model captures the high-price regime, not month-to-month variation. Result: C1_inst slightly better during the 2022 shock, neutral otherwise.

3. **MCP in Europe vs Spain**: the MCP pipeline (ECB press releases) adds value in Europe because ECB communications are directly relevant to Eurozone monetary policy. In Spain, domestic CPI dynamics (energy, housing, food components) are not well captured by ECB communiqués alone.

4. **TimeGPT with exogenous signals**: TimeGPT is the most fragile model with extreme signals. During the 2022 energy shock, energy price signals spiked and TimeGPT produced carry-forward corrections that compounded into large errors. Consistently worse than C0 when signals are added across all series.

5. **Ridge normalization**: the Ridge corrector applied to foundation-model residuals requires StandardScaler. Without it, EPU (std~65) dominates and produces a near-constant correction (+1.11 pp at every rolling origin), inflating TimesFM MAE by +534% compared to C0.

6. **HICP `hicp_europe_index.parquet` index**: uses integer index 0..275, NOT a DatetimeIndex. Every Europe script must call `df = df.set_index('date')` immediately after loading.

7. **`'europa'` vs `'europe'` keys**: internally all Python dicts and series_list use `'europe'` (English). Earlier versions had a runtime bug where `'europa'` keys in BEST/FAMILIES/STAT_REF mismatched `series_list = ['spain', 'global', 'europe']`. This is fully fixed across all notebooks.
