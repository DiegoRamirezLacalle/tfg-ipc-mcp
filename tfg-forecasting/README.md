# tfg-forecasting — Inflation Forecasting Pipeline

Data-science half of the TFG: ETL → EDA → models → evaluation for three inflation
series (Spain CPI, Global CPI, Europe HICP), comparing statistical baselines,
deep models and foundation models (Chronos-2, TimesFM, TimeGPT) with and without
exogenous signals (institutional + MCP news signals).

For results, research questions and the full experimental design see the
[root README](../README.md) and [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)
(the latter is the canonical reference: data schemas, MASE scales, model keys).

## Pipeline layout

Folders run in numeric order; each stage writes artifacts the next one reads.

```
01_etl/                 13 scripts — ingest targets (INE, World Bank, Eurostat) and
                        exogenous series (ECB rates, FEDFUNDS, EPU, VIX, Brent, TTF…)
                        → data/processed/features_c1*.parquet
02_eda/                 13 notebooks — stationarity, seasonality, ACF/PACF, regimes
03_models_baseline/     ARIMA / SARIMA / SARIMAX / AutoARIMA (Spain, Global, Europe)
04_models_deep/         LSTM / N-BEATS / N-HiTS via NeuralForecast
05_mcp_pipeline/        Spain news pipeline: fetch → MCP server + Claude extractor
05_mcp_pipeline_global/ Global pipeline (FOMC, ECB, BLS press releases)
06_models_foundation/   Chronos-2 / TimesFM / TimeGPT × conditions (C0, C1_inst,
                        C1_mcp, C1_full, C1_macro, C1_energy) × 3 series
07_evaluation/          comparison notebooks + Diebold-Mariano tests
08_results/             *_metrics.json + *_predictions.parquet per model + figures
09_future_work/         exploratory extensions (MIDAS, multi-agent revisers)
tests/                  pytest suite + artifact/leakage integrity checks
```

## Setup

```bash
# from the repo root
pip install -r tfg-forecasting/requirements.txt
cp .env.example .env   # NIXTLA_API_KEY, ANTHROPIC_API_KEY, MONGO_URI…
```

## Running

Every script is standalone (no orchestrator): it bootstraps `sys.path` to import
`shared/` and writes its outputs to `08_results/`.

```bash
python tfg-forecasting/01_etl/01_ingest_cpi_global.py          # ETL
python tfg-forecasting/03_models_baseline/04_backtesting_rolling.py
python tfg-forecasting/06_models_foundation/15_chronos2_C1_inst_global.py
python tfg-forecasting/07_evaluation/build_metrics_summary_final.py
```

Tests:

```bash
pytest tfg-forecasting/tests/test_metrics.py            # unit tests (no data needed)
python tfg-forecasting/tests/check_artifacts_and_leakage.py   # needs generated artifacts
```

## Methodological invariants

Do not change these without re-running every affected backtest
(details in [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)):

- **Anti-leakage**: all exogenous features enter with `shift(+1)` — the signal
  from `t-1` predicts `t`.
- **MASE scale is frozen** on 2002–2020 with the naive lag-12 forecast
  (Spain 1.4051 · Global 1.1720 · Europe 1.4558 pp). Each script recomputes it
  locally on purpose, to keep results refactor-proof.
- **StandardScaler before Ridge** in every TimesFM C1 correction — without it
  the EPU scale (σ≈65) dominates `diff(target)` (σ≈0.44) and MAE explodes.
- **Splits by date** (`shared/constants.py`), never by positional index.
- **Expanding window**: each rolling origin trains on `[2002-01, t]` and
  predicts `[t+1, t+h]`, h ∈ {1, 3, 6, 12}, 48 monthly origins over 2021–2024.
