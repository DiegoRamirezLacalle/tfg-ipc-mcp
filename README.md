# TFG IPC-MCP — Inflation Forecasting with Foundation Models and MCP Signals

Dual engineering thesis: time-series forecasting of inflation using foundation models augmented with semantic context via MCP (Model Context Protocol).

## Research Question

> Do foundation time-series models (Chronos-2, TimesFM, TimeGPT) improve inflation forecasting over classical statistical models? Do institutional/MCP signals add value, and how does this depend on the data context?

**Three series evaluated**: Spain CPI (INE), Global CPI (IMF), Europe HICP (Eurostat)  
**Test period**: 2021–2024 rolling-origin backtesting  
**Primary metric**: MASE — normalized by the naive lag-12 baseline on 2002–2020

---

## Repository Structure

```
tfg-ipc-mcp/
├── tfg-forecasting/          # TFG 1 — Data Science
│   ├── 01_etl/               # 13 ingestion & feature-engineering scripts
│   ├── 02_eda/               # 13 notebooks (visual, stationarity, seasonality, ACF/PACF, regimes)
│   ├── 03_models_baseline/   # ARIMA / SARIMA / SARIMAX / AutoARIMA — Spain, Global, Europe
│   ├── 04_models_deep/       # LSTM / N-BEATS / N-HiTS — Spain, Global, Europe
│   ├── 05_mcp_pipeline/      # Spain MCP pipeline (news → features via Claude)
│   ├── 05_mcp_pipeline_global/ # Global MCP pipeline (Fed, ECB, BLS press releases)
│   ├── 06_models_foundation/ # 29 scripts — Chronos-2, TimesFM, TimeGPT (C0/C1, 3 series)
│   ├── 07_evaluation/        # Evaluation notebooks + Diebold-Mariano tests
│   ├── 08_results/           # JSON metrics, Parquet predictions, figures
│   └── configs/              # YAML model configurations
├── tfg-arquitectura/         # TFG 2 — Web platform
├── shared/                   # Common metrics, utilities, constants
├── pyproject.toml            # Shared Python dependencies
├── docker-compose.yml        # Full orchestration
└── .env.example              # Environment variables (copy to .env)
```

---

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **C0** | Univariate — model trained on historical series only |
| **C1_inst** | + Institutional signals (Fed Funds, EPU, Brent, DFR, ESI, TTF…) |
| **C1_mcp** | + MCP news signals (GDELT headlines extracted via Claude) |
| **C1_full** | + Institutional AND MCP signals combined |

All exogenous signals use **shift+1** (value known at forecast time) and are normalized with **StandardScaler** before Ridge correction.

---

## Models

| Category | Models |
|----------|--------|
| Statistical | ARIMA, SARIMA, SARIMAX, AutoARIMA (dynamic) |
| Deep learning | LSTM, N-BEATS, N-HiTS |
| Foundation | **Chronos-2** (Amazon), **TimesFM** (Google), **TimeGPT** (Nixtla) |

---

## Key Results

### Forecast accuracy — MASE at h=12 (test 2021–2024)

| Series | Best statistical | MASE | Best foundation | MASE | C1 effect |
|--------|-----------------|------|-----------------|------|-----------|
| Spain CPI | ARIMA | 0.868 | TimesFM C1_inst | 0.862 | ~0% (neutral) |
| Global CPI | ARIMA | 1.326 | Chronos-2 C1_inst ★★ | **0.976** | −26% |
| Europe HICP | SARIMA | 1.656 | TimesFM C1_full ★★ | **1.370** | −17% |

### Main findings

1. **Foundation models are context-dependent**: they beat statistical baselines for Global and Europe at long horizons (h≥3–6), but not for Spain where ARIMA dominates at all horizons.

2. **C1 signals are beneficial only for the right series**: they improve Global (−26% Chronos-2 h=12) and Europe (−17% TimesFM C1_full h=12), but *degrade* Spain (+55% TimesFM C1-MCP — signals only available from 2021, insufficient history).

3. **Family ranking**:
   - **Chronos-2**: most robust with global institutional signals. Only model with MASE < 1.0 (Global h=12 = 0.976).
   - **TimesFM**: most sensitive to MCP signals, clear benefit in Europe. Best for Spain at h=1.
   - **TimeGPT**: least reliable — extreme carry-forward errors in 2022, worst across all series.

4. **Horizon matters**: statistical models (ARIMA/SARIMA) are nearly unbeatable at h=1; foundation models start competing at h=3–6 and win at h=12 for Global and Europe.

5. **Dynamic AutoARIMA ≠ better**: reselecting ARIMA orders at each rolling origin consistently *worsens* performance versus a fixed model calibrated once on the full historical sample.

6. **Scaling is critical**: without StandardScaler, Ridge coefficients become spurious (EPU std~65 vs diff(HICP) std~0.44), inflating MAE by +534%. StandardScaler is mandatory before any exogenous correction.

---

## Evaluation Notebooks (`07_evaluation/`)

| Notebook | Content |
|----------|---------|
| `02_compare_all_models.ipynb` | Full model ranking — Spain (all C0/C1 variants) |
| `03_evaluation_global.ipynb` | Global CPI — MASE/MAE profiles, family comparison |
| `04_evaluation_europe.ipynb` | Europe HICP — MASE/MAE profiles, C1 ablation |
| `05_spain_vs_global_vs_europe.ipynb` | Cross-series synthesis — main thesis figure |

---

## Quick Start

```bash
cp .env.example .env
# Edit .env with real credentials

# Start infrastructure
docker compose up -d postgres mongo

# Install Python dependencies (virtual environment recommended)
pip install -e ".[dev]"

# Run ETL for a series
python tfg-forecasting/01_etl/01_ingest_cpi_global.py
python tfg-forecasting/01_etl/11_ingest_hicp_europe.py

# Run a foundation model (example: Chronos-2 C1_inst on Global)
python tfg-forecasting/06_models_foundation/15_chronos2_C1_inst_global.py

# Evaluate
jupyter notebook tfg-forecasting/07_evaluation/05_spain_vs_global_vs_europe.ipynb
```

---

## Output Figures (`08_results/`)

| File | Description |
|------|-------------|
| `fig_MAIN_comparison.png` | 2×3 panel — main thesis figure (all series, all panels) |
| `fig_comp1_difficulty.png` | Forecast difficulty by series (naive MASE) |
| `fig_comp2_foundation_vs_stat.png` | Foundation vs statistical — MAE profiles |
| `fig_comp3_families.png` | Family comparison (Chronos-2 / TimesFM / TimeGPT) |
| `fig_comp4_c1_effect.png` | C1 signal effect heatmap (Δ MAE %) |
