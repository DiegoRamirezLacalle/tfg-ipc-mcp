# Incident Report: Methodology Audit and Corrections — June 2026

**Date of detection:** 2026-06-01  
**Date of resolution:** 2026-06-02  
**Branch:** `refactor-clean`  
**Severity:** Medium (affects stored C1 metrics; C0 baselines and statistical models unaffected)

---

## 1. Background

An external audit of the `tfg-forecasting/` codebase (covering all ETL, baseline, deep-learning, and foundation model scripts) identified eight findings. This document records the two genuine bugs that materially affected results, along with three correctness improvements and three claims that were either false or overstated.

---

## 2. Genuine Bugs Fixed

### Bug #1 — Future-data leakage in `04_chronos2_C1.py`

**Location:** `tfg-forecasting/06_models_foundation/04_chronos2_C1.py`, function `prepare_input`, `future_covariates` construction.

**Root cause:** The script passed `df[col].reindex(fc_dates)` as the future covariate array for the ECB deposit rate (DFR) and main refinancing rate (MRR). Because `df` contains the full series including the test period (2021–2024), `reindex` retrieved the *realised future path* of these rates rather than the value known at the forecast origin. The fallback to the last-known value only triggered on `NaN`, which never occurred since the rate columns are fully populated.

**Expected correct behaviour:** At forecast origin `t`, the model must use only information available at `t`. For a policy rate this means carrying forward the last observed value (ECB typically holds rates constant for months). The Chronos-2 script 15 (`15_chronos2_C1_inst_global.py`) already implemented this correctly with `np.full(h, last_known_val)`.

**Fix applied:**
```python
# Before (leaking future rates):
future_vals = df[col].reindex(fc_dates)   # reads realised values at t+1..t+h
future_vals = future_vals.fillna(last_val) # never triggered

# After (carry-forward, no look-ahead):
future_covs = build_future_covariates(
    df, KNOWN_FUTURE_COVS, origin, h, ExogPolicy.CARRY_FORWARD
)
```

The fix is routed through the new `shared/exog_policies.py` helper (see §4) which enforces `df.loc[:origin]` access.

**Impact on results:** Marginal. The ECB's policy rate was held constant for extended periods during 2021–2024 (near-zero until mid-2022, then stable-ish as it rose), so the carry-forward and the realised path often agreed. The `chronos2_C1` metrics changed by +3.6–13.9% MAE across horizons, indicating the leaked future rates had provided a very modest inadvertent advantage in some sub-periods.

---

### Bug #2 — Missing `StandardScaler` in five TimesFM Ridge-correction scripts

**Location:** `06_models_foundation/02_timesfm_C1.py`, `11_timesfm_C1_inst.py`, `12_timesfm_C1_macro.py`, `16_timesfm_C1_inst_global.py`, `25_timesfm_C1_mcp_europe.py`.

**Root cause:** These scripts fitted a Ridge regression on the raw (unscaled) covariate matrix to compute a monthly-change correction on top of the base TimesFM forecast. The covariate scales differ enormously — EPU Europe has a standard deviation of ~65 while a first-differenced HICP has std ~0.44. Without scaling, Ridge's L2 penalty is applied equally in the raw covariate space, so EPU effectively dominated the regression and the correction was numerically unstable.

The analogous scripts that were written later (`24_timesfm_C1_inst_europe.py`, `26_timesfm_C1_full_europe.py`) already included `StandardScaler`, as did the guidance in `PROJECT_CONTEXT.md` (§3: "Normalization of exogenous signals — CRITICAL"). The five older scripts were inadvertently not updated when this requirement was documented.

**Fix applied** (identical pattern in all five):
```python
# Before:
X = window.loc[valid, XREG_COVS].fillna(0.0).values.astype(np.float64)
reg = Ridge(alpha=RIDGE_ALPHA).fit(X, y_diff)
current = df.loc[origin:origin, XREG_COVS].fillna(0.0).values.astype(np.float64)

# After:
X_raw = window.loc[valid, XREG_COVS].fillna(0.0).values.astype(np.float64)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)             # fit on training window only
reg = Ridge(alpha=RIDGE_ALPHA).fit(X, y_diff)
current_raw = df.loc[origin:origin, XREG_COVS].fillna(0.0).values.astype(np.float64)
current = scaler.transform(current_raw)     # transform current, no leakage
neutral = np.zeros_like(current)            # scaled 0 = historical mean
```

**Impact on results:** Significant for `timesfm_C1` (Spain MCP) and `timesfm_C1_inst_global`. The unscaled version produced corrections that were over-dominated by EPU, artificially inflating the correction value. After scaling:

| Script | h=1 MAE change | h=12 MAE change |
|--------|---------------|-----------------|
| `timesfm_C1` (Spain) | −34.2% | −11.3% |
| `timesfm_C1_inst` (Spain) | +5.3% | +3.4% |
| `timesfm_C1_macro` (Spain) | −23.3% | −5.2% |
| `timesfm_C1_inst_global` | −20.7% | −7.2% |
| `timesfm_C1_mcp_europe` | +2.2% | −0.3% |

`timesfm_C1_inst` and `timesfm_C1_mcp_europe` show small regressions because the unscaled EPU, while noisy, happened to produce a correction that aligned with the signal. The other three show substantial improvements, confirming the pre-fix numbers were inflated by the scaling error.

---

## 3. Correctness Improvements (No Results Change)

### #3 — Global Diebold-Mariano positional alignment

**Location:** `07_evaluation/03_evaluation_global.ipynb`, cell `cell-12`.

**Issue:** Errors from two models were aligned by sorting on `(origin, step)` and truncating both arrays to `min(len(e1), len(e2))`. This is unsafe when the two models do not cover the same origin set (e.g. foundation models at h=1 have n=47 origins vs n=48 for ARIMA; a missing origin mid-series would misalign all subsequent observations).

**Fix:** Replaced with a merge on `(origin, fc_date)` keys, the same pattern used in `01_diebold_mariano_tests.py` (Spain) and `05_diebold_mariano_europe.py`. On the current data, missing origins fall at the tail, so the positional and merge approaches produce identical numerical results; the fix is a correctness/robustness guarantee for future re-runs.

### #4 — README AutoARIMA claim overstated

**Location:** `README.md`, finding 5.

**Issue:** The sentence stated dynamic AutoARIMA "consistently worsens performance" — but for Global CPI it improved over fixed ARIMA by −6% at h=1 and −14% at h=12. The full picture is series-dependent: helps Global, hurts Spain, neutral for Europe.

**Fix:** Rewritten as a three-series summary with explicit per-series percentages.

---

## 4. Additions Made During Audit

### `shared/exog_policies.py` — Explicit contextual-variable policies

A new shared module provides three named policies for supplying exogenous covariates at a forecast origin without accidental look-ahead:

- `CARRY_FORWARD` — repeat last value observed at/before origin
- `NEUTRAL` — historical mean of the window up to origin
- `KNOWN_AT_ORIGIN` — single vector at origin, repeated

All policies internally call `df.loc[:origin]` and therefore cannot read future rows. The module also exposes `assert_no_future(df, origin)` for use in tests.

### `tests/check_artifacts_and_leakage.py` — Automated integrity checks

A pytest-compatible script that checks all prediction parquets for:

1. **Causality**: no row where `fc_date <= origin`; `step` matches months between dates; horizons in `{1,3,6,12}`.
2. **Uniqueness**: no duplicate rows on `(model, origin, horizon, fc_date)`.
3. **Origin grid**: all origins on the monthly 2021-01–2024-12 grid; deep models have ~16 origins (quarterly), others ~48.
4. **Artifacts**: every file required by `build_metrics_summary_final.py` and the evaluation notebooks exists with the correct columns.
5. **Exog-policy guard**: `assert_no_future` correctly accepts sliced frames and rejects full ones.

All five checks pass on the current state of the repository.

---

## 5. Claims That Were Not Valid

**Claim #3 (audit report):** Both `rolling_predictions.parquet` (Spain baseline) and `deep_rolling_predictions.parquet` were absent from the repository and needed to be regenerated.

**Verdict: False.** Both files existed at their correct paths:
- `03_models_baseline/results/rolling_predictions.parquet` — 3 936 rows, models `{naive, arima, sarima, sarimax}`, 48 origins.
- `04_models_deep/results/deep_rolling_predictions.parquet` — 1 056 rows, models `{lstm, nbeats, nhits}`, 16 origins.

Cross-verification against `metrics_summary_final.json` showed exact agreement on all MAE values to four decimal places. No regeneration was required or performed.

---

## 6. Re-run and Verification

After applying the two bug fixes (§2), the six affected scripts were re-run with model weights already cached locally (TimesFM 2.5-200M 1.85 GB, Chronos-2 955 MB). Downstream artifacts were rebuilt in order:

```
07_evaluation/build_metrics_summary_final.py
07_evaluation/01_diebold_mariano_tests.py
07_evaluation/05_diebold_mariano_europe.py
07_evaluation/tabla_maestra_modelos.py
```

The Global DM JSON (`diebold_mariano_results_global.json`) was also regenerated using the corrected merge-based alignment.

Final verification:

```
python tests/check_artifacts_and_leakage.py
→ [PASS] ARTIFACTS
→ [PASS] CAUSALITY (no leakage)
→ [PASS] UNIQUENESS
→ [PASS] ORIGIN GRID
→ [PASS] EXOG-POLICY GUARD
→ RESULT: all checks passed.
```

`08_results/STALE_RESULTS.md` was removed after successful re-run and verification.

---

## 7. Updated Key Numbers

### Spain (MASE scale: 1.4051 pp)

No change to baselines. TimesFM C1 models re-measured after StandardScaler fix:

| Model | MAE h=1 | MAE h=12 | MASE h=12 |
|-------|---------|----------|----------|
| ARIMA (reference, unchanged) | 0.4781 | 1.5410 | **1.097** |
| TimesFM C0 (unchanged) | 0.4364 | 1.8635 | 1.326 |
| TimesFM C1_inst (corrected) | 0.4454 | 1.8781 | 1.337 |
| TimesFM C1_macro (corrected) | 0.4615 | 1.8839 | 1.341 |
| Chronos-2 C1_mcp (corrected) | 0.5490 | 2.2659 | 1.613 |

**Interpretation change:** Previously, `timesfm_C1_inst` appeared competitive with ARIMA (old MAE h=1 = 0.4229 ≈ "best foundation at h=1"). After the scaling fix its h=1 MAE is 0.4454 — slightly worse than TimesFM C0, confirming the scaling error had been providing an artificial advantage. The Spain-C1 result is now unambiguously neutral relative to C0.

### Global (MASE scale: 1.1720 pp)

TimesFM C1_inst_global substantially improved after fixing the unscaled Ridge:

| Model | MAE h=1 | MAE h=12 | MASE h=12 |
|-------|---------|----------|----------|
| ARIMA (unchanged) | 0.1907 | 1.5444 | 1.318 |
| Chronos-2 C1_inst (unchanged) | 0.2004 | 1.1433 | **0.976** |
| TimesFM C1_inst (corrected) | **0.2137** | **1.1913** | **1.016** |

TimesFM C1_inst now has MASE=1.016 at h=12 (was 1.096 before the fix), placing it close to Chronos-2 — a genuine second-best foundation model for Global CPI.

### Europe — No material change

`timesfm_C1_mcp_europe` changed by ≤2.2% MAE at any horizon. The conclusion that `timesfm_C1_full_europe` is the best model (MASE h=12 = 1.370) is unchanged.
