# Do exogenous/MCP signals improve foundation CPI forecasts? — findings

Scope: whether C1 (with energy / institutional / macro / MCP-semantic signals)
beats C0 (univariate) for Chronos-2, TimesFM and TimeGPT on three CPI series
(Spain index, Global YoY rate, Europe HICP index). Rolling-origin protocol:
origins 2021-01→2024-12, horizons {1,3,6,12}, expanding window, MASE scale fixed
on 2002–2020. No leakage; signal sets/transforms decided before evaluating.

## TL;DR (honest)

**Signals help on exactly one of the three series — the Global YoY-rate target —
and the gain is real but concentrated in the 2022–23 shock.** On the two
price-index targets (Spain, Europe HICP) signals do not beat C0; the best we
achieve is to remove the damage the original C1 models caused. This is reported
as a valid thesis finding, not engineered away.

The single biggest lever is **how the future of the exogenous path is treated**,
not which signal is added.

## What was wrong (Phase 0)

Full diagnosis: [phase0_diagnostics.md](phase0_diagnostics.md)
(script [00_phase0_diagnostics.py](../07_evaluation/00_phase0_diagnostics.py)).

1. **Signals track the price LEVEL, not the CHANGE we forecast.** e.g. Europe
   `epu_europe_ma3` corr +0.81 with the level, +0.12 with the next-month
   increment; Global `imf_comm_ma3` +0.58 / +0.00. The original C1 models pick
   covariates by correlation-with-level, so they add regime variance.
2. **Future covariates were held FLAT** (last value repeated) for the native
   covariate models (Chronos-2, TimeGPT).
3. **The TimesFM "Ridge correction" is a single scalar added flat to all 12
   horizon steps** (within-origin spread = 0 exactly) — dimensionally wrong on a
   path.
4. Failure mode: on index targets C1 hurts via a **bias** that compounds with
   the horizon; on the Global rate target C1 already had the right sign
   (it cut C0's shock-era under-forecast bias) but was statistically
   insignificant due to high variance over ~36–47 origins.

## What was changed

No original C0/C1 forecast was modified. All work reuses stored C0 forecasts or
re-runs a single model; signal sets and transforms were fixed on pre-2021 data
only.

- **Phase 1+3 — change-correlated + regime-gated context overlay**
  ([09_regime_context_overlay.py](../07_evaluation/09_regime_context_overlay.py)).
  A Ridge context correction on a *change* target, selected on pre-2021
  validation, applied to stored C0 forecasts. Adds (a) a `change_sel` candidate
  family = top-3 signals by training-window |corr with the increment|, (b) a
  fixed regime gate (apply only when trailing-12m volatility of the target's MoM
  change exceeds the training 80th percentile). Builds on the earlier always-on
  overlay [08_validated_context_overlay.py](../07_evaluation/08_validated_context_overlay.py).
- **Phase 2 — Chronos-2 Global C1 with an honest forward covariate path**
  ([33_chronos2_C1_fwd_global.py](../06_models_foundation/33_chronos2_C1_fwd_global.py)).
  Identical to script 15 except future covariates use a damped RW-with-drift
  forecast (12m drift, φ=0.85, data ≤ origin) instead of flat-hold. This isolates
  root cause #2.

## Before/after (C1 vs C0, MAE delta and DM p-value)

**Canonical strict evidence: [thesis_critical_dm_recompute.md](thesis_critical_dm_recompute.md)**
(independent recompute from Parquets, HLN-adjusted Student-t DM with bootstrap
CIs). [before_after_c1.md](before_after_c1.md) uses the same strict test, laid
out per variant. [dm_pvalues_summary.md](dm_pvalues_summary.md) is
**legacy/supplementary only** — it uses the older normal-reference
`shared.metrics.diebold_mariano` and should not be cited as the primary evidence.

**Global (the win).** dMAE% vs Global C0, strict HLN DM p in parentheses:

| variant (Chronos-2) | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| inst (flat-hold, original) | −20.4% (0.128) | −4.5% (0.745) | −7.9% (0.661) | −14.5% (0.476) |
| **fwd (forward path)** | **−23.6% (0.075\*)** | −14.1% (0.284) | −17.5% (0.216) | −20.4% (0.166) |
| regime (gated overlay) | −1.3% (0.201) | −1.6% (0.019\*\*) | −1.0% (0.039\*\*) | −0.5% (0.054\*) |

The forward-path model is the **best global model by MAE/MASE** (MASE
0.16 / 0.26 / 0.45 / 0.91) and improves on the flat-hold C1 by ≈4–10%. Its DM
significance is limited to h=1 (p=0.075) because the gain is concentrated in the
shock over few origins. The regime-gated overlay is the opposite trade-off:
small but **DM-significant at h=3/6** (p=0.019 / 0.039) and marginal at h=12
(p=0.054). For TimesFM Global the always-on overlay helps (h1–6, p<0.10) but the
regime gate removes the benefit (the signal helps broadly there, not only in
high-vol months) — reported honestly.

**Spain and Europe (no win).** The original with-signals models are at best
neutral and often significantly *worse* than C0 (Spain TimesFM inst +0.8…+2.0%,
p<0.05; Europe full +1…+24%). On **Spain** the overlays' best pre-2021 recipe
does not beat the zero-correction baseline, so they correctly emit a **no-op**
(`chronos2/timesfm_C1_validated` ≡ `_C1_regime` ≡ C0). On **Europe** the overlays
apply but stay neutral-to-slightly-worse. No variant beats C0: outside the rate
target and the shock regime, the signals do not help the foundation forecast.

## Phase 4 — data-quality note (ECB placeholder)

The known constant-ECB-columns issue is in `data/processed/mcp_signals_global.parquet`
(`ecb_hawkish_score`, `ecb_surprise_score`, `ecb_forward_guidance_num`, plus
`us_cpi_surprise_score`, `us_cpi_components_pressure` are all constant). **That
file feeds no evaluated model** — it is only referenced by the pipeline that
produces it (`05_mcp_pipeline_global/`). The real ECB deposit-facility rate
`dfr` (range −0.5→4.0) already feeds every evaluated C1 feature set
(`features_c1.parquet`, `features_c1_global_institutional.parquet`,
`features_c1_europe.parquet`). So the placeholder does not contaminate any C1
result, and no fix was fabricated for a file nothing reads. If a true
`C1_mcp_global` model is ever built, replace those columns with real `dfr` first.

## Methodology corrections (defensibility)

These do not change the modeling scope; they make the evaluation more defensible.

- **Strict DM in the before/after.** `before_after_c1.py` now uses the
  HLN-adjusted Student-t test (HAC lags h−1, df=n−1), identical to
  `thesis_critical_dm_recompute.py`, not the normal-reference
  `shared.metrics.diebold_mariano`. The strict test is more conservative: Global
  Chronos-2 regime h=12 moves p=0.048→0.054 and forward-path h=1 → 0.075; the
  h=3/h=6 regime wins stay significant. Conclusions are unchanged.
- **Comparable MASE.** Overlay metrics (scripts 08/09) now use the seasonal
  lag-12 MASE scale on the training series, matching the foundation models
  (`mean |y[t]−y[t−12]|`), instead of a lag-1 scale. MAE/RMSE were unaffected.
- **No `change_sel` leak.** Script 09 selects change-correlated features only
  from rows whose full increment lies in the pre-2021 window (t+1 ≤ 2020-12),
  removing a one-month leak into the first test increment. It is still not chosen
  by the calm validation window, so selected recipes are unchanged.
- **No-op when unvalidated.** The overlays apply a correction only if the
  selected recipe beat the zero-correction baseline on pre-2021 validation;
  otherwise they emit a no-op identical to C0. This is why the Spain overlays now
  equal C0 exactly (their best recipe validates at +0.56%, i.e. worse than zero).

## Integrity

`python 07_evaluation/audit_foundation_targets.py` exits 0 and now also verifies
the new Global C1 variants (`chronos2_C1_fwd_global`, regime/validated globals)
carry `cpi_global_rate` as `y_true` and are not on the Spain index scale.

## Reproduce

```
cd tfg-forecasting
python 07_evaluation/00_phase0_diagnostics.py          # diagnosis
python 07_evaluation/08_validated_context_overlay.py   # always-on overlay (baseline)
python 07_evaluation/09_regime_context_overlay.py      # change-sel + regime overlay
python 06_models_foundation/33_chronos2_C1_fwd_global.py  # forward-path Chronos-2 (~2 min, CPU)
python 07_evaluation/06_extract_dm_pvalues.py          # dm_pvalues_summary.{csv,md}
python 07_evaluation/07_thesis_critical_dm_recompute.py
python 07_evaluation/10_before_after_c1.py             # before_after_c1.md
python 07_evaluation/tabla_maestra_modelos.py --no-open
python 07_evaluation/audit_foundation_targets.py       # must exit 0
```
