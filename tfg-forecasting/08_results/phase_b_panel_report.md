# Phase B — euro-area country panel: forward-path covariates significantly help

Phase A (3 series) showed forward-path covariates beat flat carry-forward in
point estimate but couldn't reach significance — too few cross-sectional units.
Phase B fixes the power problem with a **19-country euro-area HICP panel** and
finds the effect is **significant and robust**.

## What was done

- **Data** ([01_etl/14_ingest_hicp_panel_europe.py](../01_etl/14_ingest_hicp_panel_europe.py)):
  per-country HICP (all-items price index, 2002-01–2024-12) for 19 euro-area
  members from the ECB SDMX API — AT BE CY DE EE ES FI FR GR IE IT LT LU LV MT
  NL PT SI SK, all with complete coverage → `data/processed/hicp_panel_europe.parquet`.
- **Forecasts** ([06_models_foundation/36_chronos2_panel_europe.py](../06_models_foundation/36_chronos2_panel_europe.py)):
  the *same* zero-shot Chronos-2 protocol as the single Europe series, per
  country, three conditions — C0 (univariate), C1_inst (flat carry-forward
  covariates), C1_fwd (forward-path covariates). Covariates are the **shared
  euro-area set** (EPU Europe, Brent, ECB rate, ESI, EUR/USD) — area-wide drivers
  common to every member, so no per-country covariate sourcing is needed. Rolling
  origins 2021-01–2024-12, h={1,3,6,12}, MASE scale per country on 2002–2020.
- **Inference** ([07_evaluation/15_panel_dm_country.py](../07_evaluation/15_panel_dm_country.py)):
  pooled cluster-robust mean test on per-country MASE-scaled loss differentials,
  reported under both origin (time) and country clustering; `p_honest` = the more
  conservative of the two. No retraining anywhere.

## Headline results (all-horizon pooled, p_honest)

| Contrast | pooled effect | p_honest | countries better | verdict |
|---|---|---|---|---|
| C1 **flat-hold** context vs C0 | +0.028 | 0.60 | 10/19 | ns — flat context doesn't help |
| C1 **forward-path** context vs C0 | +0.094 | **0.021 \*\*** | 13/19 | **significant** |
| C1 **forward-path vs flat-hold** | +0.066 | **0.0072 \*\*** | 16/19 | **significant** |

Full per-horizon table: [panel_dm_country.md](panel_dm_country.md).

- **Forward-path vs flat-hold** is significant at *every* horizon (h1 p=0.090, h3
  p=0.013, h6 p=0.0055, h12 p=0.0092), 13–16 of 19 countries improve, under both
  clusterings. The most robust result in the study.
- **Forward-path vs C0** strengthens with horizon (ns at h1 → p=0.008 at h12,
  15/19 countries) and is significant pooled (p=0.021). Forward-path covariates
  beat the univariate model on the panel.
- **Flat-hold context vs C0** is never significant — exactly as in every prior
  test.

## Interpretation (the contribution)

The value is **not the signal, but representing its future path honestly.**
Adding institutional context with a frozen (flat) future does not beat univariate
C0; the *identical* context with a damped-drift forward path does — significantly,
and on 13–16 of 19 independent country series. This sharpens the thesis from "did
signals help?" (mostly no) to "how must a covariate's future be supplied for a
foundation model to use it?" (with a forecast, not a constant), which is a
cleaner and more general methodological claim.

## Does this change the defense conclusion?

**Yes — it upgrades it.** The earlier forward-path write-up
([forward_path_report.md](forward_path_report.md)) concluded the gain "does not
change the headline conclusion" because, on 3 series, it wasn't significant. Phase
B shows that was a **power artifact**: with an adequately powered panel, honest
forward-path covariates **significantly** improve foundation HICP forecasts over
both flat-hold context (p<0.01) and univariate C0 (p=0.02). The flat-hold context
still fails, which is the point.

## Robustness + placebo (is the win real?)

Full table: [panel_robustness_dm.md](panel_robustness_dm.md)
(scripts [37_chronos2_panel_robustness.py](../06_models_foundation/37_chronos2_panel_robustness.py),
[16_panel_robustness_dm.py](../07_evaluation/16_panel_robustness_dm.py)).

**Not tuned — every informed forward-path setting beats flat-hold.** Re-running
with other a-priori hyperparameters (all-horizon pooled, honest p):

| forward-path variant | vs flat-hold | vs C0 |
|---|---|---|
| canonical (φ=0.85, w=12) | p=0.007 ** (16/19) | p=0.021 ** (13/19) |
| undamped (φ=1.0, w=12)   | p=0.081 * (14/19)  | p=0.009 ** (15/19) |
| window=24 (φ=0.85)       | p=0.003 ** (14/19) | p=0.148 ns (10/19) |

The forward-vs-flat win survives every setting (φ=0.85 is not cherry-picked);
damping helps a little (undamped is weaker vs flat) but even the undamped path
still beats flat-hold.

**Informed, not just non-flat — the placebo fails.** A random-sign forward path
(identical damped-drift *magnitude*, randomized *direction*, seed 20260629):

| contrast | result |
|---|---|
| placebo vs flat-hold | **A better** (flat beats placebo), p=0.080 — a non-flat but uninformed path does not help, it slightly hurts |
| placebo vs C0 | ns (p=0.86) — no benefit over univariate |
| canonical forward vs placebo | **p=0.003 ** (16/19)** — informed path significantly beats placebo |

So the gain is specifically the **informed direction of recent momentum**, not
merely supplying any non-flat covariate path. This is the decisive defense
against "you just added noise / tuned the heuristic".

## What to say — and not say

**Say:** on a 19-country euro-area HICP panel, honest forward-path covariates
significantly beat both flat carry-forward and the univariate baseline (cluster-
robust, both clusterings, 13–16/19 countries). The effect is in the *future-path
representation*, not the signal.

**Do not over-claim:**
- It is demonstrated on **euro-area HICP with shared covariates** and the
  **Chronos-2** family. External validity to non-euro economies, per-country
  covariates, and other model families (TimesFM, TimeGPT) is untested.
- **Plain flat-hold context still does not beat C0** — the win is specifically the
  forward path.
- Significance comes from the cross-sectional panel; any single country is still
  individually noisy.

## What remains untested (next)

- Per-country covariates (national EPU, energy mix) instead of shared euro-area.
- The same panel for TimesFM (Ridge path-aware) and TimeGPT (API permitting).
- A non-euro country panel (OECD CPI) for external validity.

## Reproduce

```
cd tfg-forecasting
python 01_etl/14_ingest_hicp_panel_europe.py        # fetch 19-country HICP (network)
python 06_models_foundation/36_chronos2_panel_europe.py  # ~8 min CPU
python 07_evaluation/15_panel_dm_country.py         # panel_dm_country.{md,csv}
```
