# Forward-path covariates — generalization across Spain, Global, Europe

> **Update (Phase B):** the "not statistically significant / does not change the
> conclusion" verdict below was a *power artifact* of having only 3 series. On a
> 19-country euro-area HICP panel the forward-path effect is **significant** (vs
> flat-hold p<0.01, vs C0 p=0.02). See [phase_b_panel_report.md](phase_b_panel_report.md).


Generalizes the Global Chronos-2 `C1_fwd` idea (honest forward covariate path
instead of flat carry-forward) to all three CPI series. Same rolling-origin
protocol throughout: origins 2021-01→2024-12, horizons {1,3,6,12}, expanding
window, MASE scale fixed on 2002-2020. The forward path is a **damped
random-walk-with-drift** (trailing-12m drift, φ=0.85), generated per origin from
data ≤ origin only — no temporal leakage. Implemented once in
`shared/exog_policies.py::ExogPolicy.FORWARD_PATH` and reused by every variant.

## Which models / datasets were tested?

| Series | Model | flat-hold (C1) | forward-path (C1_fwd) | script |
|---|---|---|---|---|
| Spain CPI (index) | Chronos-2, EPU-Europe covariates | `chronos2_C1_inst` | `chronos2_C1_fwd_spain` | `06_models_foundation/34_…` |
| Global CPI (YoY rate) | Chronos-2, institutional covariates | `chronos2_C1_inst_global` | `chronos2_C1_fwd_global` | `06_models_foundation/33_…` (preserved) |
| Europe HICP (index) | Chronos-2, institutional covariates | `chronos2_C1_inst_europe` | `chronos2_C1_fwd_europe` | `06_models_foundation/35_…` |

Acceptance criterion met: ≥1 forward-path variant per dataset; the Global result
is preserved exactly (MAE 0.1925 / 0.3074 / 0.5299 / 1.0648). New variants are
clearly named and do **not** overwrite any existing C0/C1 result.

### What remains untested (and why)

- **TimesFM + Ridge** — its context enters as a *single scalar correction added
  flat to all 12 steps*; there is no future covariate **path** to replace, so the
  flat-vs-forward contrast does not apply without redesigning the corrector.
  Out of scope for a like-for-like comparison.
- **TimeGPT** — natively supports a future-covariate path, but the Nixtla API is
  unavailable/rate-limited in this environment (the Global TimeGPT C0 could not
  be generated either). Blocked, not declined. `[VERIFY]` when API access returns.
- **Classical SARIMAX/ARIMAX** — these already consume the *realised* future exog
  (oracle), not a flat carry-forward, so the relevant contrast there is
  oracle-vs-forecast (which can only make them worse by removing oracle info), a
  different experiment from flat-vs-forward. Deliberately not relabelled as fwd.

## Did forward-path beat flat carry-forward? (dMAE %, − = better)

| Series | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Spain  | −1.3% | −1.5% | −1.5% | +1.0% |
| Global | −3.9% | −10.0% | −10.4% | −6.9% |
| Europe | +2.3% | −3.7% | −5.6% | −5.7% |

Forward-path lowers MAE in **10 of 12** (series × horizon) cells; the two
exceptions are boundary horizons (Spain h=12, Europe h=1) and small. The gain is
**largest at medium/long horizons** (Global −7…−10%, Europe −4…−6% at h=3/6/12) —
exactly where flat-hold's frozen covariate path is most wrong. Point estimates
(descriptive).

## Consistent by horizon?

Yes in direction for Global (all horizons) and Europe (h≥3); for Spain the gain is
flat and small (≈−1.5%) at h≤6 and reverses slightly at h=12. The pattern is
coherent: replacing a frozen future with a momentum-aware one helps more as the
horizon (hence the flat-hold error) grows, until very long horizons where the
damped drift itself becomes uncertain.

## Statistically tested? (strict HLN-adjusted Diebold–Mariano)

Every comparison is tested in `07_thesis_critical_dm_recompute.py` (HLN-adjusted
Student-t DM, HAC lags h−1, df=n−1, abs & sq loss, origin-clustered bootstrap CI),
path- and endpoint-level. Results in `thesis_critical_dm_recompute.{md,csv}`;
the consolidated view is `forward_path_comparison.{md,csv}`.

- **flat-hold vs forward-path:** no single cell reaches p<0.10 — the MAE gains are
  real in point estimate but within sampling noise over 36–47 origins. **Descriptive.**
- **forward-path vs C0:** significant only for **Global h=1** (p=0.075, fwd better
  by −23.6%). All other fwd-vs-C0 cells are not significant.
- **Honest caveat:** at the Spain **endpoint** h=12 (step=12 only), flat-hold is
  *significantly* better than forward-path (p=0.011, +1.8%). The long-horizon
  endpoint is the one place the forward path measurably hurts.

## Which results are descriptive only?

All MAE/RMSE/MASE numbers and every dMAE% are descriptive point estimates. The
only inferential (DM-significant) forward-path finding is Global fwd-vs-C0 at h=1
(marginal, p=0.075) — plus the negative Spain-endpoint-h12 caveat above. The
flat-vs-forward improvements are descriptive, not significant.

## What to say — and not say — in the defense

**Say:**
- A leakage-free forward covariate path (damped RW-with-drift) **consistently
  lowers MAE vs the naïve flat carry-forward** across all three series, most at
  medium/long horizons. *How the covariate future is represented matters more than
  which signal is added* — the central methodological point, now shown to
  generalize beyond the single Global case.
- On the Global YoY-rate target the forward-path model is the **best global model**
  (MASE 0.16 / 0.26 / 0.45 / 0.91) and beats univariate C0 (DM-marginal at h=1).

**Do not say:**
- That forward-path makes context **beat C0 on the index targets** (Spain, Europe).
  It does not; it mainly **repairs the damage** flat-hold caused, reaching parity
  with C0 (Europe ≈neutral, slight win at h=12; Spain still ≈C0-or-worse).
- That the flat-vs-forward improvement is **statistically significant** — it is not
  (descriptive only); and at Spain endpoint h=12 flat-hold is significantly better.

**Bottom line:** this **extends the robustness analysis** — it strengthens the
methodological claim and generalizes it across datasets — but it **does not change
the headline conclusion**: exogenous/semantic context helps the foundation forecast
only on the Global rate target (concentrated in the 2022–23 shock); on the price-
index targets it does not beat univariate C0.

## Reproduce

```
cd tfg-forecasting
python 06_models_foundation/33_chronos2_C1_fwd_global.py   # (preserved) Global, ~2 min CPU
python 06_models_foundation/34_chronos2_C1_fwd_spain.py    # Spain,  ~1 min CPU
python 06_models_foundation/35_chronos2_C1_fwd_europe.py   # Europe, ~1 min CPU
python 07_evaluation/07_thesis_critical_dm_recompute.py    # strict HLN-DM (+ fwd comparisons)
python 07_evaluation/12_forward_path_comparison.py         # forward_path_comparison.{md,csv}
python 07_evaluation/audit_foundation_targets.py           # must exit 0
```
