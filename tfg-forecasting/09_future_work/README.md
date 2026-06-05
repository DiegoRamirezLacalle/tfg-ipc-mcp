# 09 · Future Work — MIDAS and Nexus-inspired Extensions

This folder implements two architectures from the recent state-of-the-art
(2020–2026) that extend the experimental framework of the TFG beyond the
Ridge-XReg + Foundation-model baselines in `06_models_foundation/`.

Four experiments in total, two per architecture (monthly proxy → daily HF for
MIDAS; single-step heuristic → 3-agent Ollama chain for Nexus).

---

## Context in the literature (2020–2026)

The state-of-the-art has evolved through three waves:

| Wave | Period | Approach |
|------|--------|----------|
| 1st | 2020–2023 | Text → scalar aggregates aligned with series via **MIDAS** |
| 2nd | 2024–2025 | Text → embeddings fused with series via **cross-attention**, dual-tower, experts |
| 3rd | 2026 | **TSFM strong prior + LLM contextual reviser** (PostTime, Nexus) |

The experiments here sit at Wave 1 (MIDAS) and Wave 3 (Nexus/PostTime), and
connect directly to the three-level contextualisation framework recommended as
"most defensible" for a TFG (Babii et al. 2020; PostTime, Nexus 2026).

---

## Experiment A · MIDAS-ADL — Beta polynomial lag weights (monthly proxy)

**Script:** `01_midas_spain.py` / `02_midas_global.py`

MIDAS (Ghysels et al. 2004) constrains the distributed-lag polynomial to a
two-parameter Beta function instead of fitting one free coefficient per lag:

```
y_{t+h} = α + ρ·y_t  +  Σⱼ  βⱼ · (wⱼ(θⱼ) · xⱼ_{t:t-K})  + ε
w_k(θ₁,θ₂) ∝ (k/K)^(θ₁-1) · (1-k/K)^(θ₂-1)   (Beta polynomial)
```

This enforces smooth temporal decay and limits overfitting to 3 parameters
per variable, regardless of K.

**Variables (K=12 monthly lags each):** `brent_log`, `epu_europe_log`, `dfr`

### Results — Spain CPI

| Model | MAE h=1 | MAE h=3 | MAE h=6 | MAE h=12 | MASE h=1 |
|-------|---------|---------|---------|----------|----------|
| ARIMA (reference) | 0.4781 | 0.6716 | 0.9660 | 1.5410 | 0.340 |
| MIDAS-ADL (monthly proxy) | **0.4677** | 1.0348 | 1.5707 | 2.7863 | **0.333** |

### Framing: MIDAS as a one-step *nowcaster*, not a 12-month forecaster

The honest reading of these numbers is that MIDAS is **competitive exactly where
it was designed to operate — the nowcast (h=1)** — and should be presented as a
nowcasting tool, not a long-horizon forecaster:

- At **h=1** MIDAS achieves MAE 0.4677, **beating ARIMA by −2.2%** and matching it
  on MASE (0.333 vs 0.340). The Beta polynomial extracts a small but real edge
  from the recent-lag structure of energy/uncertainty signals at the nowcast
  horizon — the classic MIDAS use case (Ghysels et al. 2004; ML-MIDAS for GDP
  nowcasting, Babii et al. 2020).
- At **h≥3** MIDAS degrades sharply because a single fixed lag polynomial cannot
  carry exogenous information several months ahead; iterated/dynamic-factor MIDAS
  would be required, and even then monthly proxies cap the gain (see Experiment B).

**Conclusion for the thesis:** report MIDAS as a *complement* to ARIMA at the
nowcast horizon (where mixed-frequency signal aggregation adds value), not as a
competitor across all horizons. This is both honest and aligned with how MIDAS is
used in the macro-nowcasting literature.

---

## Experiment B · Daily-frequency MIDAS (3 variants)

**Script:** `01c_midas_daily_spain.py`  (fully vectorised — 3 variants × 48 origins
× 4 horizons run in ~65 s; reproducible).

Three variants using **daily WTI crude oil** (CL=F via yfinance, 2002–2024) as the
high-frequency signal alongside monthly ECB DFR and EPU. All predict the h-step
CPI *change* (stationary) and reconstruct the level as `y_t + Δ`:

**(B1) Daily Beta-MIDAS (levels)** — Beta polynomial over N=82 daily log-price
lags. The polynomial learns *which days within the window* matter most.

**(B2) Daily Beta-MIDAS (returns)** ← *the "second attempt"*. Daily log-prices
are near-unit-root: within one month they barely move from their mean, so the
level lags add little beyond the monthly average. Daily log-**returns** instead
capture *within-month momentum and volatility* — genuinely sub-monthly signal.

**(B3) Ridge-daily** — unconstrained free lag weights (no polynomial shape),
the data-driven upper bound on what the daily block can extract.

### Results — Spain CPI

| Model | MAE h=1 | MAE h=3 | MAE h=6 | MAE h=12 | MASE h=1 |
|-------|---------|---------|---------|----------|----------|
| ARIMA | 0.4781 | 0.6716 | 0.9660 | 1.5410 | 0.340 |
| MIDAS-ADL monthly proxy | **0.4677** | 1.0348 | 1.5707 | 2.7863 | **0.333** |
| Daily Beta-MIDAS (levels) | 0.4940 | 1.0858 | 1.8180 | 3.5465 | 0.352 |
| **Daily Beta-MIDAS (returns)** | 0.4923 | 1.0888 | **1.7715** | **3.4030** | 0.350 |
| Ridge-daily (free lags) | 0.5065 | 1.0792 | 1.7841 | 3.5836 | 0.360 |

**Did the returns variant help?** Marginally — within-month oil *momentum* edges
out the *level* at long horizons (h=12: 3.4030 vs 3.5465, **−4%**) and ties it at
h=1, confirming the diagnosis that the daily level is near-redundant with its
monthly mean. But the gain is small and **none of the daily variants beats either
the monthly proxy or ARIMA**.

**Key finding:** daily oil dynamics do **not** add information beyond the monthly
average for *monthly* CPI — the energy→headline-CPI transmission operates at a
monthly lag, not a daily one. This reproduces the ML-MIDAS insight (Babii et al.
2020): daily data helps when the signal is truly high-frequency *relative to the
target's publication lag* (GDP nowcasting with daily surveys), not when both
target and the economically-relevant signal are monthly. The right HF signal to
revisit would be **daily news/sentiment flow** (GDELT), not commodity prices.

**Connection to literature:** ML-MIDAS (Babii et al. 2020, arXiv:2005.14057)
demonstrates exactly this gain for US GDP nowcasting with daily news sentiment.
Factor-augmented sparse MIDAS (2023) extends to panels of daily news — the
natural next step for our GDELT daily tone signal.

---

## Experiment C · Nexus — Single-step heuristic reviser

**Script:** `03_nexus_reviser_spain.py`

```
PRIOR     TimesFM C0 (frozen)
   |
REVISER   Rule-based macro heuristic:
           {ECB rate, Brent deviation, EPU percentile, BCE tone}
           → {direction, confidence, magnitude}
   |
CORRECTION  Δ(h) = d × c × m × 0.30 × 0.70^(h-1)   [bounded ±0.30 pp at h=1]
```

### Results (48 origins, 2021–2024)

| Model | MAE h=1 | MAE h=3 | MAE h=6 | MAE h=12 |
|-------|---------|---------|---------|----------|
| TimesFM C0 (prior) | 0.4364 | 0.7320 | 1.0866 | 1.8635 |
| TimesFM C1_inst (Ridge) | 0.4454 | 0.7460 | 1.1000 | 1.8781 |
| **Nexus heuristic** | 0.4400 | **0.7284** | **1.0849** | **1.8634** |

Marginally improves over the prior at h=3,6 and beats TimesFM C1_inst at all
horizons (−0.8% to −2.4%).  Direction distribution: UP=21, NEUTRAL=20, DOWN=6
— the heuristic correctly flags the 2022 energy shock as inflationary.

---

## Experiment D · Nexus multi-agent — 3-stage LLM chain (Ollama)

**Script:** `04_nexus_multiagent_ollama.py`

Proper multi-agent implementation of the Nexus/PostTime architecture using a
**locally-running LLM** (`qwen3:4b` via Ollama) with three specialist agents:

```
Agent 1  SIGNAL ANALYST
  In:  macro snapshot (ECB, Brent, EPU, GDELT, BCE tone)
  Out: {ecb_regime, energy_regime, uncertainty_regime, signal_summary}
         ↓
Agent 2  TRANSMISSION REASONER
  In:  signal regimes + recent CPI trajectory
  Out: {main_channel, expected_direction, peak_effect_months,
        confidence, reasoning_chain}
         ↓
Agent 3  REVISION DECIDER
  In:  TimesFM C0 prior + transmission assessment
  Out: {should_revise, direction, correction_pp, revision_rationale}
```

Each agent's output feeds the next.  The full chain-of-thought trace is
stored in `results/nexus_multiagent_traces.json`.

### Results (48 origins, 2021–2024 · qwen3:4b · ~3 min/call × 3 agents)

| Model | MAE h=1 | MAE h=3 | MAE h=6 | MAE h=12 |
|-------|---------|---------|---------|----------|
| TimesFM C0 (prior) | 0.4364 | 0.7320 | 1.0866 | 1.8635 |
| Nexus heuristic (Exp C) | 0.4400 | 0.7284 | 1.0849 | 1.8634 |
| **Nexus multi-agent qwen3:4b** | 0.4398 | 0.7402 | 1.0895 | 1.8639 |

**Agent chain quality:** 47/48 origins succeeded through all 3 agents (98%).
42/48 origins resulted in a revision decision. Direction distribution: UP=20,
DOWN=22, NEUTRAL=1 — the model identifies 2022 as dominantly disinflationary
from ECB tightening, and 2021 as inflationary from energy shocks.

**Key finding (reproduces PostTime's core insight):** the zero-shot multi-agent
chain is neutral to marginally worse than the simple heuristic at h=1,3.
The LLM reasons correctly about macro regimes (Agent 2's `reasoning_chain` is
coherent and references the right transmission channels), but over-revises
because it was not trained to know *when not to revise*.  This is exactly
PostTime's motivation for RLVR: the reward signal explicitly penalises
unnecessary revisions, teaching the model to defer to the TSFM prior when
confidence is low.

**Sample trace — origin 2023-01:**
- Agent 1: `ecb_regime=tightening | energy_regime=shock_high | uncertainty=elevated`
- Agent 2: `channel=mixed | direction=DOWN | confidence=MEDIUM | peak_effect=3m`
  > *"The recent energy shock will have a short-term impact on CPI due to pass-through
  > effects. The ECB tightening will start to take effect in the next few months,
  > leading to higher interest rates and reducing demand."*
- Agent 3: `should_revise=True | correction=−0.075 pp`
  > *"Downward pressure on prices over the next few months."*

---

## Experiment E · Why the reviser fails — gating, sign, and directional accuracy

**Script:** `05_nexus_analysis.py` (post-hoc, re-uses cached traces — runs in <1 s,
no LLM calls). Answers *"would gating make Nexus win, and why/why not?"*

### E.1 — Confidence cannot be gated (degenerate self-report)

`qwen3:4b` reports **`confidence=MEDIUM` for all 47 origins** — never HIGH or LOW.
The model's self-reported confidence is unusable as a gate. So we test *behavioural*
gates instead (revise only when the proposed correction is large, or when the two
reasoning agents agree on direction).

### E.2 — A discovered local-LLM failure mode: sign inconsistency

On **21 / 47 origins** the model states `direction=UP` but emits a **negative**
`correction_pp`. It cannot keep the sign of its numeric output consistent with its
own stated direction. The original run (P1) therefore applied a silent *downward*
bias on every revised origin.

### E.3 — No correction/gating policy beats the prior

| Policy | h=1 MAE | vs prior |
|--------|---------|----------|
| **P0 TimesFM C0 (no revision)** | **0.4364** | — |
| P1 as-run (raw corr, sign-buggy) | 0.4398 | +0.8% |
| P2 sign-enforced (`dir × |mag|`) | 0.4586 | +5.1% |
| P3 sign + `|corr|≥0.075` gate | 0.4569 | +4.7% |
| P4 sign + agent-agreement gate | 0.4581 | +5.0% |

Fixing the sign bug makes it **worse** (P2 +5.1%), and neither gate recovers a win.
The reason is E.4.

### E.4 — The root cause: directional accuracy is below chance

Comparing the LLM's stated direction against the realised sign of the prior's error
(`sign(y_true − prior)`, dead-band ±0.05 pp):

- **Directional accuracy: 34%** on the 38 decisive (UP-vs-DOWN) origins —
  **13 correct / 25 wrong**. The reasoning is *anti-informative*: when it says
  "UP" the prior actually under-shot only 7 times but over-shot 13 times.

| stated → actual | count |
|-----------------|-------|
| UP → UP (correct) | 7 |
| UP → DOWN (wrong) | 13 |
| DOWN → DOWN (correct) | 6 |
| DOWN → UP (wrong) | 12 |

This is *the* decisive result: **you cannot gate your way to a win when the
underlying directional signal is worse than a coin flip.** The fluent
`reasoning_chain` (correct regime identification, plausible transmission logic)
does not translate into correctly predicting where the strong TSFM prior will err.

### E.5 — What would actually be needed

This empirically confirms PostTime's motivation (arXiv:2402.03885): a zero-shot
LLM reviser is not enough. A reviser must be **trained** (SFT + RLVR) so that its
stated direction is *rewarded for matching realised outcomes* and *penalised for
unnecessary revision*. The gating experiments here show the inference-time shortcut
(thresholds, agreement rules) cannot substitute for that training signal. The
positive contribution of the reasoning layer is its *interpretability* (every
revision carries an auditable macro rationale), not its point-forecast accuracy.

---

## Experiment F · Is it the model size? (4B vs 8B ablation)

**Script:** `06_nexus_model_probe.py` (single consolidated reviser call per origin,
to isolate *model capacity* from the multi-agent architecture).

The natural follow-up: was the 34% directional accuracy a *small-model* limitation
or a *fundamental zero-shot* one? We re-ran the reviser as a single call with two
Qwen3 models on identical prompts. (Run locally on a GTX 1050 Ti / 4 GB VRAM;
qwen3:8b Q4_K_M spills ~2 GB to CPU and runs ~2× slower but completes.)

Two robustness fixes were needed first — both are findings about small local LLMs:
- **Thinking mode** (`/no_think` + `think=False`): Qwen3 reasons by default and, in
  JSON mode, burns the entire token budget *thinking* and returns empty content
  (`done_reason='length'`). Must be disabled for structured output.
- **Output extraction**: take the first balanced `{...}` block — eliminates the
  sign-inconsistency bug that plagued the multi-agent run (0 vs 21/47).

### Results (single-call, 47 valid origins each)

| Reviser | Directional acc. | Sign-bug | MAE h=1 | MAE h=12 |
|---------|------------------|----------|---------|----------|
| TimesFM C0 prior (no revision) | — | — | **0.4364** | 1.8635 |
| qwen3:4b multi-agent (Exp D) | 34% | 21/47 | 0.4398 | 1.8639 |
| qwen3:4b single-call | 48.7% | 0 | 0.5189 | 1.8617 |
| **qwen3:8b single-call** | **51.3%** | 0 | 0.5242 | 1.8618 |

### Findings

1. **Bigger model barely helps:** 4B → 8B (2× parameters) lifts directional
   accuracy only **48.7% → 51.3% (+2.6 pp)** — both within noise of a coin flip.
   The scaling curve is essentially flat → extrapolating, a 14B/70B would not cross
   into "useful" territory. **Model size is not the bottleneck.**

2. **Architecture & output-handling matter more than size:** fixing the sign-bug
   and using a single clean call lifted accuracy **34% → 49% (+15 pp)** — far more
   than doubling the model (+2.6 pp). The 3-agent chain was actively *hurting*.

3. **Still no MAE win:** at ~50% directional accuracy the applied correction adds
   noise, so single-call MAE (0.52) is *worse* than the prior (0.44) at h=1. The
   reviser cannot improve a strong TSFM prior it cannot out-predict.

**Verdict:** the limitation is **fundamental to zero-shot revision**, confirmed
across model sizes — not a capacity problem. The only path to a real gain is a
*trained* reviser (PostTime's SFT+RLVR), which is out of TFG scope. qwen3:8b was
deleted after the test (re-pull with `ollama pull qwen3:8b`); the cached answers
in `results/nexus_probe_qwen3_8b.json` keep the result reproducible.

---

## Cross-experiment summary

| Experiment | Model | MAE h=1 | MAE h=12 | vs ARIMA h=1 | Key finding |
|------------|-------|---------|----------|-------------|-------------|
| A | MIDAS-ADL monthly proxy | 0.4677 | 2.7863 | **−2.2%** | Beta polynomial helps at h=1 only |
| B | Daily Beta-MIDAS (levels) | 0.4940 | 3.5465 | +3.3% | Daily oil level ≈ redundant with monthly mean |
| B | Daily Beta-MIDAS (returns) | 0.4923 | 3.4030 | +3.0% | Within-month momentum: tiny gain, still < ARIMA |
| B | Ridge-daily (free lag weights) | 0.5065 | 3.5836 | +5.9% | Free lags overfit; ARIMA stronger |
| C | Nexus heuristic (dry-run) | 0.4400 | 1.8634 | −7.9% vs C0 | Bounded correction beats Ridge |
| D | Nexus multi-agent (qwen3:4b) | 0.4398 | 1.8639 | −8.0% vs C0 | Zero-shot LLM ≈ heuristic |
| E | Nexus best gated policy | 0.4569 | 1.8635 | — | No gate beats prior (dir. acc. 34%) |
| F | qwen3:8b single-call reviser | 0.5242 | 1.8618 | — | 2× model size → +2.6pp dir.acc only (51%) |
| ref | TimesFM C0 | 0.4364 | 1.8635 | — | Strong TSFM prior |
| ref | ARIMA | 0.4781 | 1.5410 | — | Statistical baseline |

---

## Methodological lessons

| Finding | Implication |
|---------|-------------|
| MIDAS h=1 −2.2% vs ARIMA with monthly proxies | Beta polynomial constraint has value even without true daily data |
| MIDAS h≥3 degrades vs ARIMA | Direct multi-horizon MIDAS requires daily/weekly input |
| Global MIDAS degrades severely | Pre-aggregated MA3 columns cannot serve as MIDAS input — raw daily Brent needed |
| Nexus heuristic beats C1_inst at all horizons | Bounded correction (±0.30 pp max) more robust than free Ridge |
| Multi-agent LLM ≈ heuristic quality (zero-shot) | LLM reasons correctly but over-revises; PostTime's RLVR insight: *learning when NOT to revise* is the key training signal |
| Agent chain 98% success rate (JSON) | `qwen3:4b` reliably produces structured multi-step JSON output |
| qwen3:4b confidence is always MEDIUM | Local LLM does not calibrate self-reported confidence → cannot be used as a gate |
| Sign inconsistency on 21/47 origins (multi-agent) | Local LLM emits negative `correction_pp` while saying `direction=UP` → structured outputs need validation / robust parsing |
| Qwen3 thinking burns the JSON budget | `/no_think` + `think=False` mandatory for structured output with reasoning models |
| Single-call 49% vs multi-agent 34% dir. acc. | Architecture + output-robustness matter more than the agent chain; the 3-agent design hurt |
| 4B→8B lifts dir. acc. only +2.6pp (49%→51%) | Model size is **not** the bottleneck — limitation is fundamental to zero-shot revision |
| No reviser config beats the TSFM prior on MAE | At ~50% directional accuracy, corrections add noise; only a *trained* reviser (SFT+RLVR) could win |

---

## How to run

```bash
# From tfg-forecasting/

# MIDAS (monthly proxy, ~3 min)
python 09_future_work/01_midas_spain.py
python 09_future_work/02_midas_global.py

# MIDAS (daily WTI + Lasso, ~10 min)
python 09_future_work/01c_midas_daily_spain.py

# Nexus heuristic (instant, no API)
python 09_future_work/03_nexus_reviser_spain.py

# Nexus multi-agent via Ollama (requires Ollama running with qwen3:4b or llama3.2:3b)
# Full 48-origin run ~2.5 hours; cached, safe to interrupt and resume
python 09_future_work/04_nexus_multiagent_ollama.py

# Nexus post-hoc analysis (gating, sign, directional accuracy) — instant, no LLM
python 09_future_work/05_nexus_analysis.py

# Model-size ablation: single-call reviser with a given Ollama model (cached per model)
python 09_future_work/06_nexus_model_probe.py qwen3:4b
python 09_future_work/06_nexus_model_probe.py qwen3:8b   # ollama pull qwen3:8b first
```

All results land in `09_future_work/results/`.
Nexus traces (full chain-of-thought) in `results/nexus_multiagent_traces.json`.
Gating/directional analysis in `results/nexus_analysis.json`.

---

## Key references

| Paper | Year | arXiv | Role in this folder |
|-------|------|-------|---------------------|
| Ghysels, Santa-Clara & Valkanov | 2004 | — | Original MIDAS |
| Babii, Ghysels & Striaukas (ML-MIDAS) | 2020 | 2005.14057 | Daily news MIDAS, Exp B |
| Nexus (agentic forecasting) | 2026 | 2506.21611 | Exp D architecture |
| PostTime (LLM reviser on TSFM prior) | 2026 | 2402.03885 | Exp C+D direct reference |
| SpecTF (spectral fusion) | 2026 | 2602.01588 | Frequency-domain MIDAS analogue |
| TESS (temporal primitive bottleneck) | 2026 | 2603.12664 | Structured LLM output (Agent 2) |
