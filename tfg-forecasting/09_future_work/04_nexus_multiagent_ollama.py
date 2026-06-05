"""
04_nexus_multiagent_ollama.py  —  Nexus multi-agent reviser via Ollama
=======================================================================

Proper implementation of the Nexus/PostTime two-stage architecture using a
locally-running LLM (Ollama) instead of a closed API.  Three specialist
agents run in sequence, each with a focused role:

  Agent 1  SIGNAL ANALYST  (qwen3:4b / llama3.2:3b)
  -------------------------------------------------------
  Input  : structured macro snapshot at the forecast origin
  Task   : identify which signals are in unusual territory and explain why
  Output : {ecb_regime, energy_regime, uncertainty_regime, signal_summary}

  Agent 2  TRANSMISSION REASONER
  -------------------------------------------------------
  Input  : Agent-1 output + Spain CPI recent path
  Task   : reason about the transmission mechanisms from the identified
           regimes to future CPI (energy->CPI, ECB->demand, EPU->investment)
           and estimate the likely lag (how many months until the shock lands)
  Output : {main_channel, expected_direction, lag_months,
            confidence, reasoning_chain}

  Agent 3  REVISION DECIDER
  -------------------------------------------------------
  Input  : TimesFM C0 prior forecast + Agent-2 assessment
  Task   : decide whether to revise the prior, in which direction, and
           by how much — staying within the bounded correction envelope
  Output : {should_revise, direction, magnitude_pp,
            final_forecast_h1, revision_rationale}

The three outputs are stored in a single chain-of-thought trace per origin.
The final correction applied to the prior is extracted from Agent 3.

Correction envelope (same as 03_nexus_reviser_spain.py):
  delta(h) = direction * magnitude_pp * decay^(h-1)
  decay = 0.70,  max magnitude_pp = 0.30 pp

Ollama configuration
--------------------
  Primary model   : qwen3:4b   (better structured reasoning)
  Fallback model  : llama3.2:3b
  Format          : JSON mode (Ollama `format="json"`)
  Temperature     : 0.1  (near-deterministic for reproducibility)
  Timeout         : 30 s per agent call

Cache
-----
All traces are stored in 09_future_work/results/nexus_multiagent_traces.json.
Already-cached origins are skipped on re-runs.

State-of-art connection
-----------------------
  · Nexus (2026, arXiv:2506.21611): agentic decomposition into macro/micro
    fluctuations + contextual synthesis → our 3-agent chain is the zero-shot,
    locally-runnable version of the same pattern.
  · PostTime (2026, arXiv:2402.03885): TSFM prior + LLM contextual reviser
    trained via SFT+RLVR on TimesX; our script is the *inference* mode of
    the same architecture with an open-source local model.
  · TESS (2026, arXiv:2603.12664): forces the LLM to emit interpretable
    temporal primitives rather than free text → Agent 2's structured
    {direction, lag_months, confidence} output is a TESS-inspired bottleneck.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "09_future_work" / "results"
HORIZONS = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "nexus_multiagent"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

TRACES_CACHE = RESULTS_DIR / "nexus_multiagent_traces.json"

# ── Ollama configuration ───────────────────────────────────────────────────────
PRIMARY_MODEL  = "qwen3:4b"
FALLBACK_MODEL = "llama3.2:3b"
TEMPERATURE    = 0.1
AGENT_TIMEOUT  = 30       # seconds per agent call

# ── Correction parameters ─────────────────────────────────────────────────────
MAX_CORRECTION_PP = 0.30
HORIZON_DECAY     = 0.70


# ── Ollama client ──────────────────────────────────────────────────────────────

def ollama_call(model: str, system: str, user: str, timeout: int = AGENT_TIMEOUT) -> dict | None:
    """
    Call Ollama in JSON mode.  Returns parsed dict or None on failure.
    Falls back to FALLBACK_MODEL if primary fails.
    """
    import ollama

    for m in [model, FALLBACK_MODEL]:
        try:
            resp = ollama.chat(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                format="json",
                options={"temperature": TEMPERATURE, "num_predict": 512},
            )
            text = resp["message"]["content"].strip()
            # Strip thinking tags that qwen3 sometimes emits
            if "<think>" in text:
                text = text.split("</think>")[-1].strip()
            # Strip markdown code fences
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            return json.loads(text)
        except Exception as e:
            logger.debug("Ollama call failed with %s: %s", m, e)
            time.sleep(1)
    return None


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(df: pd.DataFrame, origin: pd.Timestamp) -> dict:
    w = df.loc[:origin]
    cur  = w.iloc[-1]
    prev = w.iloc[-2] if len(w) >= 2 else cur

    def g(col, default=None):
        v = cur.get(col, default)
        return round(float(v), 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    dfr_now  = g("dfr", 0.0)
    dfr_prev = float(prev.get("dfr", dfr_now))
    dfr_trend = "rising" if dfr_now > dfr_prev + 0.01 else (
                "falling" if dfr_now < dfr_prev - 0.01 else "stable")

    brent = g("brent_log")
    brent_mean = float(df["brent_log"].loc[:origin].mean())
    brent_dev  = round((brent - brent_mean) if brent else 0.0, 3)

    epu = g("epu_europe_log")
    epu_hist = df["epu_europe_log"].loc[:origin]
    epu_pct = round(float((epu_hist < epu).mean() * 100), 1) if epu else 50.0

    cpi_now  = g("indice_general", 0.0)
    cpi_prev = round(float(prev.get("indice_general", cpi_now)), 4)
    cpi_mom  = round(cpi_now - cpi_prev, 4) if cpi_now and cpi_prev else 0.0

    # Recent 6-month CPI trend
    recent_cpi = df["indice_general"].loc[:origin].tail(6).values.tolist()
    recent_cpi = [round(x, 2) for x in recent_cpi]

    return {
        "origin_date":    origin.strftime("%Y-%m"),
        "cpi_index":      cpi_now,
        "cpi_mom_change": cpi_mom,
        "cpi_last_6m":    recent_cpi,
        "ecb_dfr":        dfr_now,
        "ecb_dfr_trend":  dfr_trend,
        "brent_log_deviation": brent_dev,
        "brent_label":    ("high" if brent_dev > 0.1 else ("low" if brent_dev < -0.1 else "near_average")),
        "epu_europe_log": epu,
        "epu_percentile": epu_pct,
        "epu_label":      ("very_high" if epu_pct > 80 else ("high" if epu_pct > 60 else (
                           "low" if epu_pct < 30 else "normal"))),
        "gdelt_tone_ma3": g("gdelt_tone_ma3"),
        "bce_shock_score": g("bce_shock_score"),
        "bce_tone":        str(cur.get("bce_tone", "unknown")),
        "ine_surprise_score": g("ine_surprise_score"),
    }


# ── Agent 1: Signal Analyst ────────────────────────────────────────────────────

AGENT1_SYSTEM = (
    "You are a macroeconomic signal analyst. "
    "Given a structured snapshot of key indicators at a specific date, "
    "identify the regime of each signal and write a concise summary. "
    "Respond ONLY with a valid JSON object, no extra text."
)

AGENT1_USER = """\
Date: {origin_date}

Signals:
- ECB deposit rate: {ecb_dfr}% (trend: {ecb_dfr_trend})
- Brent crude log deviation from 2002-{year} mean: {brent_log_deviation} ({brent_label})
- EPU Europe percentile: {epu_percentile}% ({epu_label})
- GDELT news tone MA3: {gdelt_tone_ma3}
- BCE shock score: {bce_shock_score}
- BCE communication tone: {bce_tone}
- INE surprise score: {ine_surprise_score}

Classify each signal regime and summarise the overall macro environment.

Required JSON format:
{{
  "ecb_regime": "tightening" | "easing" | "on_hold" | "emergency_cut",
  "energy_regime": "shock_high" | "above_avg" | "normal" | "below_avg" | "shock_low",
  "uncertainty_regime": "crisis" | "elevated" | "normal" | "calm",
  "sentiment_regime": "hawkish" | "neutral" | "dovish" | "unknown",
  "signal_summary": "<2 sentences describing the overall macro environment>"
}}
"""


def run_agent1(ctx: dict, model: str) -> dict | None:
    user = AGENT1_USER.format(
        origin_date=ctx["origin_date"],
        year=ctx["origin_date"][:4],
        ecb_dfr=ctx["ecb_dfr"],
        ecb_dfr_trend=ctx["ecb_dfr_trend"],
        brent_log_deviation=ctx["brent_log_deviation"],
        brent_label=ctx["brent_label"],
        epu_percentile=ctx["epu_percentile"],
        epu_label=ctx["epu_label"],
        gdelt_tone_ma3=ctx.get("gdelt_tone_ma3", "n/a"),
        bce_shock_score=ctx.get("bce_shock_score", "n/a"),
        bce_tone=ctx.get("bce_tone", "unknown"),
        ine_surprise_score=ctx.get("ine_surprise_score", "n/a"),
    )
    return ollama_call(model, AGENT1_SYSTEM, user)


# ── Agent 2: Transmission Reasoner ────────────────────────────────────────────

AGENT2_SYSTEM = (
    "You are an inflation transmission economist. "
    "Given macro signal regimes and a recent CPI trajectory, reason about "
    "how the current environment will affect CPI over the next 1-12 months. "
    "Respond ONLY with a valid JSON object, no extra text."
)

AGENT2_USER = """\
Forecast origin: {origin_date}
Recent Spain CPI (last 6 months): {cpi_last_6m}
Last monthly CPI change: {cpi_mom_change:+.3f} pp

Signal regime assessment:
- ECB: {ecb_regime}
- Energy: {energy_regime}
- Uncertainty: {uncertainty_regime}
- Communication sentiment: {sentiment_regime}
- Summary: {signal_summary}

Reason about the transmission of these regimes to Spain CPI over the next 12 months.
Consider: energy pass-through (1-3 month lag), ECB tightening (6-12 month lag),
uncertainty -> demand compression (3-6 months).

Required JSON format:
{{
  "main_channel": "energy" | "monetary" | "uncertainty" | "mixed" | "none",
  "expected_direction": "UP" | "DOWN" | "NEUTRAL",
  "peak_effect_months": <integer 1-12>,
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "expected_magnitude_pp": <float, expected deviation from baseline in pp>,
  "reasoning_chain": "<3-4 sentences of step-by-step reasoning>"
}}
"""


def run_agent2(ctx: dict, agent1_out: dict, model: str) -> dict | None:
    user = AGENT2_USER.format(
        origin_date=ctx["origin_date"],
        cpi_last_6m=ctx["cpi_last_6m"],
        cpi_mom_change=ctx["cpi_mom_change"],
        ecb_regime=agent1_out.get("ecb_regime", "unknown"),
        energy_regime=agent1_out.get("energy_regime", "unknown"),
        uncertainty_regime=agent1_out.get("uncertainty_regime", "unknown"),
        sentiment_regime=agent1_out.get("sentiment_regime", "unknown"),
        signal_summary=agent1_out.get("signal_summary", ""),
    )
    return ollama_call(model, AGENT2_SYSTEM, user)


# ── Agent 3: Revision Decider ──────────────────────────────────────────────────

AGENT3_SYSTEM = (
    "You are a forecast revision specialist. "
    "Given a statistical model's baseline forecast and a macro transmission assessment, "
    "decide whether the baseline should be revised and by how much. "
    "The maximum allowed revision is +/-0.30 percentage points for h=1. "
    "Be conservative: only revise if the transmission assessment is highly confident. "
    "Respond ONLY with a valid JSON object, no extra text."
)

AGENT3_USER = """\
Origin: {origin_date}
TimesFM C0 baseline forecast h=1: {prior_h1:.4f}

Transmission assessment:
- Main channel: {main_channel}
- Expected direction: {expected_direction}
- Peak effect at: {peak_effect_months} months
- Confidence: {confidence}
- Expected magnitude: {expected_magnitude_pp:.3f} pp
- Reasoning: {reasoning_chain}

Should the baseline be revised for h=1? If yes, by how much?
Maximum allowed: +/-0.30 pp. Be conservative with LOW confidence.

Required JSON format:
{{
  "should_revise": true | false,
  "direction": "UP" | "DOWN" | "NEUTRAL",
  "correction_pp": <float in range [-0.30, +0.30]>,
  "final_forecast_h1": <float>,
  "revision_rationale": "<1-2 sentences explaining the revision decision>"
}}
"""


def run_agent3(ctx: dict, agent2_out: dict, prior_h1: float, model: str) -> dict | None:
    user = AGENT3_USER.format(
        origin_date=ctx["origin_date"],
        prior_h1=prior_h1,
        main_channel=agent2_out.get("main_channel", "none"),
        expected_direction=agent2_out.get("expected_direction", "NEUTRAL"),
        peak_effect_months=agent2_out.get("peak_effect_months", 3),
        confidence=agent2_out.get("confidence", "LOW"),
        expected_magnitude_pp=agent2_out.get("expected_magnitude_pp", 0.0) or 0.0,
        reasoning_chain=agent2_out.get("reasoning_chain", ""),
    )
    return ollama_call(model, AGENT3_SYSTEM, user)


# ── Full agent chain ───────────────────────────────────────────────────────────

def run_agent_chain(ctx: dict, prior_h1: float, model: str) -> dict:
    """Run all 3 agents and return the full trace."""
    trace = {
        "origin": ctx["origin_date"],
        "prior_h1": prior_h1,
        "agent1": None,
        "agent2": None,
        "agent3": None,
        "final_correction_pp": 0.0,
        "final_direction": "NEUTRAL",
    }

    a1 = run_agent1(ctx, model)
    trace["agent1"] = a1
    if a1 is None:
        return trace

    a2 = run_agent2(ctx, a1, model)
    trace["agent2"] = a2
    if a2 is None:
        return trace

    a3 = run_agent3(ctx, a2, prior_h1, model)
    trace["agent3"] = a3
    if a3 is None:
        return trace

    # Extract correction — clamp to [-MAX, +MAX]
    correction = float(a3.get("correction_pp", 0.0) or 0.0)
    correction = max(-MAX_CORRECTION_PP, min(MAX_CORRECTION_PP, correction))
    if not a3.get("should_revise", False):
        correction = 0.0

    trace["final_correction_pp"] = round(correction, 4)
    trace["final_direction"] = a3.get("direction", "NEUTRAL")
    return trace


# ── Load data ──────────────────────────────────────────────────────────────────

def load_prior_predictions() -> pd.DataFrame:
    p = ROOT / "08_results" / "timesfm_C0_predictions.parquet"
    df = pd.read_parquet(p)
    df["origin"]  = pd.to_datetime(df["origin"])
    df["fc_date"] = pd.to_datetime(df["fc_date"])
    return df


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    return df


# ── Rolling loop ───────────────────────────────────────────────────────────────

def run_rolling(df_prior: pd.DataFrame, df_feat: pd.DataFrame, model: str) -> pd.DataFrame:
    # Load or initialise trace cache
    traces: dict[str, dict] = {}
    if TRACES_CACHE.exists():
        traces = json.load(open(TRACES_CACHE, encoding="utf-8"))

    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    records = []
    new_traces = 0

    for origin in tqdm(origins, desc="Nexus multi-agent rolling"):
        origin_key = origin.strftime("%Y-%m")

        ctx = build_context(df_feat, origin)

        if origin_key not in traces:
            prior_h1_rows = df_prior[(df_prior["origin"] == origin) & (df_prior["horizon"] == 1)]
            prior_h1 = float(prior_h1_rows["y_pred"].mean()) if not prior_h1_rows.empty else np.nan

            if np.isnan(prior_h1):
                traces[origin_key] = {"origin": origin_key, "final_correction_pp": 0.0,
                                      "final_direction": "NEUTRAL", "agent1": None,
                                      "agent2": None, "agent3": None}
            else:
                trace = run_agent_chain(ctx, prior_h1, model)
                traces[origin_key] = trace
                new_traces += 1

            # Persist after every new call
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(TRACES_CACHE, "w", encoding="utf-8") as f:
                json.dump(traces, f, indent=2, ensure_ascii=False)

        trace = traces[origin_key]
        correction_h1 = float(trace.get("final_correction_pp", 0.0) or 0.0)

        for h in HORIZONS:
            h_preds = df_prior[(df_prior["origin"] == origin) & (df_prior["horizon"] == h)]
            if h_preds.empty:
                continue
            correction = correction_h1 * (HORIZON_DECAY ** (h - 1))
            for _, row in h_preds.iterrows():
                y_pred_rev = float(row["y_pred"]) + correction
                y_true = float(row["y_true"])
                records.append({
                    "origin":       origin,
                    "fc_date":      row["fc_date"],
                    "step":         int(row["step"]),
                    "horizon":      h,
                    "model":        MODEL_NAME,
                    "y_true":       y_true,
                    "y_pred":       y_pred_rev,
                    "y_pred_prior": float(row["y_pred"]),
                    "correction":   correction,
                    "direction":    trace.get("final_direction", "NEUTRAL"),
                    "error":        y_true - y_pred_rev,
                    "abs_error":    abs(y_true - y_pred_rev),
                })

    logger.info("New agent traces computed: %d  |  Total cached: %d", new_traces, len(traces))
    return pd.DataFrame(records)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    res = {}
    for h in HORIZONS:
        hd = df_preds[df_preds["horizon"] == h]
        if hd.empty:
            continue
        yt, yp = hd["y_true"].values, hd["y_pred"].values
        ypp = hd["y_pred_prior"].values
        mae  = float(np.mean(np.abs(yt - yp)))
        maep = float(np.mean(np.abs(yt - ypp)))
        res[f"h{h}"] = {
            "MAE":  round(mae, 4),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp)**2))), 4),
            "MASE": round(mae / mase_scale, 4),
            "MAE_prior": round(maep, 4),
            "delta_vs_prior_pct": round((mae - maep) / maep * 100, 2),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


def log_agent_quality(traces: dict) -> None:
    """Print summary statistics about the agent chain quality."""
    a1_ok = sum(1 for t in traces.values() if t.get("agent1") is not None)
    a2_ok = sum(1 for t in traces.values() if t.get("agent2") is not None)
    a3_ok = sum(1 for t in traces.values() if t.get("agent3") is not None)
    revised = sum(1 for t in traces.values()
                  if t.get("agent3", {}) and t["agent3"].get("should_revise", False))
    n = len(traces)

    logger.info("\n--- Agent chain quality (%d origins) ---", n)
    logger.info("  Agent 1 success  : %d/%d (%.0f%%)", a1_ok, n, 100*a1_ok/n if n else 0)
    logger.info("  Agent 2 success  : %d/%d (%.0f%%)", a2_ok, n, 100*a2_ok/n if n else 0)
    logger.info("  Agent 3 success  : %d/%d (%.0f%%)", a3_ok, n, 100*a3_ok/n if n else 0)
    logger.info("  Revisions decided: %d/%d (%.0f%%)", revised, n, 100*revised/n if n else 0)

    dirs = [t.get("final_direction","NEUTRAL") for t in traces.values()]
    from collections import Counter
    logger.info("  Direction counts : %s", dict(Counter(dirs)))

    # Sample a few traces to show reasoning quality
    logger.info("\n--- Sample reasoning trace (last origin) ---")
    last = sorted(traces.items())[-1][1] if traces else {}
    a2 = last.get("agent2") or {}
    a3 = last.get("agent3") or {}
    if a2:
        logger.info("  Agent 2 channel     : %s | direction: %s | confidence: %s",
                    a2.get("main_channel","?"), a2.get("expected_direction","?"),
                    a2.get("confidence","?"))
        logger.info("  Agent 2 reasoning   : %s", a2.get("reasoning_chain","")[:200])
    if a3:
        logger.info("  Agent 3 correction  : %+.4f pp | revised: %s",
                    last.get("final_correction_pp", 0), a3.get("should_revise", "?"))
        logger.info("  Agent 3 rationale   : %s", a3.get("revision_rationale","")[:200])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 65)
    logger.info("NEXUS MULTI-AGENT REVISER  —  Spain CPI")
    logger.info("LLM backend   : Ollama (%s -> fallback %s)", PRIMARY_MODEL, FALLBACK_MODEL)
    logger.info("Pipeline      : SignalAnalyst -> TransmissionReasoner -> RevisionDecider")
    logger.info("Max correction: +/-%.2f pp  |  horizon decay gamma=%.2f",
                MAX_CORRECTION_PP, HORIZON_DECAY)
    logger.info("=" * 65)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_prior = load_prior_predictions()
    df_feat  = load_features()
    logger.info("Prior predictions: %d rows, %d origins",
                len(df_prior), df_prior["origin"].nunique())

    y_train = df_feat["indice_general"].loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(y_train.values[12:] - y_train.values[:-12])))

    df_preds = run_rolling(df_prior, df_feat, model=PRIMARY_MODEL)

    if df_preds.empty:
        logger.warning("[!] No revised predictions generated.")
        return

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n%-12s %8s %8s %8s  %10s %8s", "Horizon", "MAE", "RMSE", "MASE",
                "MAE_prior", "delta%")
    logger.info("-" * 60)
    for h in HORIZONS:
        k = "h%d" % h
        if k in metrics:
            m = metrics[k]
            logger.info("h=%-10d %8.4f %8.4f %8.4f  %10.4f %+7.1f%%",
                        h, m["MAE"], m["RMSE"], m["MASE"],
                        m["MAE_prior"], m["delta_vs_prior_pct"])

    # Load traces for quality summary
    if TRACES_CACHE.exists():
        traces = json.load(open(TRACES_CACHE, encoding="utf-8"))
        log_agent_quality(traces)

    # Comparison vs simple reviser
    dry_path = RESULTS_DIR / "nexus_reviser_dry_metrics.json"
    if dry_path.exists():
        dry = json.load(open(dry_path)).get("nexus_reviser_dry", {})
        logger.info("\nMulti-agent Nexus vs single-step heuristic (dry-run):")
        for h in HORIZONS:
            m_ma = metrics.get("h%d" % h, {}).get("MAE")
            m_dry = dry.get("h%d" % h, {}).get("MAE")
            if m_ma and m_dry:
                logger.info("  h=%d: multi-agent=%.4f  dry-run=%.4f  delta=%+.1f%%",
                            h, m_ma, m_dry, (m_ma - m_dry) / m_dry * 100)

    df_preds.to_parquet(RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet", index=False)
    with open(RESULTS_DIR / f"{MODEL_NAME}_metrics.json", "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    logger.info("\nSaved: %s_metrics.json", MODEL_NAME)
    logger.info("Traces: %s", TRACES_CACHE.name)


if __name__ == "__main__":
    main()
