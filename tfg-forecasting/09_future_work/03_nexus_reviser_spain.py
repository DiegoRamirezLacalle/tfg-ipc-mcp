"""
03_nexus_reviser_spain.py  —  Nexus-inspired LLM contextual reviser
====================================================================

Architecture: TSFM strong prior  +  LLM contextual revision

Inspired by two 2026 papers:

  · **PostTime** (arXiv:2402.03885) — trains a language model to revise,
    preserve or reject the prior from a frozen TSFM based on multimodal
    context.  Uses SFT + RLVR on the TimesX benchmark with Gemma-3-4B
    and TimesFM-2.5.

  · **Nexus** — agentic forecasting that decomposes macro/micro
    fluctuations, integrates contextual reasoning, and produces a final
    prediction with an explicit reasoning trace.

This script implements the same two-stage logic without end-to-end
training (zero-shot revision):

  Stage 1 — PRIOR:  TimesFM C0 predictions (already computed and stored
            in 08_results/timesfm_C0_predictions.parquet).

  Stage 2 — REVISION:  At each rolling origin, a structured context
            summary is built from the institutional signals available
            at that date.  A Claude model evaluates the context and
            returns a structured JSON assessment:

              {
                "direction":  "UP" | "DOWN" | "NEUTRAL",
                "confidence": "LOW" | "MEDIUM" | "HIGH",
                "magnitude":  "SMALL" | "MEDIUM" | "LARGE",
                "reasoning":  "< 2 sentences >"
              }

            A bounded correction proportional to confidence × magnitude
            is applied to the h=1 prior.  For longer horizons the
            correction is damped geometrically (less certainty about
            the future macro path).

  Stage 3 — CACHE:  All assessments are persisted to
            09_future_work/results/nexus_spain_assessments.json so the
            experiment is fully reproducible without re-calling the API.

DRY_RUN flag
------------
Set DRY_RUN = True  to run without any API calls.  A deterministic
rule-based heuristic replaces the LLM call (useful for CI, demos, and
cost control).  The heuristic encodes exactly the macro logic the LLM
is asked to reason about, so the two tracks are methodologically
comparable.

Set DRY_RUN = False to use the real Claude API (requires ANTHROPIC_API_KEY).
Already-cached origins are skipped — safe to run incrementally.

Parameters
----------
MAX_CORRECTION_H1 = 0.30 pp  — max h=1 correction (±0.30 pp)
The h=k correction is  MAX_CORRECTION_H1 × γ^(k-1)  with γ=0.7.
This ensures the LLM cannot override the TSFM by more than ≈1 pp at h=12
and reflects decreasing confidence at longer horizons.
"""

from __future__ import annotations

import json
import os
import sys
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
MODEL_NAME_DRY = "nexus_reviser_dry"
MODEL_NAME_LLM = "nexus_reviser_llm"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

# ── Tuneable parameters ────────────────────────────────────────────────────────
DRY_RUN = True                # set False to use real Claude API
CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for structured extraction
MAX_CORRECTION_H1 = 0.30     # pp — maximum correction at h=1
HORIZON_DECAY = 0.70         # γ: correction at h decays as MAX × γ^(h-1)

# Direction / confidence / magnitude → scalar multipliers
DIRECTION_MAP  = {"UP": +1.0, "DOWN": -1.0, "NEUTRAL": 0.0}
CONFIDENCE_MAP = {"LOW": 0.25, "MEDIUM": 0.55, "HIGH": 0.85}
MAGNITUDE_MAP  = {"SMALL": 0.30, "MEDIUM": 0.65, "LARGE": 1.00}

ASSESSMENTS_CACHE = RESULTS_DIR / "nexus_spain_assessments.json"


# ── Context builder ────────────────────────────────────────────────────────────

def _percentile_label(value: float, series: pd.Series) -> str:
    pct = (series < value).mean() * 100
    if pct < 20:
        return f"very low (p{pct:.0f})"
    if pct < 40:
        return f"below average (p{pct:.0f})"
    if pct < 60:
        return f"around average (p{pct:.0f})"
    if pct < 80:
        return f"above average (p{pct:.0f})"
    return f"very high (p{pct:.0f})"


def build_context(df: pd.DataFrame, origin: pd.Timestamp) -> dict:
    """Return a structured dict describing the macro situation at `origin`."""
    w = df.loc[:origin]
    prev = w.iloc[-2] if len(w) >= 2 else w.iloc[-1]
    cur  = w.iloc[-1]

    def get(col, default=np.nan):
        return float(cur.get(col, default)) if col in cur.index else default

    def get_prev(col, default=np.nan):
        return float(prev.get(col, default)) if col in prev.index else default

    # ECB rate and trend
    dfr_now  = get("dfr")
    dfr_prev = get_prev("dfr")
    dfr_trend = "rising" if dfr_now > dfr_prev + 0.01 else (
                "falling" if dfr_now < dfr_prev - 0.01 else "stable")

    # Brent vs historical mean
    brent_now = get("brent_log")
    brent_hist_mean = float(df["brent_log"].loc[:origin].mean())
    brent_dev = brent_now - brent_hist_mean

    # EPU level and percentile
    epu_now = get("epu_europe_log")
    epu_pct_lbl = _percentile_label(epu_now, df["epu_europe_log"].loc[:origin])

    # GDELT tone
    gdelt_now = get("gdelt_avg_tone", np.nan)
    gdelt_ma3 = get("gdelt_tone_ma3", np.nan)

    # BCE MCP signals
    bce_shock = get("bce_shock_score", np.nan)
    bce_tone  = cur.get("bce_tone", "neutral") if "bce_tone" in cur.index else "unknown"

    # INE MCP signals
    ine_surprise = get("ine_surprise_score", np.nan)

    # Last known CPI value
    cpi_now = get("indice_general")
    cpi_prev = get_prev("indice_general")
    cpi_mom = cpi_now - cpi_prev

    return {
        "origin_date":    origin.strftime("%Y-%m"),
        "cpi_index":      round(cpi_now, 2),
        "cpi_mom_change": round(cpi_mom, 3),
        "ecb_dfr":        round(dfr_now, 2),
        "ecb_dfr_trend":  dfr_trend,
        "brent_log":      round(brent_now, 3),
        "brent_vs_hist":  f"{brent_dev:+.3f} log-pp vs 2002-{origin.year} mean",
        "epu_europe_log": round(epu_now, 3),
        "epu_level":      epu_pct_lbl,
        "gdelt_avg_tone": round(gdelt_now, 2) if not np.isnan(gdelt_now) else None,
        "gdelt_tone_ma3": round(gdelt_ma3, 2) if not np.isnan(gdelt_ma3) else None,
        "bce_shock_score": round(bce_shock, 3) if not np.isnan(bce_shock) else None,
        "bce_tone":        bce_tone,
        "ine_surprise_score": round(ine_surprise, 3) if not np.isnan(ine_surprise) else None,
    }


# ── Dry-run heuristic (no API call) ───────────────────────────────────────────

def heuristic_assessment(ctx: dict) -> dict:
    """
    Rule-based simulation of the LLM's contextual assessment.
    Encodes the same macro logic as the LLM prompt so results are comparable.
    Rules are intentionally simple and transparent.
    """
    score = 0.0  # positive → inflationary pressure, negative → disinflationary

    # ECB tightening → disinflationary with lag
    if ctx["ecb_dfr"] > 3.5:
        score -= 0.4
    elif ctx["ecb_dfr"] > 1.5:
        score -= 0.2
    if ctx["ecb_dfr_trend"] == "rising":
        score -= 0.2
    elif ctx["ecb_dfr_trend"] == "falling":
        score += 0.15

    # Brent above historical average → inflationary
    brent_dev_raw = float(ctx["brent_vs_hist"].split()[0])
    if brent_dev_raw > 0.15:
        score += 0.4
    elif brent_dev_raw > 0.05:
        score += 0.2
    elif brent_dev_raw < -0.15:
        score -= 0.3

    # High EPU → uncertainty, mild upward bias via imported costs
    if "very high" in ctx["epu_level"]:
        score += 0.15
    elif "very low" in ctx["epu_level"]:
        score -= 0.1

    # GDELT tone: negative sentiment → growth fears → disinflationary
    if ctx.get("gdelt_avg_tone") is not None:
        if ctx["gdelt_avg_tone"] < -2.5:
            score -= 0.2
        elif ctx["gdelt_avg_tone"] > 0.0:
            score += 0.1

    # BCE shock signals
    if ctx.get("bce_shock_score") is not None:
        if ctx["bce_shock_score"] > 0.5:
            score += 0.25
        elif ctx["bce_shock_score"] < -0.3:
            score -= 0.2

    # INE surprise
    if ctx.get("ine_surprise_score") is not None:
        if ctx["ine_surprise_score"] > 0.3:
            score += 0.15
        elif ctx["ine_surprise_score"] < -0.3:
            score -= 0.15

    # Map score to structured assessment
    if abs(score) < 0.15:
        direction = "NEUTRAL"
    elif score > 0:
        direction = "UP"
    else:
        direction = "DOWN"

    abs_s = abs(score)
    confidence = "LOW" if abs_s < 0.3 else ("MEDIUM" if abs_s < 0.55 else "HIGH")
    magnitude  = "SMALL" if abs_s < 0.3 else ("MEDIUM" if abs_s < 0.6 else "LARGE")

    return {
        "direction": direction,
        "confidence": confidence,
        "magnitude": magnitude,
        "reasoning": (
            f"[DRY RUN — heuristic] Score={score:+.2f}. "
            f"Key drivers: ECB DFR {ctx['ecb_dfr']:.2f}% ({ctx['ecb_dfr_trend']}), "
            f"Brent {ctx['brent_vs_hist']}, EPU {ctx['epu_level']}."
        ),
        "source": "heuristic",
    }


# ── Real LLM assessment (Claude API) ──────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a monetary policy analyst specialising in euro-area inflation. "
    "You receive a structured snapshot of macro and news signals at a given "
    "forecast origin and a statistical model's h=1 forecast for Spain CPI. "
    "Your task is to assess whether the current macro context suggests the "
    "realised CPI will be above, below, or in line with the statistical forecast. "
    "Respond ONLY with a JSON object — no explanation outside the JSON."
)

USER_PROMPT_TEMPLATE = """\
Forecast origin: {origin_date}

Macro context at origin:
- Spain CPI index (last known): {cpi_index}  |  MoM change: {cpi_mom_change:+.3f} pp
- ECB deposit-facility rate: {ecb_dfr}%  ({ecb_dfr_trend})
- Brent crude (log, monthly): {brent_log}  |  vs 2002–{year} mean: {brent_vs_hist}
- EPU Europe (log): {epu_europe_log}  |  level: {epu_level}
- GDELT news tone (ECB press, MA3): {gdelt_tone}
- BCE communication: shock_score={bce_shock}  |  tone={bce_tone}
- INE surprise score: {ine_surprise}

Statistical model (TimesFM C0, h=1) forecast: {prior_h1:.4f}

Given this context, will realised h=1 CPI be above, below, or in line with the forecast?

Respond with exactly this JSON structure:
{{
  "direction":  "UP" | "DOWN" | "NEUTRAL",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "magnitude":  "SMALL" | "MEDIUM" | "LARGE",
  "reasoning":  "< two sentences explaining the main driver >"
}}
"""


def llm_assessment(ctx: dict, prior_h1: float) -> dict:
    """Call Claude to assess the macro context. Returns structured dict."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Set DRY_RUN=True or pip install anthropic.")
        return heuristic_assessment(ctx)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("[!] ANTHROPIC_API_KEY not set — falling back to heuristic.")
        return heuristic_assessment(ctx)

    client = anthropic.Anthropic(api_key=api_key)
    user_msg = USER_PROMPT_TEMPLATE.format(
        origin_date=ctx["origin_date"],
        cpi_index=ctx["cpi_index"],
        cpi_mom_change=ctx["cpi_mom_change"],
        ecb_dfr=ctx["ecb_dfr"],
        ecb_dfr_trend=ctx["ecb_dfr_trend"],
        brent_log=ctx["brent_log"],
        brent_vs_hist=ctx["brent_vs_hist"],
        year=ctx["origin_date"][:4],
        epu_europe_log=ctx["epu_europe_log"],
        epu_level=ctx["epu_level"],
        gdelt_tone=ctx.get("gdelt_tone_ma3", "n/a"),
        bce_shock=ctx.get("bce_shock_score", "n/a"),
        bce_tone=ctx.get("bce_tone", "n/a"),
        ine_surprise=ctx.get("ine_surprise_score", "n/a"),
        prior_h1=prior_h1,
    )

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        # Extract JSON even if wrapped in markdown code fences
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        result = json.loads(text)
        # Validate required fields
        for field in ["direction", "confidence", "magnitude"]:
            if field not in result or result[field] not in \
                    {"UP", "DOWN", "NEUTRAL"} | {"LOW", "MEDIUM", "HIGH"} | {"SMALL", "MEDIUM", "LARGE"}:
                raise ValueError(f"Invalid field {field}={result.get(field)}")
        result["source"] = "claude"
        return result
    except Exception as e:
        logger.warning("[!] LLM call failed (%s) — falling back to heuristic.", e)
        return heuristic_assessment(ctx)


# ── Correction application ────────────────────────────────────────────────────

def compute_correction(assessment: dict, h: int) -> float:
    """
    Bounded scalar correction for horizon h.

    correction(h) = direction × confidence × magnitude × MAX_H1 × γ^(h-1)

    The decay factor γ = HORIZON_DECAY reflects that macro signals
    inform short-term adjustments more reliably than long-horizon ones.
    """
    d = DIRECTION_MAP.get(assessment.get("direction", "NEUTRAL"), 0.0)
    c = CONFIDENCE_MAP.get(assessment.get("confidence", "LOW"), 0.25)
    m = MAGNITUDE_MAP.get(assessment.get("magnitude", "SMALL"), 0.30)
    decay = HORIZON_DECAY ** (h - 1)
    return d * c * m * MAX_CORRECTION_H1 * decay


# ── Load data ─────────────────────────────────────────────────────────────────

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


# ── Main rolling loop ─────────────────────────────────────────────────────────

def run_rolling(df_prior: pd.DataFrame, df_feat: pd.DataFrame,
                dry_run: bool) -> pd.DataFrame:
    model_name = MODEL_NAME_DRY if dry_run else MODEL_NAME_LLM

    # Load or initialise the assessment cache
    cache: dict[str, dict] = {}
    if ASSESSMENTS_CACHE.exists():
        cache = json.load(open(ASSESSMENTS_CACHE))

    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    records = []

    for origin in tqdm(origins, desc=f"{model_name} rolling"):
        origin_key = origin.strftime("%Y-%m")

        # Build context (always cheap)
        try:
            ctx = build_context(df_feat, origin)
        except Exception as e:
            logger.warning("[!] Context build failed at %s: %s", origin_key, e)
            continue

        # Get assessment (from cache or fresh call)
        if origin_key in cache:
            assessment = cache[origin_key]
        else:
            # Get h=1 prior for this origin
            prior_h1_rows = df_prior[(df_prior["origin"] == origin) & (df_prior["horizon"] == 1)]
            prior_h1 = float(prior_h1_rows["y_pred"].mean()) if not prior_h1_rows.empty else np.nan

            if dry_run:
                assessment = heuristic_assessment(ctx)
            else:
                assessment = llm_assessment(ctx, prior_h1)

            cache[origin_key] = assessment
            # Persist after each new call
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(ASSESSMENTS_CACHE, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

        # Apply correction to prior predictions for this origin
        for h in HORIZONS:
            h_preds = df_prior[(df_prior["origin"] == origin) & (df_prior["horizon"] == h)]
            if h_preds.empty:
                continue
            correction = compute_correction(assessment, h)
            for _, row in h_preds.iterrows():
                y_pred_revised = float(row["y_pred"]) + correction
                y_true = float(row["y_true"])
                records.append({
                    "origin":    origin,
                    "fc_date":   row["fc_date"],
                    "step":      int(row["step"]),
                    "horizon":   h,
                    "model":     model_name,
                    "y_true":    y_true,
                    "y_pred":    y_pred_revised,
                    "y_pred_prior": float(row["y_pred"]),
                    "correction": correction,
                    "direction": assessment.get("direction", "?"),
                    "confidence": assessment.get("confidence", "?"),
                    "error":     y_true - y_pred_revised,
                    "abs_error": abs(y_true - y_pred_revised),
                })

    logger.info("Assessments cached: %d", len(cache))
    return pd.DataFrame(records)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float,
                    model_col: str = "model") -> dict:
    res = {}
    for h in HORIZONS:
        hd = df_preds[df_preds["horizon"] == h]
        if hd.empty:
            continue
        yt, yp = hd["y_true"].values, hd["y_pred"].values
        yp_prior = hd["y_pred_prior"].values
        mae_rev = float(np.mean(np.abs(yt - yp)))
        mae_prior = float(np.mean(np.abs(yt - yp_prior)))
        res[f"h{h}"] = {
            "MAE": round(mae_rev, 4),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp)**2))), 4),
            "MASE": round(mae_rev / mase_scale, 4),
            "MAE_prior": round(mae_prior, 4),
            "delta_vs_prior_pct": round((mae_rev - mae_prior) / mae_prior * 100, 2),
            "n_evals": int(len(hd["origin"].unique())),
        }
    return res


# ── Assessment analysis ───────────────────────────────────────────────────────

def log_assessment_summary(df_preds: pd.DataFrame) -> None:
    if "direction" not in df_preds.columns:
        return
    h1 = df_preds[df_preds["horizon"] == 1].drop_duplicates("origin")
    counts = h1["direction"].value_counts()
    logger.info("\nAssessment directions across 48 origins: %s", dict(counts))
    correct_dir = 0
    total = 0
    for _, row in h1.iterrows():
        true_change = row["y_true"] - row["y_pred_prior"]
        pred_dir = row["direction"]
        actual_dir = "UP" if true_change > 0.05 else ("DOWN" if true_change < -0.05 else "NEUTRAL")
        if pred_dir == actual_dir:
            correct_dir += 1
        total += 1
    if total:
        logger.info("Directional accuracy (h=1, |true_change|>0.05): %.1f%%",
                    100 * correct_dir / total)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model_name = MODEL_NAME_DRY if DRY_RUN else MODEL_NAME_LLM

    logger.info("=" * 65)
    logger.info("NEXUS-INSPIRED REVISER  —  Spain CPI")
    logger.info("Base prior : TimesFM C0")
    logger.info("Revision   : %s", "Heuristic (DRY RUN)" if DRY_RUN else f"Claude ({CLAUDE_MODEL})")
    logger.info("Max correction h=1 : ±%.2f pp  |  decay γ=%.2f", MAX_CORRECTION_H1, HORIZON_DECAY)
    logger.info("=" * 65)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_prior = load_prior_predictions()
    df_feat  = load_features()
    logger.info("Prior predictions: %d rows, %d origins",
                len(df_prior), df_prior["origin"].nunique())

    # Compute MASE scale from training window
    y_train = df_feat["indice_general"].loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(y_train.values[12:] - y_train.values[:-12])))
    logger.info("MASE scale: %.4f pp", mase_scale)

    df_preds = run_rolling(df_prior, df_feat, dry_run=DRY_RUN)
    if df_preds.empty:
        logger.warning("[!] No revised predictions generated.")
        return

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n%-12s %8s %8s %8s   %-10s %-8s", "Horizon", "MAE", "RMSE", "MASE",
                "MAE_prior", "Δ%")
    logger.info("-" * 58)
    for h in HORIZONS:
        k = f"h{h}"
        if k in metrics:
            m = metrics[k]
            logger.info("h=%-10d %8.4f %8.4f %8.4f   %10.4f  %+6.1f%%",
                        h, m["MAE"], m["RMSE"], m["MASE"],
                        m["MAE_prior"], m["delta_vs_prior_pct"])

    log_assessment_summary(df_preds)

    # Compare vs stored TimesFM C1_inst
    c1_path = ROOT / "08_results" / "timesfm_C1_inst_metrics.json"
    if c1_path.exists():
        c1 = json.load(open(c1_path)).get("timesfm_C1_inst", {})
        logger.info("\nΔ Nexus-revised vs TimesFM C1_inst:")
        for h in HORIZONS:
            m_rev  = metrics.get(f"h{h}", {}).get("MAE")
            m_c1   = c1.get(f"h{h}", {}).get("MAE")
            if m_rev and m_c1:
                logger.info("  h=%d: %+.1f%%", h, (m_rev - m_c1) / m_c1 * 100)

    # Save
    df_preds.to_parquet(RESULTS_DIR / f"{model_name}_predictions.parquet", index=False)
    out = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(out, "w") as f:
        json.dump({model_name: metrics}, f, indent=2)
    logger.info("\nSaved: %s", out.name)
    logger.info("Assessments: %s", ASSESSMENTS_CACHE.name)


if __name__ == "__main__":
    main()
