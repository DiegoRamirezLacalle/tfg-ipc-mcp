"""
06_nexus_model_probe.py  —  Does a bigger local LLM fix the Nexus reviser?
===========================================================================

The 3-agent pipeline (04) with qwen3:4b produced fluent reasoning but only
34% directional accuracy (anti-informative).  Question: is that a *small-model*
limitation, or a *fundamental zero-shot* limitation?

This probe answers it cheaply.  Instead of the full 3-agent chain (3 LLM calls
× ~3 min × 48 origins = hours), it issues ONE consolidated reviser call per
origin and measures the only thing that matters for a MAE win — whether the
model's stated direction matches the realised sign of the TimesFM prior's
error.  Architecture is held fixed (single call); only the MODEL changes, so
the comparison is a clean model-size ablation.

Run it for several models (set MODEL or pass as argv[1]); results are cached
per model in results/nexus_probe_<model>.json so re-runs are instant.

Metrics reported per model:
  · directional accuracy (stated dir vs sign(y_true_h1 - prior_h1), dead-band ±0.05)
  · sign-consistency failures (direction=UP but magnitude<0)
  · MAE at h=1..12 if the bounded correction is applied to the TimesFM C0 prior

Baseline to beat: qwen3:4b multi-agent → directional accuracy 34%, MAE h1 0.4398.

Usage:
    python 09_future_work/06_nexus_model_probe.py qwen3:8b
    python 09_future_work/06_nexus_model_probe.py qwen3:4b      # control
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))
sys.path.insert(0, str(ROOT / "09_future_work"))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

# Reuse the validated context builder from the multi-agent script
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "nexus_ma", str(ROOT / "09_future_work" / "04_nexus_multiagent_ollama.py"))
_ma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ma)
build_context = _ma.build_context

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "09_future_work" / "results"
HORIZONS = [1, 3, 6, 12]
HORIZON_DECAY = 0.70
MAX_CORR = 0.30
DEADBAND = 0.05
TEMPERATURE = 0.1

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"

DIR_SIGN = {"UP": +1.0, "DOWN": -1.0, "NEUTRAL": 0.0}

SYSTEM = (
    "You are a euro-area inflation analyst. You receive a macro snapshot at a "
    "forecast origin and a statistical model's one-month-ahead CPI forecast. "
    "Decide whether the realised CPI will come in ABOVE, BELOW, or IN LINE with "
    "that forecast, and by how much. Reason about energy pass-through (1-3 mo), "
    "ECB tightening (6-12 mo) and uncertainty (3-6 mo). "
    "Respond with ONLY a JSON object, no extra text."
)

USER_TMPL = """\
Origin: {origin_date}
TimesFM baseline h=1 forecast: {prior:.4f}

Macro snapshot:
- Spain CPI (last): {cpi_index}  MoM change: {cpi_mom:+.3f}
- ECB deposit rate: {dfr}% ({dfr_trend})
- Brent log dev vs hist mean: {brent_dev} ({brent_label})
- EPU Europe percentile: {epu_pct}% ({epu_label})
- GDELT tone MA3: {gdelt}
- BCE shock score: {bce_shock} | tone: {bce_tone}

Will realised CPI be above/below/in line with the baseline?

JSON only:
{{"direction":"UP|DOWN|NEUTRAL","confidence":"LOW|MEDIUM|HIGH",
 "magnitude_pp":<signed float, your best estimate of (realised - baseline)>,
 "reasoning":"<one sentence>"}}
"""


def ollama_json(model: str, system: str, user: str) -> dict | None:
    """
    Call Ollama in JSON mode with thinking DISABLED.

    Qwen3 models think by default; with format=json the reasoning consumes the
    whole token budget and the JSON answer never appears (done_reason='length',
    empty content). We disable it two ways for robustness:
      · `/no_think` appended to both messages (Qwen3 soft switch), and
      · `think=False` kwarg (newer ollama-python; ignored via fallback if absent).
    A generous num_predict leaves room for the answer either way.
    """
    import re
    import ollama
    sys_msg = system + " /no_think"
    usr_msg = user + "\n/no_think"
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": usr_msg}]
    # Generous budget: even if the model leaks some thinking, the JSON still fits.
    opts = {"temperature": TEMPERATURE, "num_predict": 1200}
    try:
        try:
            r = ollama.chat(model=model, messages=msgs, format="json",
                            options=opts, think=False)
        except TypeError:
            r = ollama.chat(model=model, messages=msgs, format="json", options=opts)
        txt = r["message"]["content"].strip()
        if "<think>" in txt:
            txt = txt.split("</think>")[-1].strip()
        if not txt:
            return None
        # Robust extraction: take the first balanced {...} object in the text.
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            return json.loads(m.group(0)) if m else None
    except Exception as e:
        logger.warning("call failed: %s", e)
        return None


def load_prior() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "08_results" / "timesfm_C0_predictions.parquet")
    d["origin"] = pd.to_datetime(d["origin"])
    d["fc_date"] = pd.to_datetime(d["fc_date"])
    return d


def load_features() -> pd.DataFrame:
    f = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f = f.set_index("date")
    f.index.freq = "MS"
    return f


def main() -> None:
    logger.info("=" * 64)
    logger.info("NEXUS MODEL PROBE  —  single-call reviser  —  model=%s", MODEL)
    logger.info("=" * 64)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = RESULTS_DIR / f"nexus_probe_{MODEL.replace(':', '_')}.json"
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}

    prior = load_prior()
    feat = load_features()
    origins = pd.date_range("2021-01-01", DATE_TEST_END, freq="MS")

    t0 = time.time()
    new = 0
    for oi, origin in enumerate(origins):
        ok = origin.strftime("%Y-%m")
        if ok in cache and cache[ok].get("answer") is not None:
            continue   # keep valid answers; retry NULLs from earlier runs
        ctx = build_context(feat, origin)
        ph = prior[(prior["origin"] == origin) & (prior["horizon"] == 1)]
        prior_h1 = float(ph["y_pred"].mean()) if not ph.empty else np.nan
        if np.isnan(prior_h1):
            continue
        user = USER_TMPL.format(
            origin_date=ctx["origin_date"], prior=prior_h1,
            cpi_index=ctx["cpi_index"], cpi_mom=ctx["cpi_mom_change"],
            dfr=ctx["ecb_dfr"], dfr_trend=ctx["ecb_dfr_trend"],
            brent_dev=ctx["brent_log_deviation"], brent_label=ctx["brent_label"],
            epu_pct=ctx["epu_percentile"], epu_label=ctx["epu_label"],
            gdelt=ctx.get("gdelt_tone_ma3", "n/a"),
            bce_shock=ctx.get("bce_shock_score", "n/a"), bce_tone=ctx.get("bce_tone", "?"))
        ans = ollama_json(MODEL, SYSTEM, user)
        cache[ok] = {"prior_h1": prior_h1, "answer": ans}
        new += 1
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        if new % 6 == 0:
            el = time.time() - t0
            logger.info("  %d new calls, %.0fs elapsed (~%.1fs/call)", new, el, el / new)

    logger.info("Calls cached: %d (%d new) in %.0fs", len(cache), new, time.time() - t0)

    # ── Evaluate ──
    h1 = prior[prior["horizon"] == 1]
    correct = total = signbug = 0
    for _, r in h1.iterrows():
        ok = pd.Timestamp(r["origin"]).strftime("%Y-%m")
        e = cache.get(ok, {}).get("answer")
        if not e:
            continue
        stated = e.get("direction", "NEUTRAL")
        mag = float(e.get("magnitude_pp", 0) or 0)
        if stated == "UP" and mag < 0:
            signbug += 1
        gap = float(r["y_true"]) - float(r["y_pred"])
        actual = "UP" if gap > DEADBAND else ("DOWN" if gap < -DEADBAND else "NEUTRAL")
        if actual != "NEUTRAL":
            total += 1
            correct += (stated == actual)
    da = round(100 * correct / total, 1) if total else None

    # ── Apply bounded correction, compute MAE ──
    rows = []
    for _, r in prior.iterrows():
        ok = pd.Timestamp(r["origin"]).strftime("%Y-%m")
        e = cache.get(ok, {}).get("answer")
        corr = 0.0
        if e:
            d = DIR_SIGN.get(e.get("direction", "NEUTRAL"), 0.0)
            mag = min(abs(float(e.get("magnitude_pp", 0) or 0)), MAX_CORR)
            corr = d * mag * (HORIZON_DECAY ** (int(r["horizon"]) - 1))
        rows.append({"horizon": int(r["horizon"]), "origin": r["origin"],
                     "y_true": float(r["y_true"]), "y_pred": float(r["y_pred"]) + corr,
                     "y_prior": float(r["y_pred"])})
    dd = pd.DataFrame(rows)
    mae = {}
    for h in HORIZONS:
        hd = dd[dd["horizon"] == h]
        mae[h] = (round(float(np.mean(np.abs(hd["y_true"] - hd["y_pred"]))), 4),
                  round(float(np.mean(np.abs(hd["y_true"] - hd["y_prior"]))), 4))

    logger.info("\n--- PROBE RESULTS  (model=%s) ---", MODEL)
    logger.info("Directional accuracy : %s%%  over %d decisive origins", da, total)
    logger.info("Sign-consistency bug : %d (UP but magnitude<0)", signbug)
    logger.info("\n%-6s %10s %10s %8s", "h", "MAE_rev", "MAE_prior", "delta%")
    for h in HORIZONS:
        rev, pri = mae[h]
        logger.info("%-6d %10.4f %10.4f %+7.1f%%", h, rev, pri, (rev - pri) / pri * 100)
    logger.info("\nBaseline (qwen3:4b multi-agent): dir.acc 34%%, MAE h1 0.4398, prior 0.4364")

    summary = {"model": MODEL, "directional_accuracy_pct": da, "n_decisive": total,
               "sign_bug": signbug,
               "mae": {f"h{h}": {"revised": mae[h][0], "prior": mae[h][1]} for h in HORIZONS}}
    json.dump(summary, open(RESULTS_DIR / f"nexus_probe_summary_{MODEL.replace(':', '_')}.json", "w"),
              indent=2)
    logger.info("\nSaved probe summary.")


if __name__ == "__main__":
    main()
