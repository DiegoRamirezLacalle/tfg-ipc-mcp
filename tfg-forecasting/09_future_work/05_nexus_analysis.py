"""
05_nexus_analysis.py  —  Post-hoc analysis of the Nexus multi-agent traces
===========================================================================

This script does NOT call any LLM.  It re-processes the cached reasoning
traces in `results/nexus_multiagent_traces.json` to answer three questions
the raw MAE table could not:

  1. CONFIDENCE GATING — "when *not* to revise" (PostTime's core hypothesis).
     The local LLM (qwen3:4b) self-reports MEDIUM confidence for every origin,
     so its confidence field is unusable as a gate.  We therefore test
     *behavioural* gates instead: revise only when the proposed correction is
     large (|Δ| ≥ threshold) or when the two reasoning agents agree on
     direction.

  2. SIGN-CONSISTENCY — a discovered failure mode of the small local model:
     `agent3.correction_pp` is negative on EVERY origin, even when the stated
     `direction` is UP.  The model cannot keep the sign of its magnitude
     consistent with its own stated direction.  We compare the "as-run"
     (sign-buggy) corrections against a "sign-enforced" variant where the
     correction is reconstructed as  sign(direction) × |magnitude|.

  3. DIRECTIONAL ACCURACY — the real contribution of the reasoning layer:
     does the stated direction match the sign of the prior's error
     (i.e. did the LLM correctly say "the TSFM will under/over-shoot")?
     This is the interesting result even when MAE does not improve, and is
     exactly what a trained reviser (PostTime SFT+RLVR) would exploit.

Policies compared (correction applied to TimesFM C0 prior, decayed γ^(h-1)):
  P0  prior            — no revision (TimesFM C0)
  P1  as-run           — raw agent3.correction_pp (sign-buggy, the original)
  P2  sign-enforced    — sign(direction) × |agent3.correction_pp|
  P3  P2 + magnitude gate    — revise only if |Δ| ≥ MAG_GATE
  P4  P2 + agreement gate     — revise only if agent2.dir == agent3.dir

Output: results/nexus_analysis.json  +  console tables.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "09_future_work" / "results"
TRACES = RESULTS_DIR / "nexus_multiagent_traces.json"
PRIOR_PRED = ROOT / "08_results" / "timesfm_C0_predictions.parquet"
HORIZONS = [1, 3, 6, 12]
HORIZON_DECAY = 0.70

MAG_GATE = 0.075        # revise only if |correction| ≥ this (behavioural confidence proxy)
DIR_SIGN = {"UP": +1.0, "DOWN": -1.0, "NEUTRAL": 0.0}


def load_prior() -> pd.DataFrame:
    df = pd.read_parquet(PRIOR_PRED)
    df["origin"] = pd.to_datetime(df["origin"])
    df["fc_date"] = pd.to_datetime(df["fc_date"])
    return df


def mase_scale() -> float:
    f = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    f["date"] = pd.to_datetime(f["date"])
    y = f.set_index("date")["indice_general"].loc[:DATE_TRAIN_END]
    return float(np.mean(np.abs(y.values[12:] - y.values[:-12])))


def correction_for_policy(trace: dict, policy: str) -> float:
    """Return the h=1 correction (pp) under the given policy."""
    a2 = trace.get("agent2") or {}
    a3 = trace.get("agent3") or {}
    if not a3:
        return 0.0

    raw = float(a3.get("correction_pp", 0.0) or 0.0)
    direction = a3.get("direction", "NEUTRAL")
    should = a3.get("should_revise", False)
    mag = abs(raw)
    signed = DIR_SIGN.get(direction, 0.0) * mag    # sign-enforced

    if policy == "as_run":
        return float(trace.get("final_correction_pp", 0.0) or 0.0)
    if policy == "sign_enforced":
        return signed if should else 0.0
    if policy == "mag_gate":
        return signed if (should and mag >= MAG_GATE) else 0.0
    if policy == "agreement_gate":
        agree = a2.get("expected_direction") == direction
        return signed if (should and agree) else 0.0
    return 0.0


def metrics_for_policy(prior: pd.DataFrame, traces: dict, policy: str,
                       scale: float) -> dict:
    rows = []
    for _, r in prior.iterrows():
        ok = pd.Timestamp(r["origin"]).strftime("%Y-%m")
        tr = traces.get(ok)
        corr_h1 = correction_for_policy(tr, policy) if tr else 0.0
        corr = corr_h1 * (HORIZON_DECAY ** (int(r["horizon"]) - 1))
        yp = float(r["y_pred"]) + corr
        rows.append({"horizon": int(r["horizon"]), "origin": r["origin"],
                     "y_true": float(r["y_true"]), "y_pred": yp})
    d = pd.DataFrame(rows)
    out = {}
    for h in HORIZONS:
        hd = d[d["horizon"] == h]
        if hd.empty:
            continue
        e = np.abs(hd["y_true"].values - hd["y_pred"].values)
        out[f"h{h}"] = {"MAE": round(float(e.mean()), 4),
                        "MASE": round(float(e.mean() / scale), 4),
                        "n_evals": int(hd["origin"].nunique())}
    return out


def directional_accuracy(prior: pd.DataFrame, traces: dict) -> dict:
    """
    Did the stated direction match the sign of the prior's h=1 error?
    actual_dir = sign(y_true - prior)  ('UP' if TSFM under-forecast).
    Only count origins where |actual change| exceeds a dead-band.
    """
    h1 = prior[prior["horizon"] == 1].copy()
    correct = total = 0
    confusion = Counter()
    deadband = 0.05
    for _, r in h1.iterrows():
        ok = pd.Timestamp(r["origin"]).strftime("%Y-%m")
        tr = traces.get(ok)
        if not tr or not tr.get("agent3"):
            continue
        stated = tr["agent3"].get("direction", "NEUTRAL")
        gap = float(r["y_true"]) - float(r["y_pred"])
        actual = "UP" if gap > deadband else ("DOWN" if gap < -deadband else "NEUTRAL")
        confusion[(stated, actual)] += 1
        if actual != "NEUTRAL":
            total += 1
            if stated == actual:
                correct += 1
    return {
        "directional_accuracy_pct": round(100 * correct / total, 1) if total else None,
        "n_evaluated": total,
        "confusion": {f"{s}->{a}": c for (s, a), c in sorted(confusion.items())},
    }


def reasoning_summary(traces: dict) -> dict:
    channels = Counter()
    a2_dirs = Counter()
    sign_bug = 0
    n = 0
    for tr in traces.values():
        a2 = tr.get("agent2") or {}
        a3 = tr.get("agent3") or {}
        if a2:
            channels[a2.get("main_channel", "?")] += 1
            a2_dirs[a2.get("expected_direction", "?")] += 1
        if a3 and a3.get("direction") == "UP" and float(a3.get("correction_pp", 0) or 0) < 0:
            sign_bug += 1
        if a3:
            n += 1
    return {
        "main_channel_counts": dict(channels),
        "agent2_direction_counts": dict(a2_dirs),
        "sign_inconsistency_UP_with_negative_corr": f"{sign_bug}/{n}",
    }


def main() -> None:
    if not TRACES.exists():
        logger.error("Traces not found: %s — run 04_nexus_multiagent_ollama.py first.", TRACES)
        return

    traces = json.load(open(TRACES, encoding="utf-8"))
    prior = load_prior()
    scale = mase_scale()
    logger.info("Loaded %d traces, prior %d rows, MASE scale %.4f",
                len(traces), len(prior), scale)

    policies = ["prior", "as_run", "sign_enforced", "mag_gate", "agreement_gate"]
    labels = {
        "prior":          "P0 TimesFM C0 (no revision)",
        "as_run":         "P1 as-run (raw corr, sign-buggy)",
        "sign_enforced":  "P2 sign-enforced",
        "mag_gate":       f"P3 sign + |corr|>={MAG_GATE} gate",
        "agreement_gate": "P4 sign + agent agreement gate",
    }

    all_metrics = {}
    for pol in policies:
        if pol == "prior":
            m = metrics_for_policy(prior, {}, "noop", scale)   # no traces → no correction
        else:
            m = metrics_for_policy(prior, traces, pol, scale)
        all_metrics[pol] = m

    # ── Table ──
    logger.info("\n%-34s %8s %8s %8s %8s", "Policy", "h1 MAE", "h3 MAE", "h6 MAE", "h12 MAE")
    logger.info("-" * 70)
    for pol in policies:
        row = "%-34s" % labels[pol]
        for h in HORIZONS:
            v = all_metrics[pol].get(f"h{h}", {}).get("MAE")
            row += " %8.4f" % v if v is not None else "        ?"
        logger.info(row)

    # Best policy at h=1
    prior_h1 = all_metrics["prior"]["h1"]["MAE"]
    logger.info("\nDelta vs TimesFM C0 prior (h=1 MAE = %.4f):", prior_h1)
    for pol in policies[1:]:
        v = all_metrics[pol].get("h1", {}).get("MAE")
        if v:
            logger.info("  %-30s %.4f  (%+.1f%%)", labels[pol], v, (v - prior_h1) / prior_h1 * 100)

    # ── Directional accuracy ──
    da = directional_accuracy(prior, traces)
    logger.info("\n--- Directional accuracy (stated direction vs sign of TSFM error) ---")
    logger.info("  Accuracy: %s%% over %d non-flat origins (dead-band ±0.05 pp)",
                da["directional_accuracy_pct"], da["n_evaluated"])
    logger.info("  Confusion (stated->actual): %s", da["confusion"])

    # ── Reasoning summary ──
    rs = reasoning_summary(traces)
    logger.info("\n--- Reasoning summary ---")
    logger.info("  Transmission channel chosen: %s", rs["main_channel_counts"])
    logger.info("  Agent-2 direction calls    : %s", rs["agent2_direction_counts"])
    logger.info("  Sign inconsistency (UP but corr<0): %s  <- local-LLM failure mode",
                rs["sign_inconsistency_UP_with_negative_corr"])

    # ── Save ──
    summary = {
        "policies": {pol: all_metrics[pol] for pol in policies},
        "policy_labels": labels,
        "directional_accuracy": da,
        "reasoning_summary": rs,
        "params": {"horizon_decay": HORIZON_DECAY, "mag_gate": MAG_GATE},
    }
    out = RESULTS_DIR / "nexus_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("\nSaved: %s", out.name)


if __name__ == "__main__":
    main()
