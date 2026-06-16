"""
06_extract_dm_pvalues.py
------------------------
Consolidate Diebold-Mariano p-values for the MAIN C0 vs C1 comparisons across
the three CPI series (Spain, Global, Europe), ready for the thesis results
tables.

Two jobs:
  1. EXTRACT the already-computed DM tests:
       - Spain  foundation C0 vs C1{,_energy,_energy_only,_inst,_macro}
                (08_results/diebold_mariano_results_final.json, script 01)
       - Europe foundation C0 vs C1_{inst,mcp,full}
                (08_results/diebold_mariano_results_europe.json, script 05)
  2. FILL the missing comparisons from the already-stored forecasts
     (NO retraining):
       - Global foundation C0_global vs C1_inst_global (chronos2, timesfm).
         TimeGPT C0_global was never generated (Nixtla API) -> marked [VERIFY].
       - Europe classical SARIMA (C0) vs ARIMAX C1_inst / C1_full (script 08).

Method: shared.metrics.diebold_mariano (Harvey-Leybourne-Newbold corrected,
power=1 => MAE-based), identical to scripts 01 and 05. Errors aligned by
(origin, fc_date) per horizon. Significance: ** p<0.05, * p<0.10.

Output:
  08_results/dm_pvalues_summary.csv   (long format, all periods)
  08_results/dm_pvalues_summary.md    (global period, grouped by series)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger
from shared.metrics import diebold_mariano

logger = get_logger(__name__)

RESULTS = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
FAMILIES = ["timesfm", "chronos2", "timegpt"]

# Human-readable C1 signal label per model-key suffix.
SPAIN_SIGNAL = {
    "C1": "mcp", "C1_energy": "energy", "C1_energy_only": "energy_only",
    "C1_inst": "inst", "C1_macro": "macro",
}
VERIFY = "[VERIFY]"


# ── shared helpers ────────────────────────────────────────────────

def _align(df1: pd.DataFrame, df2: pd.DataFrame, h: int) -> tuple[np.ndarray, np.ndarray] | None:
    a = df1[df1["horizon"] == h][["origin", "fc_date", "error"]]
    b = df2[df2["horizon"] == h][["origin", "fc_date", "error"]]
    m = a.merge(b, on=["origin", "fc_date"], suffixes=("_1", "_2"))
    if len(m) < 8:
        return None
    return m["error_1"].values, m["error_2"].values


def _load_preds(path: Path, model: str | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["origin"] = pd.to_datetime(df["origin"])
    df["fc_date"] = pd.to_datetime(df["fc_date"])
    if model is not None:
        df = df[df["model"] == model].copy()
    return df if len(df) else None


def _dm_from_forecasts(c0: pd.DataFrame, c1: pd.DataFrame) -> dict[int, dict]:
    """DM per horizon from two stored-forecast frames. model1=C0, model2=C1."""
    out = {}
    for h in HORIZONS:
        al = _align(c0, c1, h)
        if al is None:
            out[h] = {"dm_stat": None, "p_value": None, "better": "insufficient_data", "n": 0}
            continue
        e1, e2 = al
        d = diebold_mariano(e1, e2, h=h, power=1)
        d["n"] = len(e1)
        out[h] = d
    return out


def _row(series, track, family, signal, h, dm):
    """Build a flat record from a per-horizon DM dict."""
    p = dm.get("p_value")
    dmstat = dm.get("dm_stat")
    n = dm.get("n", 0)
    if p is None:
        sig, winner = "ns", "insuf"
    else:
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
        # model2 == C1; dm_stat>0 => C1 better, <0 => C0 better
        winner = "C1" if dmstat > 0 else "C0"
    return {
        "series": series, "track": track, "family": family, "signal": signal,
        "horizon": h, "n": n,
        "dm_stat": None if dmstat is None else round(dmstat, 4),
        "p_value": None if p is None else round(p, 4),
        "signif": sig, "better": winner,
    }


# ── 1. extract existing JSON DM results (global period only) ───────

def extract_spain(records: list[dict]) -> None:
    p = RESULTS / "diebold_mariano_results_final.json"
    if not p.exists():
        logger.warning("[!] Spain DM file missing: %s", p.name)
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    for r in data:
        if r.get("period") != "global":
            continue
        m1, m2 = r["model1"], r["model2"]
        # main C0 vs C1: model1 == {family}_C0, model2 == {family}_C1*
        if not m1.endswith("_C0"):
            continue
        family = m1[:-3]
        if family not in FAMILIES or not m2.startswith(f"{family}_C1"):
            continue
        suffix = m2[len(family) + 1:]  # drop "{family}_"
        signal = SPAIN_SIGNAL.get(suffix, suffix)
        for h in HORIZONS:
            dm = r.get(f"h{h}")
            if dm:
                records.append(_row("Spain", "Foundation", family, signal, h, dm))


def extract_europe(records: list[dict]) -> None:
    p = RESULTS / "diebold_mariano_results_europe.json"
    if not p.exists():
        logger.warning("[!] Europe DM file missing: %s", p.name)
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    for r in data:
        if r.get("period") != "global":
            continue
        m1, m2 = r["model1"], r["model2"]
        if not m1.endswith("_C0_europe"):
            continue
        family = m1[:-len("_C0_europe")]
        if family not in FAMILIES or not m2.startswith(f"{family}_C1_"):
            continue
        signal = m2[len(family) + 4:-len("_europe")]  # "{fam}_C1_<sig>_europe"
        for h in HORIZONS:
            dm = r.get(f"h{h}")
            if dm:
                records.append(_row("Europe", "Foundation", family, signal, h, dm))


# ── 2. fill missing comparisons from stored forecasts ─────────────

def fill_global_foundation(records: list[dict]) -> None:
    """Global foundation C0_global vs C1_inst_global (no retraining)."""
    for family in FAMILIES:
        c0 = _load_preds(RESULTS / f"{family}_C0_global_predictions.parquet")
        c1 = _load_preds(RESULTS / f"{family}_C1_inst_global_predictions.parquet")
        if c1 is None:
            logger.warning("[!] Global C1_inst missing for %s", family)
            continue
        if c0 is None:
            # e.g. TimeGPT: C0_global never generated -> cannot test
            for h in HORIZONS:
                rec = _row("Global", "Foundation", family, "inst", h,
                           {"dm_stat": None, "p_value": None, "n": 0})
                rec["p_value"] = VERIFY
                rec["dm_stat"] = VERIFY
                rec["signif"] = VERIFY
                rec["better"] = VERIFY
                records.append(rec)
            logger.warning("[!] Global C0_global missing for %s -> %s", family, VERIFY)
            continue
        dm = _dm_from_forecasts(c0, c1)
        for h in HORIZONS:
            records.append(_row("Global", "Foundation", family, "inst", h, dm[h]))


def fill_global_improved(records: list[dict]) -> None:
    """Improved Global C1 variants vs C0_global (Phase 1-3, no original C1 touched).

    - inst+fwd       : Chronos-2 with honest damped-drift forward covariates (script 33)
    - inst+regime    : change/regime-gated context overlay (script 09)
    - inst+validated : pre-2021 Ridge context overlay (script 08)
    """
    variants = [
        ("chronos2", "inst+fwd",       "chronos2_C1_fwd_global_predictions.parquet"),
        ("chronos2", "inst+regime",    "chronos2_C1_regime_global_predictions.parquet"),
        ("timesfm",  "inst+regime",    "timesfm_C1_regime_global_predictions.parquet"),
        ("chronos2", "inst+validated", "chronos2_C1_validated_global_predictions.parquet"),
        ("timesfm",  "inst+validated", "timesfm_C1_validated_global_predictions.parquet"),
    ]
    for family, signal, fname in variants:
        c0 = _load_preds(RESULTS / f"{family}_C0_global_predictions.parquet")
        c1 = _load_preds(RESULTS / fname)
        if c0 is None or c1 is None:
            logger.warning("[!] Global improved variant missing: %s", fname)
            continue
        dm = _dm_from_forecasts(c0, c1)
        for h in HORIZONS:
            records.append(_row("Global", "Foundation", family, signal, h, dm[h]))


def fill_europe_classical(records: list[dict]) -> None:
    """Europe classical SARIMA (C0) vs ARIMAX C1_inst / C1_full (script 08)."""
    c0 = _load_preds(RESULTS / "rolling_predictions_europe.parquet", model="sarima")
    if c0 is None:
        logger.warning("[!] Europe baseline sarima predictions missing")
        return
    for cond, fpath in [
        ("inst", RESULTS / "rolling_predictions_C1_inst_europe.parquet"),
        ("full", RESULTS / "rolling_predictions_C1_full_europe.parquet"),
    ]:
        c1 = _load_preds(fpath, model=f"arimax_C1_{cond}_europe")
        if c1 is None:
            logger.warning("[!] Europe classical C1_%s missing", cond)
            continue
        dm = _dm_from_forecasts(c0, c1)
        for h in HORIZONS:
            records.append(_row("Europe", "Classical (ARIMAX)", "sarimax", cond, h, dm[h]))


# ── output ─────────────────────────────────────────────────────────

def write_csv(records: list[dict]) -> None:
    cols = ["series", "track", "family", "signal", "horizon", "n",
            "dm_stat", "p_value", "signif", "better"]
    out = RESULTS / "dm_pvalues_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)
    logger.info("CSV saved: %s (%d rows)", out.name, len(records))


def write_md(records: list[dict]) -> None:
    out = RESULTS / "dm_pvalues_summary.md"
    lines = [
        "# Diebold-Mariano p-values - main C0 vs C1 comparisons",
        "",
        "> **LEGACY / SUPPLEMENTARY.** This table uses the older "
        "`shared.metrics.diebold_mariano` (normal reference, circular-`np.roll` "
        "autocovariance). The **canonical strict evidence for the thesis is "
        "`thesis_critical_dm_recompute.md`** (HLN-adjusted Student-t, df=n-1, "
        "proper HAC) and the per-variant `before_after_c1.md`. Keep this file for "
        "continuity only.",
        "",
        "> DM, MAE-based (power=1). model1 = C0, model2 = C1. `better` = C1 if the "
        "C1 (with-signals) model has lower error, C0 otherwise. Significance: "
        "** p<0.05, * p<0.10, ns = not significant. `[VERIFY]` = forecast not "
        "available (cannot be tested).",
        "",
    ]
    hdr = "| Series | Track | Family | C1 signal | h=1 | h=3 | h=6 | h=12 |"
    sep = "|--------|-------|--------|-----------|-----|-----|-----|------|"

    # group by (series, track, family, signal); one row, 4 horizon cells
    groups: dict[tuple, dict[int, dict]] = {}
    order: list[tuple] = []
    for r in records:
        key = (r["series"], r["track"], r["family"], r["signal"])
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][r["horizon"]] = r

    def cell(rec: dict | None) -> str:
        if rec is None:
            return "-"
        p = rec["p_value"]
        if p == VERIFY:
            return VERIFY
        if p is None:
            return "insuf"
        return f"{p:.3f}{rec['signif'] if rec['signif'] != 'ns' else ''} ({rec['better']})"

    lines += [hdr, sep]
    prev_series = None
    for key in order:
        series, track, family, signal = key
        if series != prev_series:
            prev_series = series
        cells = [cell(groups[key].get(h)) for h in HORIZONS]
        lines.append(
            f"| {series} | {track} | {family} | {signal} | "
            + " | ".join(cells) + " |"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("MD  saved: %s", out.name)


def main() -> None:
    logger.info("=" * 64)
    logger.info("DM p-value extraction - main C0 vs C1 (Spain / Global / Europe)")
    logger.info("=" * 64)

    records: list[dict] = []
    extract_spain(records)
    extract_europe(records)
    fill_global_foundation(records)
    fill_global_improved(records)
    fill_europe_classical(records)

    if not records:
        logger.warning("[!] No DM records produced.")
        return

    # console summary (global period, significant only)
    logger.info("\nSignificant C1 improvements (p<0.05, C1 better):")
    any_sig = False
    for r in records:
        if r["p_value"] not in (None, VERIFY) and r["signif"] == "**" and r["better"] == "C1":
            any_sig = True
            logger.info("  %-7s %-18s %-9s %-11s h=%-2d  p=%.4f  n=%d",
                        r["series"], r["track"], r["family"], r["signal"],
                        r["horizon"], r["p_value"], r["n"])
    if not any_sig:
        logger.info("  (none at p<0.05)")

    write_csv(records)
    write_md(records)
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
