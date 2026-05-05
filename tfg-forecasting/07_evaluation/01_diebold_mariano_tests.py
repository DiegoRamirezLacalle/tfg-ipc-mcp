"""
01_diebold_mariano_tests.py
---------------------------
Diebold-Mariano test (Harvey et al., 1997) to compare:
  - C0 vs C1 (does adding MCP signals significantly improve?)
  - Each model vs seasonal naive (do they beat the baseline?)

Loads prediction parquets from 08_results/ and baseline from
03_models_baseline/results/rolling_predictions.parquet.

Errors are aligned by (origin, horizon, fc_date) to ensure
comparisons are over the same observations.

Output:
  - 08_results/diebold_mariano_results.json
  - Table printed to console
"""

from __future__ import annotations

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

RESULTS_DIR = ROOT / "08_results"
BASELINE_PREDS = ROOT / "03_models_baseline" / "results" / "rolling_predictions.parquet"
HORIZONS = [1, 3, 6, 12]

# Sub-periods for DM by regime (same scheme as chronos2_C1)
SUBPERIODS = {
    "global": (None, None),
    "A_2021": ("2021-01-01", "2021-12-01"),
    "B_2022_shock": ("2022-01-01", "2022-12-01"),
    "C_2023_2024": ("2023-01-01", "2024-12-01"),
}


# ── Load predictions ──────────────────────────────────────────

def load_foundation_preds() -> dict[str, pd.DataFrame]:
    """Load the foundation model parquet files."""
    models = [
        "timesfm_C0", "timesfm_C1",
        "chronos2_C0", "chronos2_C1",
        "timegpt_C0", "timegpt_C1",
        "chronos2_C1_energy", "timegpt_C1_energy",
        "chronos2_C1_energy_only", "timegpt_C1_energy_only",
        "chronos2_C1_inst", "chronos2_C1_macro",
        "timesfm_C1_inst", "timesfm_C1_macro",
        "timegpt_C1_inst", "timegpt_C1_macro",
    ]
    preds = {}
    for m in models:
        path = RESULTS_DIR / f"{m}_predictions.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["origin"] = pd.to_datetime(df["origin"])
            df["fc_date"] = pd.to_datetime(df["fc_date"])
            preds[m] = df
        else:
            logger.warning(f"[!] Not found: {path.name}")
    return preds


def load_baseline_preds() -> dict[str, pd.DataFrame]:
    """Load baseline predictions (naive, arima, sarima, etc.)."""
    if not BASELINE_PREDS.exists():
        logger.warning(f"[!] Baseline predictions not found: {BASELINE_PREDS}")
        return {}
    df = pd.read_parquet(BASELINE_PREDS)
    df["origin"] = pd.to_datetime(df["origin"])
    df["fc_date"] = pd.to_datetime(df["fc_date"])
    result = {}
    for model_name in df["model"].unique():
        result[model_name] = df[df["model"] == model_name].copy()
    return result


# ── Error alignment ────────────────────────────────────────────

def align_errors(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    h: int,
    period_start: str | None = None,
    period_end: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract aligned errors from two prediction DataFrames.
    Only horizon h, intersection of (origin, fc_date).
    """
    h1 = df1[df1["horizon"] == h][["origin", "fc_date", "error"]].copy()
    h2 = df2[df2["horizon"] == h][["origin", "fc_date", "error"]].copy()

    if period_start:
        h1 = h1[h1["origin"] >= pd.Timestamp(period_start)]
        h2 = h2[h2["origin"] >= pd.Timestamp(period_start)]
    if period_end:
        h1 = h1[h1["origin"] <= pd.Timestamp(period_end)]
        h2 = h2[h2["origin"] <= pd.Timestamp(period_end)]

    merged = h1.merge(h2, on=["origin", "fc_date"], suffixes=("_1", "_2"))
    if len(merged) < 10:
        return None  # Insufficient sample size

    e1 = merged["error_1"].values
    e2 = merged["error_2"].values
    return e1, e2


# ── DM test per model pair ────────────────────────────────────

def run_dm_pair(
    name1: str,
    name2: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    period_name: str = "global",
    period_start: str | None = None,
    period_end: str | None = None,
) -> dict:
    """Run DM test for all horizons between two models."""
    result = {
        "model1": name1,
        "model2": name2,
        "period": period_name,
    }
    for h in HORIZONS:
        aligned = align_errors(df1, df2, h, period_start, period_end)
        if aligned is None:
            result[f"h{h}"] = {"dm_stat": None, "p_value": None,
                                "better": "insufficient_data", "n": 0}
            continue
        e1, e2 = aligned
        dm = diebold_mariano(e1, e2, h=h, power=1)  # power=1: MAE-based
        dm["n"] = len(e1)
        result[f"h{h}"] = dm
    return result


# ── Results table ──────────────────────────────────────────────

def print_dm_table(results: list[dict]) -> None:
    logger.info(f"\n{'Comparison':<32} {'Period':<16} {'h':>3} "
                f"{'DM-stat':>9} {'p-value':>9} {'Winner':>10} {'N':>5}")
    logger.info("-" * 90)
    for r in results:
        label = f"{r['model1']} vs {r['model2']}"
        for h in HORIZONS:
            key = f"h{h}"
            if key not in r:
                continue
            m = r[key]
            if m["dm_stat"] is None:
                logger.info(f"{label:<32} {r['period']:<16} {h:>3} {'-':>9} {'-':>9} "
                            f"{'insuf':>10} {m.get('n', 0):>5}")
                continue
            sig = "*" if m["p_value"] < 0.10 else ""
            sig += "*" if m["p_value"] < 0.05 else ""
            winner = m["better"]
            logger.info(f"{label:<32} {r['period']:<16} {h:>3} {m['dm_stat']:>9.4f} "
                        f"{m['p_value']:>9.4f} {winner:>10}{sig} {m['n']:>5}")


# ── Main ──────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("DIEBOLD-MARIANO TESTS - C0 vs C1 and models vs naive")
    logger.info("Metric: MAE (power=1), significance level: 5% (**) / 10% (*)")
    logger.info("=" * 60)

    foundation = load_foundation_preds()
    baseline = load_baseline_preds()

    if not foundation:
        logger.warning("[!] No foundation model predictions found. Run scripts 01-04 first.")
        return

    all_results = []

    # ── 1. C0 vs C1 per model and sub-period ─────────────────
    for family in ["timesfm", "chronos2", "timegpt"]:
        c0_name = f"{family}_C0"
        c1_name = f"{family}_C1"
        if c0_name not in foundation or c1_name not in foundation:
            logger.warning(f"[!] Missing {c0_name} or {c1_name}, skipping.")
            continue

        df_c0 = foundation[c0_name]
        df_c1 = foundation[c1_name]

        for period_name, (pstart, pend) in SUBPERIODS.items():
            r = run_dm_pair(c0_name, c1_name, df_c0, df_c1,
                            period_name, pstart, pend)
            all_results.append(r)

    # ── 1b. C0 vs C1_energy per model and sub-period ─────────
    for family in ["chronos2", "timegpt"]:
        c0_name = f"{family}_C0"
        c1e_name = f"{family}_C1_energy"
        if c0_name not in foundation or c1e_name not in foundation:
            logger.warning(f"[!] Missing {c0_name} or {c1e_name}, skipping.")
            continue

        for period_name, (pstart, pend) in SUBPERIODS.items():
            r = run_dm_pair(c0_name, c1e_name,
                            foundation[c0_name], foundation[c1e_name],
                            period_name, pstart, pend)
            all_results.append(r)

    # ── 1c. C1 vs C1_energy (MCP-only vs MCP+energy) ─────────
    for family in ["chronos2", "timegpt"]:
        c1_name = f"{family}_C1"
        c1e_name = f"{family}_C1_energy"
        if c1_name not in foundation or c1e_name not in foundation:
            continue
        r = run_dm_pair(c1_name, c1e_name,
                        foundation[c1_name], foundation[c1e_name],
                        "global", None, None)
        all_results.append(r)

    # ── 1d. C0 vs C1_energy_only per model and sub-period ────
    for family in ["chronos2", "timegpt"]:
        c0_name = f"{family}_C0"
        c1eo_name = f"{family}_C1_energy_only"
        if c0_name not in foundation or c1eo_name not in foundation:
            logger.warning(f"[!] Missing {c0_name} or {c1eo_name}, skipping.")
            continue
        for period_name, (pstart, pend) in SUBPERIODS.items():
            r = run_dm_pair(c0_name, c1eo_name,
                            foundation[c0_name], foundation[c1eo_name],
                            period_name, pstart, pend)
            all_results.append(r)

    # ── 1e. C1_energy_only vs C1_energy (energy-only vs energy+MCP) ──
    for family in ["chronos2", "timegpt"]:
        c1eo_name = f"{family}_C1_energy_only"
        c1e_name = f"{family}_C1_energy"
        if c1eo_name not in foundation or c1e_name not in foundation:
            continue
        r = run_dm_pair(c1eo_name, c1e_name,
                        foundation[c1eo_name], foundation[c1e_name],
                        "global", None, None)
        all_results.append(r)

    # ── 1f. C1_energy_only vs C1 (energy-only vs MCP-only) ───
    for family in ["chronos2", "timegpt"]:
        c1eo_name = f"{family}_C1_energy_only"
        c1_name = f"{family}_C1"
        if c1eo_name not in foundation or c1_name not in foundation:
            continue
        r = run_dm_pair(c1eo_name, c1_name,
                        foundation[c1eo_name], foundation[c1_name],
                        "global", None, None)
        all_results.append(r)

    # ── 1g. C0 vs C1_inst per model and sub-period ───────────
    for family in ["chronos2", "timesfm", "timegpt"]:
        c0_name = f"{family}_C0"
        c1i_name = f"{family}_C1_inst"
        if c0_name not in foundation or c1i_name not in foundation:
            logger.warning(f"[!] Missing {c0_name} or {c1i_name}, skipping.")
            continue
        for period_name, (pstart, pend) in SUBPERIODS.items():
            r = run_dm_pair(c0_name, c1i_name,
                            foundation[c0_name], foundation[c1i_name],
                            period_name, pstart, pend)
            all_results.append(r)

    # ── 1h. C0 vs C1_macro per model ─────────────────────────
    for family in ["chronos2", "timesfm", "timegpt"]:
        c0_name = f"{family}_C0"
        c1m_name = f"{family}_C1_macro"
        if c0_name not in foundation or c1m_name not in foundation:
            logger.warning(f"[!] Missing {c0_name} or {c1m_name}, skipping.")
            continue
        r = run_dm_pair(c0_name, c1m_name,
                        foundation[c0_name], foundation[c1m_name],
                        "global", None, None)
        all_results.append(r)

    # ── 2. Foundation C0 vs naive (reference baseline) ────────
    naive_df = baseline.get("naive")
    if naive_df is not None:
        for model_name in ["timesfm_C0", "chronos2_C0", "timegpt_C0"]:
            if model_name not in foundation:
                continue
            r = run_dm_pair("naive", model_name,
                            naive_df, foundation[model_name],
                            "global", None, None)
            all_results.append(r)

        # C1 vs naive
        for model_name in ["timesfm_C1", "chronos2_C1", "timegpt_C1",
                           "chronos2_C1_energy", "timegpt_C1_energy",
                           "chronos2_C1_energy_only", "timegpt_C1_energy_only"]:
            if model_name not in foundation:
                continue
            r = run_dm_pair("naive", model_name,
                            naive_df, foundation[model_name],
                            "global", None, None)
            all_results.append(r)

    # ── 3. Cross-model C0 comparisons ─────────────────────────
    c0_pairs = [
        ("timesfm_C0", "chronos2_C0"),
        ("timesfm_C0", "timegpt_C0"),
        ("chronos2_C0", "timegpt_C0"),
    ]
    for m1, m2 in c0_pairs:
        if m1 in foundation and m2 in foundation:
            r = run_dm_pair(m1, m2, foundation[m1], foundation[m2],
                            "global", None, None)
            all_results.append(r)

    # ── 4. Cross-model C1 comparisons ─────────────────────────
    c1_pairs = [
        ("timesfm_C1", "chronos2_C1"),
        ("timesfm_C1", "timegpt_C1"),
        ("chronos2_C1", "timegpt_C1"),
    ]
    for m1, m2 in c1_pairs:
        if m1 in foundation and m2 in foundation:
            r = run_dm_pair(m1, m2, foundation[m1], foundation[m2],
                            "global", None, None)
            all_results.append(r)

    # ── Print table ────────────────────────────────────────────
    print_dm_table(all_results)
    logger.info("\nNote: ** p<0.05, * p<0.10. DM-stat<0 => model1 better; >0 => model2 better")

    # ── Save JSON ──────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "diebold_mariano_results_final.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nDM results: {out_path}")


if __name__ == "__main__":
    main()
