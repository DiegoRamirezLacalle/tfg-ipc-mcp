"""
05_diebold_mariano_europe.py — DM tests HICP Eurozone

Comparisons:
  1. C0 vs C1_inst, C1_mcp, C1_full (per family and sub-period)
  2. C1_inst vs C1_mcp (which signal contributes more)
  3. C1_inst vs C1_full, C1_mcp vs C1_full
  4. Each foundation C0 vs SARIMA Europe (reference baseline)
  5. Cross-model C0 (TimesFM vs Chronos-2 vs TimeGPT)

Output:
  08_results/diebold_mariano_results_europe.json
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
HORIZONS = [1, 3, 6, 12]

SUBPERIODS = {
    "global":          (None, None),
    "A_pre_shock":     ("2021-01-01", "2022-06-01"),
    "B_shock":         ("2022-07-01", "2023-06-01"),
    "C_normalizacion": ("2023-07-01", "2024-12-01"),
}


def load_europe_preds() -> dict[str, pd.DataFrame]:
    models = [
        # C0
        "chronos2_C0_europe", "timesfm_C0_europe", "timegpt_C0_europe",
        # C1 inst
        "chronos2_C1_inst_europe", "timesfm_C1_inst_europe", "timegpt_C1_inst_europe",
        # C1 mcp
        "chronos2_C1_mcp_europe", "timesfm_C1_mcp_europe", "timegpt_C1_mcp_europe",
        # C1 full
        "chronos2_C1_full_europe", "timesfm_C1_full_europe", "timegpt_C1_full_europe",
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


def load_baseline_europe() -> dict[str, pd.DataFrame]:
    """Load rolling_predictions_europe.parquet with naive and sarima."""
    path = RESULTS_DIR / "rolling_predictions_europe.parquet"
    if not path.exists():
        logger.warning(f"[!] Baseline not found: {path}")
        return {}
    df = pd.read_parquet(path)
    df["origin"] = pd.to_datetime(df["origin"])
    df["fc_date"] = pd.to_datetime(df["fc_date"])
    result = {}
    for name in df["model"].unique():
        result[name] = df[df["model"] == name].copy()
    return result


def align_errors(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    h: int,
    period_start: str | None = None,
    period_end: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    h1 = df1[df1["horizon"] == h][["origin", "fc_date", "error"]].copy()
    h2 = df2[df2["horizon"] == h][["origin", "fc_date", "error"]].copy()
    if period_start:
        h1 = h1[h1["origin"] >= pd.Timestamp(period_start)]
        h2 = h2[h2["origin"] >= pd.Timestamp(period_start)]
    if period_end:
        h1 = h1[h1["origin"] <= pd.Timestamp(period_end)]
        h2 = h2[h2["origin"] <= pd.Timestamp(period_end)]
    merged = h1.merge(h2, on=["origin", "fc_date"], suffixes=("_1", "_2"))
    if len(merged) < 8:
        return None
    return merged["error_1"].values, merged["error_2"].values


def run_dm_pair(
    name1: str,
    name2: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    period_name: str = "global",
    period_start: str | None = None,
    period_end: str | None = None,
) -> dict:
    result = {"model1": name1, "model2": name2, "period": period_name}
    for h in HORIZONS:
        aligned = align_errors(df1, df2, h, period_start, period_end)
        if aligned is None:
            result[f"h{h}"] = {"dm_stat": None, "p_value": None,
                                "better": "insufficient_data", "n": 0}
            continue
        e1, e2 = aligned
        dm = diebold_mariano(e1, e2, h=h, power=1)
        dm["n"] = len(e1)
        result[f"h{h}"] = dm
    return result


def print_dm_table(results: list[dict]) -> None:
    logger.info(f"\n{'Comparison':<42} {'Period':<18} {'h':>3} "
                f"{'DM-stat':>9} {'p-val':>8} {'Winner':>14} {'N':>5}")
    logger.info("-" * 100)
    for r in results:
        label = f"{r['model1']} vs {r['model2']}"
        for h in HORIZONS:
            key = f"h{h}"
            if key not in r:
                continue
            m = r[key]
            if m["dm_stat"] is None:
                logger.info(f"{label:<42} {r['period']:<18} {h:>3} {'—':>9} {'—':>8} "
                            f"{'insuf':>14} {m.get('n', 0):>5}")
                continue
            sig = "**" if m["p_value"] < 0.05 else ("*" if m["p_value"] < 0.10 else "")
            win = m["better"]
            logger.info(f"{label:<42} {r['period']:<18} {h:>3} {m['dm_stat']:>9.4f} "
                        f"{m['p_value']:>8.4f} {win:>14}{sig} {m['n']:>5}")


def main():
    logger.info("=" * 65)
    logger.info("DIEBOLD-MARIANO TESTS — HICP Eurozone")
    logger.info("Metric: MAE (power=1) | ** p<0.05 | * p<0.10")
    logger.info("=" * 65)

    preds = load_europe_preds()
    baseline = load_baseline_europe()
    all_results = []

    if not preds:
        logger.warning("[!] No predictions available.")
        return

    # ── 1. C0 vs C1_inst / C1_mcp / C1_full per family and sub-period
    for family in ["chronos2", "timesfm", "timegpt"]:
        c0_key = f"{family}_C0_europe"
        if c0_key not in preds:
            continue
        for c1_tag in ["inst", "mcp", "full"]:
            c1_key = f"{family}_C1_{c1_tag}_europe"
            if c1_key not in preds:
                continue
            for pname, (ps, pe) in SUBPERIODS.items():
                r = run_dm_pair(c0_key, c1_key, preds[c0_key], preds[c1_key],
                                pname, ps, pe)
                all_results.append(r)

    # ── 2. C1_inst vs C1_mcp (which signal contributes more)
    for family in ["chronos2", "timesfm", "timegpt"]:
        k_inst = f"{family}_C1_inst_europe"
        k_mcp = f"{family}_C1_mcp_europe"
        if k_inst in preds and k_mcp in preds:
            r = run_dm_pair(k_inst, k_mcp, preds[k_inst], preds[k_mcp], "global")
            all_results.append(r)

    # ── 3. C1_inst vs C1_full, C1_mcp vs C1_full
    for family in ["chronos2", "timesfm", "timegpt"]:
        k_inst = f"{family}_C1_inst_europe"
        k_mcp = f"{family}_C1_mcp_europe"
        k_full = f"{family}_C1_full_europe"
        if k_inst in preds and k_full in preds:
            r = run_dm_pair(k_inst, k_full, preds[k_inst], preds[k_full], "global")
            all_results.append(r)
        if k_mcp in preds and k_full in preds:
            r = run_dm_pair(k_mcp, k_full, preds[k_mcp], preds[k_full], "global")
            all_results.append(r)

    # ── 4. Foundation C0 vs SARIMA Europe (reference baseline)
    sarima_df = baseline.get("sarima")
    naive_df = baseline.get("naive")
    if sarima_df is not None:
        for c0_key in ["chronos2_C0_europe", "timesfm_C0_europe", "timegpt_C0_europe"]:
            if c0_key in preds:
                r = run_dm_pair("sarima_europe", c0_key,
                                sarima_df, preds[c0_key], "global")
                all_results.append(r)
        # C1 vs SARIMA
        for suffix in ["C1_inst_europe", "C1_mcp_europe", "C1_full_europe"]:
            for family in ["chronos2", "timesfm", "timegpt"]:
                key = f"{family}_{suffix}"
                if key in preds:
                    r = run_dm_pair("sarima_europe", key, sarima_df, preds[key], "global")
                    all_results.append(r)

    if naive_df is not None:
        for c0_key in ["chronos2_C0_europe", "timesfm_C0_europe", "timegpt_C0_europe"]:
            if c0_key in preds:
                r = run_dm_pair("naive_europe", c0_key,
                                naive_df, preds[c0_key], "global")
                all_results.append(r)

    # ── 5. Cross-model C0
    c0_pairs = [
        ("timesfm_C0_europe", "chronos2_C0_europe"),
        ("timesfm_C0_europe", "timegpt_C0_europe"),
        ("chronos2_C0_europe", "timegpt_C0_europe"),
    ]
    for m1, m2 in c0_pairs:
        if m1 in preds and m2 in preds:
            r = run_dm_pair(m1, m2, preds[m1], preds[m2], "global")
            all_results.append(r)

    # ── Print and save ─────────────────────────────────────────
    print_dm_table(all_results)
    logger.info("\nNote: DM<0 => model1 better  |  DM>0 => model2 better")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "diebold_mariano_results_europe.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
