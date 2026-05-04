"""
build_metrics_summary_final.py — Full comparison table for all models

Models included:
  Baseline:    naive, arima, sarima, sarimax
  Deep:        lstm, nbeats, nhits
  Foundation C0: timesfm_C0, chronos2_C0, timegpt_C0
  Foundation C1: timesfm_C1, chronos2_C1, timegpt_C1

Metrics: MAE, RMSE, MASE per horizon h=1,3,6,12
Output: 08_results/metrics_summary_final.json
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

from shared.constants import DATE_TRAIN_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]


def compute_mase_scale() -> float:
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df["indice_general"]
    y.index = pd.to_datetime(y.index)
    train = y.loc[:DATE_TRAIN_END]
    return float(np.mean(np.abs(train.values[12:] - train.values[:-12])))


def metrics_from_parquet(path: Path, model_name: str, mase_scale: float) -> dict:
    df = pd.read_parquet(path)
    if "model" in df.columns:
        df = df[df["model"] == model_name]
    results = {}
    for h in HORIZONS:
        hdf = df[df["horizon"] == h]
        if hdf.empty:
            continue
        yt = hdf["y_true"].values
        yp = hdf["y_pred"].values
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mase = mae / mase_scale
        results[f"h{h}"] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MASE": round(mase, 4),
            "n_evals": int(len(hdf["origin"].unique())),
        }
    return results


def main():
    mase_scale = compute_mase_scale()
    logger.info(f"MASE scale (naive seasonal lag-12, train 2002-2020): {mase_scale:.4f}")

    all_metrics = {}

    # ── Baseline models ──
    baseline_path = ROOT / "03_models_baseline" / "results" / "rolling_predictions.parquet"
    if baseline_path.exists():
        for model in ["naive", "arima", "sarima", "sarimax"]:
            all_metrics[model] = metrics_from_parquet(baseline_path, model, mase_scale)
            logger.info(f"  {model}: OK")

    # ── Deep models ──
    deep_path = ROOT / "04_models_deep" / "results" / "deep_rolling_predictions.parquet"
    if deep_path.exists():
        for model in ["lstm", "nbeats", "nhits"]:
            all_metrics[model] = metrics_from_parquet(deep_path, model, mase_scale)
            logger.info(f"  {model}: OK")

    # ── Foundation models ──
    foundation_models = [
        "timesfm_C0", "timesfm_C1",
        "chronos2_C0", "chronos2_C1",
        "timegpt_C0", "timegpt_C1",
        "chronos2_C1_energy", "timegpt_C1_energy",
        "chronos2_C1_energy_only", "timegpt_C1_energy_only",
        "chronos2_C1_inst", "chronos2_C1_macro",
        "timesfm_C1_inst", "timesfm_C1_macro",
        "timegpt_C1_inst", "timegpt_C1_macro",
    ]
    for model in foundation_models:
        path = RESULTS_DIR / f"{model}_predictions.parquet"
        if path.exists():
            all_metrics[model] = metrics_from_parquet(path, model, mase_scale)
            logger.info(f"  {model}: OK")
        else:
            logger.warning(f"  [!] {model}: not found")

    # ── Print comparison table ──
    logger.info("\n" + "=" * 90)
    logger.info("FINAL COMPARISON TABLE — MAE by horizon")
    logger.info("=" * 90)

    header = f"{'Model':<18}"
    for h in HORIZONS:
        header += f" {'h='+str(h):>8}"
    logger.info(header)
    logger.info("-" * 55)

    # Sort by h=1 MAE
    sorted_models = sorted(
        all_metrics.keys(),
        key=lambda m: all_metrics[m].get("h1", {}).get("MAE", 999)
    )

    for model in sorted_models:
        row = f"{model:<18}"
        for h in HORIZONS:
            key = f"h{h}"
            mae = all_metrics[model].get(key, {}).get("MAE", None)
            if mae is not None:
                row += f" {mae:8.4f}"
            else:
                row += f" {'N/A':>8}"
        logger.info(row)

    # ── Best model per horizon ──
    logger.info("\n--- Best model per horizon (MAE) ---")
    for h in HORIZONS:
        key = f"h{h}"
        best_model = None
        best_mae = float("inf")
        for model, metrics in all_metrics.items():
            mae = metrics.get(key, {}).get("MAE", None)
            if mae is not None and mae < best_mae:
                best_mae = mae
                best_model = model
        if best_model:
            logger.info(f"  h={h}: {best_model} (MAE={best_mae:.4f})")

    # ── C0 vs C1 delta table ──
    logger.info("\n--- C0 vs C1 delta (MAE) ---")
    families = [
        ("timesfm",      "timesfm_C0",  "timesfm_C1"),
        ("chronos2",     "chronos2_C0", "chronos2_C1"),
        ("timegpt",      "timegpt_C0",  "timegpt_C1"),
        ("chronos2_e",   "chronos2_C0", "chronos2_C1_energy"),
        ("timegpt_e",    "timegpt_C0",  "timegpt_C1_energy"),
        ("chronos2_eo",  "chronos2_C0", "chronos2_C1_energy_only"),
        ("timegpt_eo",   "timegpt_C0",  "timegpt_C1_energy_only"),
        ("chronos2_inst","chronos2_C0", "chronos2_C1_inst"),
        ("chronos2_macro","chronos2_C0","chronos2_C1_macro"),
        ("timesfm_inst", "timesfm_C0",  "timesfm_C1_inst"),
        ("timesfm_macro","timesfm_C0",  "timesfm_C1_macro"),
        ("timegpt_inst", "timegpt_C0",  "timegpt_C1_inst"),
        ("timegpt_macro","timegpt_C0",  "timegpt_C1_macro"),
    ]

    for family, c0, c1 in families:
        if c0 in all_metrics and c1 in all_metrics:
            row = f"  {family:<12}"
            for h in HORIZONS:
                key = f"h{h}"
                m0 = all_metrics[c0].get(key, {}).get("MAE")
                m1 = all_metrics[c1].get(key, {}).get("MAE")
                if m0 and m1:
                    delta_pct = (m1 - m0) / m0 * 100
                    row += f"  h{h}:{delta_pct:+.1f}%"
            logger.info(row)

    # ── Save JSON ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "metrics_summary_final.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
