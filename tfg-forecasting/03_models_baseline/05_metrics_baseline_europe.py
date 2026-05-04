"""Baseline metrics report for HICP Eurozone.

Reads rolling_metrics_europe.json and generates a text report + consolidated JSON.

Output:
  08_results/baseline_europe_report.txt
  08_results/baseline_europe_summary.json
"""

import json
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
MODELS   = ["naive", "sarima", "sarimax"]


def main():
    metrics_path = RESULTS_DIR / "rolling_metrics_europe.json"
    if not metrics_path.exists():
        logger.error("rolling_metrics_europe.json not found. Run script 04 first.")
        return

    metrics = json.loads(metrics_path.read_text())

    lines = []
    lines.append("=" * 65)
    lines.append("INFORME BASELINE — HICP Eurozona")
    lines.append("  Rolling expanding-window | Origenes 2021-01 a 2024-12")
    lines.append("  Metrica principal: MAE (puntos de indice) | MASE (vs naive lag-12)")
    lines.append("=" * 65)

    # MAE table
    lines.append(f"\n{'Modelo':<12}" + "".join(f"  h={h:>2} MAE" for h in HORIZONS))
    lines.append("-" * 55)
    for model in MODELS:
        row = f"{model:<12}"
        for h in HORIZONS:
            key = f"h{h}"
            v = metrics.get(model, {}).get(key, {}).get("MAE")
            ref = metrics.get("naive", {}).get(key, {}).get("MAE")
            if v is not None:
                mark = "*" if ref and v < ref else " "
                row += f"  {v:>7.4f}{mark}"
            else:
                row += f"  {'N/A':>8}"
        lines.append(row)
    lines.append("  (* = bate al naive lag-12)")

    # MASE table
    lines.append(f"\n{'Modelo':<12}" + "".join(f"  h={h:>2} MASE" for h in HORIZONS))
    lines.append("-" * 58)
    for model in MODELS:
        row = f"{model:<12}"
        for h in HORIZONS:
            key = f"h{h}"
            v = metrics.get(model, {}).get(key, {}).get("MASE")
            row += f"  {v:>8.4f}" if v is not None else f"  {'N/A':>8}"
        lines.append(row)

    # DFR benefit
    lines.append("\n--- Beneficio DFR (SARIMAX vs SARIMA) ---")
    lines.append(f"{'h':>4}  {'SARIMA':>10}  {'SARIMAX':>10}  {'Delta%':>8}")
    lines.append("-" * 38)
    for h in HORIZONS:
        key = f"h{h}"
        ms = metrics.get("sarima",  {}).get(key, {}).get("MAE")
        mx = metrics.get("sarimax", {}).get(key, {}).get("MAE")
        if ms and mx:
            pct = (mx - ms) / ms * 100
            mark = " <mejora" if pct < 0 else ""
            lines.append(f"{h:>4}  {ms:>10.4f}  {mx:>10.4f}  {pct:>+7.1f}%{mark}")

    # Best model per horizon
    lines.append("\n--- Mejor modelo por horizonte ---")
    for h in HORIZONS:
        key = f"h{h}"
        best_model = min(
            (m for m in MODELS if key in metrics.get(m, {})),
            key=lambda m: metrics[m][key]["MAE"],
            default="N/A"
        )
        best_mae = metrics.get(best_model, {}).get(key, {}).get("MAE", float("nan"))
        lines.append(f"  h={h}: {best_model:<10}  MAE={best_mae:.4f}")

    lines.append("\n" + "=" * 65)

    report = "\n".join(lines)
    logger.info("\n" + report)

    (RESULTS_DIR / "baseline_europe_report.txt").write_text(report, encoding="utf-8")
    with open(RESULTS_DIR / "baseline_europe_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\nSaved: baseline_europe_report.txt  |  baseline_europe_summary.json")


if __name__ == "__main__":
    main()
