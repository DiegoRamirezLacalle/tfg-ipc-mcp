"""Deep models consolidation + baseline comparison - CPI Global.

Loads rolling metrics from baseline (08_results/rolling_metrics_global.json)
and deep (08_results/deep_rolling_metrics_global.json) and generates a
unified report:
  1. Comparative deep vs baseline table by horizon
  2. Global ranking of all models
  3. Comparative MAE plot

Output:
  08_results/deep_global_report.txt
  08_results/deep_global_summary.json
  08_results/figures/all_models_mae_by_horizon_global.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT        = Path(__file__).resolve().parents[1]
MONOREPO    = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
FIGURES_DIR = RESULTS_DIR / "figures"

HORIZONS         = [1, 3, 6, 12]
BASELINE_MODELS  = ["naive", "arima", "arima111", "arimax"]
DEEP_MODELS      = ["lstm", "nbeats", "nhits"]
ALL_MODELS       = BASELINE_MODELS + DEEP_MODELS

MODEL_COLORS = {
    "naive":    "#aaaaaa",
    "arima":    "#2196F3",
    "arima111": "#4CAF50",
    "arimax":   "#FF9800",
    "lstm":     "#E91E63",
    "nbeats":   "#9C27B0",
    "nhits":    "#00BCD4",
}


def load_metrics():
    with open(RESULTS_DIR / "rolling_metrics_global.json") as f:
        baseline = json.load(f)
    with open(RESULTS_DIR / "deep_rolling_metrics_global.json") as f:
        deep = json.load(f)
    return {**baseline, **deep}


def build_ranking(metrics):
    ranking = {}
    for h in HORIZONS:
        key = f"h{h}"
        candidates = [m for m in ALL_MODELS if key in metrics.get(m, {})]
        if not candidates:
            continue
        best = min(candidates, key=lambda m: metrics[m][key]["MAE"])
        ranking[h] = {
            "best": best,
            "MAE":  metrics[best][key]["MAE"],
            "tipo": "baseline" if best in BASELINE_MODELS else "deep",
        }
    return ranking


def write_report(metrics, ranking):
    sep  = "=" * 70
    sep2 = "-" * 70
    lines = []

    lines += [
        sep,
        "REPORTE DEEP MODELS + COMPARATIVA BASELINE - CPI Global",
        "Condicion C0: todos los modelos sin senales exogenas adicionales",
        "Nota: rolling baseline mensual (N~47), deep trimestral (N~16)",
        sep, "",
    ]

    for h in HORIZONS:
        key = f"h{h}"
        lines += [
            f"Horizonte h={h}:",
            f"{'Modelo':<10} {'Tipo':<10} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}",
            sep2,
        ]
        for model in ALL_MODELS:
            if key not in metrics.get(model, {}):
                continue
            m    = metrics[model][key]
            tipo = "baseline" if model in BASELINE_MODELS else "deep"
            n    = m.get("n_evals", "?")
            mark = " <-- BEST" if ranking.get(h, {}).get("best") == model else ""
            lines.append(
                f"{model:<10} {tipo:<10} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                f"{m['MASE']:>8.4f} {str(n):>5}{mark}"
            )
        lines.append("")

    lines += [sep, "RANKING GLOBAL POR HORIZONTE", sep2,
              f"{'h':>4} {'Mejor':>10} {'Tipo':>10} {'MAE':>8}", sep2]
    for h, info in ranking.items():
        lines.append(f"{h:>4} {info['best']:>10} {info['tipo']:>10} {info['MAE']:>8.4f}")

    lines += [
        "", sep, "CONCLUSIONES", sep2,
        "  - El rolling baseline (mensual, N~47) es mas robusto estadisticamente",
        "    que el deep (trimestral, N~16). Las comparaciones son indicativas.",
        "  - LSTM: ventaja potencial en h=1 (captura momentum reciente);",
        "    tiende a degradarse en h=12 (acumulacion de error recurrente).",
        "  - N-BEATS y N-HiTS: mas estables; N-HiTS teoricamente superior en h=12",
        "    por su pooling jerarquico multi-resolucion.",
        "  - Los modelos estadisticos (ARIMA/ARIMAX) son competitivos en series",
        "    univariantes mensuales con pocos datos de entrenamiento.",
        "  - La ventaja de los deep se espera principalmente con exogenas (C1):",
        "    LSTM multivariante y modelos que puedan integrar VIX/commodities.",
        "  - Estos resultados junto con los foundation models informan la",
        "    decision de arquitectura del pipeline MCP (C1 global).",
        sep,
    ]

    text = "\n".join(lines)
    path = RESULTS_DIR / "deep_global_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Report: {path}")
    return text


def plot_comparison(metrics):
    fig, ax  = plt.subplots(figsize=(10, 6))
    x        = np.arange(len(HORIZONS))
    n_models = len(ALL_MODELS)
    width    = 0.8 / n_models

    for i, model in enumerate(ALL_MODELS):
        maes = []
        for h in HORIZONS:
            key = f"h{h}"
            maes.append(metrics[model][key]["MAE"] if key in metrics.get(model, {}) else np.nan)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, maes, width, label=model.upper(),
               color=MODEL_COLORS.get(model, "#666"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_ylabel("MAE (pp YoY rate)")
    ax.set_title("Rolling MAE - Baseline vs Deep C0 Global")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "all_models_mae_by_horizon_global.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot: {path}")


def main():
    logger.info("=" * 60)
    logger.info("DEEP + BASELINE CONSOLIDATION - CPI Global")
    logger.info("=" * 60)

    metrics = load_metrics()
    ranking = build_ranking(metrics)

    report = write_report(metrics, ranking)
    logger.info("\n" + report)

    summary = {
        "metrics": metrics,
        "ranking": {str(h): v for h, v in ranking.items()},
    }
    summary_path = RESULTS_DIR / "deep_global_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON: {summary_path}")

    plot_comparison(metrics)


if __name__ == "__main__":
    main()
