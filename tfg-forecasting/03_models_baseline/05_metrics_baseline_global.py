"""Consolidated report and final evaluation for Global CPI baseline models (C0).

Analogous to 05_metrics_baseline.py (Spain) but adapted to the global pipeline:
  - Models: naive, arima (3,1,0), arima111 (1,1,1), arimax (3,1,0)+FEDFUNDS
  - Static metrics from 08_results/{arima,arima111,arimax}_global_metrics.json
  - Rolling from 08_results/rolling_metrics_global.json
  - Economic periods:
      A) Pre-crisis:     2021-01 to 2022-06  (low inflation, Fed flat 0-0.25%)
      B) Shock:          2022-07 to 2023-06  (inflation peak, Fed hikes)
      C) Normalisation:  2023-07 to 2024-12  (global disinflation)

Output:
  08_results/baseline_global_report.txt
  08_results/baseline_global_summary.json
  08_results/figures/rolling_mae_by_horizon_global.png
  08_results/figures/error_by_period_global.png
  08_results/figures/rolling_errors_h1_h12_global.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]
MONOREPO    = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = ROOT / "08_results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODELS   = ["naive", "arima", "arima111", "arimax"]
HORIZONS = [1, 3, 6, 12]

PERIODS = {
    "A_pre_crisis":    ("2021-01-01", "2022-06-01"),
    "B_shock":         ("2022-07-01", "2023-06-01"),
    "C_normalizacion": ("2023-07-01", "2024-12-01"),
}
PERIOD_LABELS = {
    "A_pre_crisis":    "Pre-crisis    (2021-01 / 2022-06)",
    "B_shock":         "Shock Fed     (2022-07 / 2023-06)",
    "C_normalizacion": "Normalizacion (2023-07 / 2024-12)",
}
MODEL_COLORS = {
    "naive":    "#aaaaaa",
    "arima":    "#2196F3",
    "arima111": "#4CAF50",
    "arimax":   "#FF9800",
}


def load_all():
    static = {}
    for m in ["arima", "arima111", "arimax"]:
        path = RESULTS_DIR / f"{m}_global_metrics.json"
        with open(path) as f:
            static[m] = json.load(f)

    with open(RESULTS_DIR / "rolling_metrics_global.json") as f:
        rolling = json.load(f)

    preds = pd.read_parquet(RESULTS_DIR / "rolling_predictions_global.parquet")
    preds["fc_date"] = pd.to_datetime(preds["fc_date"])
    preds["origin"]  = pd.to_datetime(preds["origin"])

    return static, rolling, preds


def metrics_by_period(preds: pd.DataFrame) -> dict:
    results = {}
    for period_key, (start, end) in PERIODS.items():
        mask = (preds["fc_date"] >= start) & (preds["fc_date"] <= end)
        p_df = preds[mask]
        results[period_key] = {}
        for model in MODELS:
            results[period_key][model] = {}
            m_df = p_df[p_df["model"] == model]
            for h in [1, 12]:
                h_df = m_df[m_df["horizon"] == h]
                if h_df.empty:
                    continue
                results[period_key][model][f"h{h}"] = {
                    "MAE": round(float(np.mean(h_df["abs_error"])), 4),
                    "n":   len(h_df["origin"].unique()),
                }
    return results


def build_ranking(rolling: dict) -> dict:
    ranking = {}
    for h in HORIZONS:
        key = f"h{h}"
        candidates = [m for m in MODELS if key in rolling.get(m, {})]
        best = min(candidates, key=lambda m: rolling[m][key]["MAE"])
        ranking[h] = {
            "best":         best,
            "MAE":          rolling[best][key]["MAE"],
            "vs_naive_pct": round(
                (1 - rolling[best][key]["MAE"] / rolling["naive"][key]["MAE"]) * 100, 1
            ),
        }
    return ranking


def fedfunds_benefit(rolling: dict) -> list:
    rows = []
    for h in HORIZONS:
        key = f"h{h}"
        if key in rolling.get("arima", {}) and key in rolling.get("arimax", {}):
            mae_a  = rolling["arima"][key]["MAE"]
            mae_ax = rolling["arimax"][key]["MAE"]
            delta  = mae_ax - mae_a
            pct    = -delta / mae_a * 100
            rows.append({"h": h, "mae_arima": mae_a, "mae_arimax": mae_ax,
                          "delta": delta, "improvement_pct": pct})
    return rows


def write_report(static, rolling, period_metrics, ranking, ff_benefit):
    sep  = "=" * 70
    sep2 = "-" * 70
    lines = []

    lines += [
        sep,
        "REPORTE BASELINE -- CPI Global (mediana cross-country HCPI_M)",
        "Condicion C0: solo datos historicos numericos",
        sep, "",
        "1. ESPECIFICACIONES DE MODELOS",
        sep2,
        f"{'Modelo':<12} {'Orden':<18} {'AIC':>10} {'BIC':>10} {'N train':>8}",
        sep2,
    ]
    specs = {
        "arima":    ("ARIMA(3,1,0)",         static["arima"]["aic"],    static["arima"]["bic"],    static["arima"]["n_train"]),
        "arima111": ("ARIMA(1,1,1)",          static["arima111"]["aic"], static["arima111"]["bic"], static["arima111"]["n_train"]),
        "arimax":   ("ARIMA(3,1,0)+FEDFUNDS", static["arimax"]["aic"],  static["arimax"]["bic"],   static["arimax"]["n_train"]),
    }
    for m, (label, aic, bic, n) in specs.items():
        lines.append(f"{m:<12} {label:<18} {aic:>10.2f} {bic:>10.2f} {n:>8d}")
    lines += [
        "",
        "  Nota: D=0 (sin diferenciacion estacional). Fs=-0.08 -> la mediana",
        "  cross-country cancela los patrones estacionales nacionales.",
        "  FEDFUNDS usado como exogena (analogo al DFR en pipeline Espana).",
    ]

    lines += [
        "", "2. EVALUACION ESTATICA (train 2002-2020 / val 2021-01 a 2022-06)",
        "   [Ruptura estructural: inflacion post-COVID 2021-2022]",
        sep2,
        f"{'Modelo':<12} {'MAE':>10} {'RMSE':>10} {'MASE':>10}",
        sep2,
    ]
    for m in ["arima", "arima111", "arimax"]:
        mv = static[m]["metrics_val"]
        lines.append(f"{m:<12} {mv['MAE']:>10.4f} {mv['RMSE']:>10.4f} {mv['MASE']:>10.4f}")
    lines += [
        "   [MASE~2.5 esperado: tasa recorre 1.7%->8.2% y ARIMA converge a",
        "    ~1.48%. Motiva el uso de senales exogenas en condicion C1.]",
    ]

    lines += ["", "3. EVALUACION ROLLING EXPANDING-WINDOW (2021-01 a 2024-12)", sep2]
    for h in HORIZONS:
        key = f"h{h}"
        n = rolling.get("arima", {}).get(key, {}).get("n_evals", "?")
        lines += [
            f"  h={h} meses ({n} evaluaciones):",
            f"  {'Modelo':<10} {'MAE':>8} {'RMSE':>8} {'MASE':>8}  vs naive",
            f"  {'-'*46}",
        ]
        for model in MODELS:
            if key not in rolling.get(model, {}):
                continue
            m  = rolling[model][key]
            ratio = m["MAE"] / rolling["naive"][key]["MAE"]
            mark  = " *" if ratio < 1.0 else ""
            lines.append(
                f"  {model:<10} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                f"{m['MASE']:>8.4f}  {ratio:.3f}x{mark}"
            )
        lines.append("")

    lines += ["4. DESGLOSE POR PERIODO ECONOMICO (MAE, h=1 y h=12)", sep2]
    for period_key, label in PERIOD_LABELS.items():
        lines += [f"  {label}:", f"  {'Modelo':<10} {'MAE h=1':>10} {'MAE h=12':>10}"]
        for model in MODELS:
            m1  = period_metrics[period_key][model].get("h1",  {}).get("MAE", float("nan"))
            m12 = period_metrics[period_key][model].get("h12", {}).get("MAE", float("nan"))
            s1  = f"{m1:>10.4f}"  if not np.isnan(m1)  else f"{'N/A':>10}"
            s12 = f"{m12:>10.4f}" if not np.isnan(m12) else f"{'N/A':>10}"
            lines.append(f"  {model:<10} {s1} {s12}")
        lines.append("")

    lines += [
        "5. RANKING DE MODELOS POR HORIZONTE",
        sep2,
        f"{'h':>4} {'Mejor modelo':<12} {'MAE':>8} {'Mejora vs naive':>16}",
        sep2,
    ]
    for h, info in ranking.items():
        lines.append(
            f"{h:>4} {info['best']:<12} {info['MAE']:>8.4f} "
            f"{info['vs_naive_pct']:>14.1f}%"
        )

    lines += [
        "", "6. BENEFICIO FEDFUNDS (ARIMA vs ARIMAX, delta MAE rolling)",
        sep2,
        f"  {'h':>4}  {'MAE ARIMA':>12}  {'MAE ARIMAX':>12}  {'Delta':>8}  {'Mejora%':>8}",
        f"  {'-'*52}",
    ]
    for r in ff_benefit:
        mark = " <-- mejora" if r["improvement_pct"] > 0 else ""
        lines.append(
            f"  {r['h']:>4}  {r['mae_arima']:>12.4f}  {r['mae_arimax']:>12.4f}"
            f"  {r['delta']:>+8.4f}  {r['improvement_pct']:>+7.1f}%{mark}"
        )

    lines += [
        "", "7. CONCLUSIONES CLAVE", sep2,
        "  - Todos los modelos baten al naive estacional en todos los horizontes",
        "    (MASE < 1 en rolling). Diferencia enorme respecto a val estatica",
        "    (MASE~2.5): el rolling captura la tendencia de desinflacion 2023.",
        "  - ARIMA(3,1,0) domina ligeramente frente a ARIMA(1,1,1) en h>=6.",
        "  - ARIMAX+FEDFUNDS mejora al ARIMA en todos los horizontes, con mayor",
        "    beneficio en h=1 (+1.6%) y h=12 (+1.4%), confirmando que la",
        "    politica monetaria de la Fed aporta senal real en el shock 2022-2024.",
        "  - El periodo B (Shock Fed, 07/22-06/23) concentra el mayor error.",
        "    Motiva el uso de senales globales (VIX, commodities) en C1.",
        "  - Estos resultados forman la linea base C0 del experimento TFG global.",
        sep,
    ]

    text = "\n".join(lines)
    path = RESULTS_DIR / "baseline_global_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Report saved: {path}")
    return text


def plot_mae_by_horizon(rolling: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(HORIZONS))
    width = 0.18

    for i, model in enumerate(MODELS):
        maes = [rolling[model][f"h{h}"]["MAE"] for h in HORIZONS
                if f"h{h}" in rolling.get(model, {})]
        offset = (i - 1.5) * width
        ax.bar(x + offset, maes, width, label=model.upper(),
               color=MODEL_COLORS[model], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_ylabel("MAE (pp YoY rate)")
    ax.set_title("MAE rolling by horizon - Baseline C0 Global")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "rolling_mae_by_horizon_global.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


def plot_error_by_period(period_metrics: dict) -> None:
    period_keys = list(PERIOD_LABELS.keys())
    models_plot = ["arima", "arima111", "arimax"]
    fig, axes   = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax_idx, h in enumerate([1, 12]):
        ax  = axes[ax_idx]
        key = f"h{h}"
        x   = np.arange(len(period_keys))
        w   = 0.22

        naive_maes = [
            period_metrics[pk]["naive"].get(key, {}).get("MAE", np.nan)
            for pk in period_keys
        ]
        ax.plot(x, naive_maes, "o--", color=MODEL_COLORS["naive"],
                label="NAIVE", linewidth=1.5, markersize=5)

        for i, model in enumerate(models_plot):
            maes = [
                period_metrics[pk][model].get(key, {}).get("MAE", np.nan)
                for pk in period_keys
            ]
            ax.bar(x + (i - 1) * w, maes, w, label=model.upper(),
                   color=MODEL_COLORS[model], alpha=0.82, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(["Pre-crisis\n01/21-06/22", "Shock Fed\n07/22-06/23",
                             "Normaliz.\n07/23-12/24"], fontsize=8)
        ax.set_ylabel("MAE (pp YoY rate)")
        ax.set_title(f"MAE by period - h={h} month{'s' if h > 1 else ''}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Performance by economic period - Baseline C0 Global", fontsize=11)
    fig.tight_layout()
    path = FIGURES_DIR / "error_by_period_global.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


def plot_rolling_errors_over_time(preds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    for ax_idx, h in enumerate([1, 12]):
        ax   = axes[ax_idx]
        h_df = preds[preds["horizon"] == h].copy()

        for model in MODELS:
            m_df = (h_df[h_df["model"] == model]
                    .groupby("fc_date")["abs_error"].mean()
                    .sort_index())
            ax.plot(m_df.index, m_df.values, label=model.upper(),
                    color=MODEL_COLORS[model],
                    linewidth=1.8 if model != "naive" else 1.0,
                    linestyle="--" if model == "naive" else "-",
                    alpha=0.9)

        ax.axvspan(pd.Timestamp("2022-07-01"), pd.Timestamp("2023-06-01"),
                   alpha=0.08, color="red", label="_Shock Fed")
        ax.set_ylabel("Absolute error (pp)")
        ax.set_title(f"Rolling absolute error - h={h} month{'s' if h > 1 else ''}")
        ax.legend(ncol=5, fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Rolling error over time - Baseline C0 Global", fontsize=11)
    fig.tight_layout()
    path = FIGURES_DIR / "rolling_errors_h1_h12_global.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


def plot_fedfunds_benefit(ff_benefit: list) -> None:
    hs     = [r["h"] for r in ff_benefit]
    deltas = [r["improvement_pct"] for r in ff_benefit]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = ["#4CAF50" if d > 0 else "#F44336" for d in deltas]
    ax.bar([f"h={h}" for h in hs], deltas, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MAE improvement ARIMAX vs ARIMA (%)")
    ax.set_title("FEDFUNDS benefit in rolling - CPI Global")
    ax.grid(axis="y", alpha=0.3)
    for i, (h, d) in enumerate(zip(hs, deltas)):
        ax.text(i, d + 0.05 * (1 if d >= 0 else -1), f"{d:+.1f}%",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "fedfunds_benefit_global.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


def main():
    logger.info("=" * 60)
    logger.info("BASELINE CONSOLIDATION - CPI Global")
    logger.info("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    static, rolling, preds = load_all()
    period_metrics = metrics_by_period(preds)
    ranking        = build_ranking(rolling)
    ff_benefit     = fedfunds_benefit(rolling)

    report = write_report(static, rolling, period_metrics, ranking, ff_benefit)
    logger.info("\n" + report)

    summary = {
        "static_val":     {m: static[m]["metrics_val"] for m in ["arima", "arima111", "arimax"]},
        "rolling":        rolling,
        "period_metrics": period_metrics,
        "ranking":        {str(h): v for h, v in ranking.items()},
        "fedfunds_benefit": ff_benefit,
    }
    summary_path = RESULTS_DIR / "baseline_global_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON: {summary_path}")

    logger.info("Generating plots...")
    plot_mae_by_horizon(rolling)
    plot_error_by_period(period_metrics)
    plot_rolling_errors_over_time(preds)
    plot_fedfunds_benefit(ff_benefit)

    logger.info("Files generated in 08_results/:")
    for p in sorted(RESULTS_DIR.glob("*global*")):
        if p.is_file():
            logger.info(f"  {p.name}")
    for p in sorted(FIGURES_DIR.glob("*global*")):
        if p.is_file():
            logger.info(f"  figures/{p.name}")


if __name__ == "__main__":
    main()
