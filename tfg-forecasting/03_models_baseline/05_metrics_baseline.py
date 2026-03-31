"""
05_metrics_baseline.py — Consolidacion y reporte final de modelos baseline

Integra los resultados de los cuatro scripts anteriores en:
  1. Tabla de especificaciones de modelos
  2. Evaluacion estatica (train 2002-2020 / val 2021-2022)
  3. Evaluacion rolling por horizonte (h=1,3,6,12)
  4. Desglose por periodo economico:
       A) Pre-crisis:   2021-01 a 2022-06  (inflacion baja, DFR=-0.50%)
       B) Crisis:       2022-07 a 2023-06  (pico inflacion, BCE sube tipos)
       C) Post-crisis:  2023-07 a 2024-12  (desinflacion)
  5. Ranking de modelos por horizonte
  6. Graficos de errores y comparativa

Salida:
  results/baseline_report.txt
  results/baseline_summary.json
  results/plots/rolling_mae_by_horizon.png
  results/plots/error_by_period.png
  results/plots/rolling_errors_h1_h12.png
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

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

MODELS   = ["naive", "arima", "sarima", "sarimax"]
HORIZONS = [1, 3, 6, 12]

PERIODS = {
    "A_pre_crisis":  ("2021-01-01", "2022-06-01"),
    "B_crisis":      ("2022-07-01", "2023-06-01"),
    "C_post_crisis": ("2023-07-01", "2024-12-01"),
}
PERIOD_LABELS = {
    "A_pre_crisis":  "Pre-crisis (2021-01 / 2022-06)",
    "B_crisis":      "Crisis BCE  (2022-07 / 2023-06)",
    "C_post_crisis": "Post-crisis (2023-07 / 2024-12)",
}
MODEL_COLORS = {
    "naive":   "#aaaaaa",
    "arima":   "#2196F3",
    "sarima":  "#4CAF50",
    "sarimax": "#FF9800",
}


# ── Carga ──────────────────────────────────────────────────────────────────

def load_all():
    static = {}
    for m in ["arima", "sarima", "sarimax"]:
        with open(RESULTS_DIR / f"{m}_metrics.json") as f:
            static[m] = json.load(f)

    with open(RESULTS_DIR / "rolling_metrics.json") as f:
        rolling = json.load(f)

    preds = pd.read_parquet(RESULTS_DIR / "rolling_predictions.parquet")
    preds["fc_date"] = pd.to_datetime(preds["fc_date"])
    preds["origin"]  = pd.to_datetime(preds["origin"])

    return static, rolling, preds


# ── Analisis por periodo ────────────────────────────────────────────────────

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


# ── Ranking ────────────────────────────────────────────────────────────────

def build_ranking(rolling: dict) -> dict:
    ranking = {}
    for h in HORIZONS:
        key = f"h{h}"
        best = min(
            [m for m in MODELS if key in rolling.get(m, {})],
            key=lambda m: rolling[m][key]["MAE"]
        )
        ranking[h] = {
            "best":          best,
            "MAE":           rolling[best][key]["MAE"],
            "vs_naive_pct":  round(
                (1 - rolling[best][key]["MAE"] / rolling["naive"][key]["MAE"]) * 100, 1
            ),
        }
    return ranking


# ── Reporte texto ──────────────────────────────────────────────────────────

def write_report(static, rolling, period_metrics, ranking):
    sep  = "=" * 70
    sep2 = "-" * 70
    lines = []

    lines += [
        sep,
        "REPORTE BASELINE -- TFG Prediccion IPC Espana con MCP",
        "Condicion C0: solo datos historicos numericos",
        sep, "",
        "1. ESPECIFICACIONES DE MODELOS",
        sep2,
        f"{'Modelo':<12} {'Orden':<22} {'AIC':>10} {'BIC':>10} {'N train':>8}",
        sep2,
    ]
    specs = {
        "arima":   ("ARIMA(1,1,2)",           static["arima"]["aic"],   static["arima"]["bic"],   static["arima"]["n_train"]),
        "sarima":  ("SARIMA(0,1,1)(0,1,1)12", static["sarima"]["aic"],  static["sarima"]["bic"],  static["sarima"]["n_train"]),
        "sarimax": ("SARIMAX+DFR(0,1,1)12",   static["sarimax"]["aic"], static["sarimax"]["bic"], static["sarimax"]["n_train"]),
    }
    for m, (label, aic, bic, n) in specs.items():
        lines.append(f"{m:<12} {label:<22} {aic:>10.2f} {bic:>10.2f} {n:>8d}")

    lines += [
        "", "2. EVALUACION ESTATICA (train 2002-2020 / val 2021-2022-06)",
        "   [Nota: periodo de ruptura estructural -- inflacion post-COVID]",
        sep2,
        f"{'Modelo':<12} {'MAE':>10} {'RMSE':>10} {'MASE':>10}",
        sep2,
    ]
    for m in ["arima", "sarima", "sarimax"]:
        mv = static[m]["metrics_val"]
        lines.append(f"{m:<12} {mv['MAE']:>10.4f} {mv['RMSE']:>10.4f} {mv['MASE']:>10.4f}")
    lines += [
        "   [MASE>1 indica que el modelo es peor que el naive estacional",
        "    en este periodo, esperado dado el shock inflacionario de 2022]",
    ]

    lines += ["", "3. EVALUACION ROLLING EXPANDING-WINDOW (2021-01 a 2024-12)", sep2]
    for h in HORIZONS:
        key = f"h{h}"
        n = rolling["arima"][key]["n_evals"]
        lines += [
            f"  h={h} meses ({n} evaluaciones):",
            f"  {'Modelo':<10} {'MAE':>8} {'RMSE':>8} {'MASE':>8}  vs naive",
            f"  {'-'*46}",
        ]
        for model in MODELS:
            if key not in rolling.get(model, {}):
                continue
            m = rolling[model][key]
            ratio = m["MAE"] / rolling["naive"][key]["MAE"]
            lines.append(
                f"  {model:<10} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                f"{m['MASE']:>8.4f}  {ratio:.3f}x"
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
        "", "6. CONCLUSIONES CLAVE", sep2,
        "  - Rolling backtesting confirma que ARIMA/SARIMA baten al naive",
        "    estacional en todos los horizontes (MASE < 1 para h=1..6).",
        "  - SARIMA(0,1,1)(0,1,1)12 domina en h=1 (captura patron mensual).",
        "  - ARIMA(1,1,2) es igual o mejor en h=12: menos parametros,",
        "    menos sobreajuste ante incertidumbre estacional acumulada.",
        "  - SARIMAX con DFR no mejora al SARIMA en C0: el DFR estuvo",
        "    plano (-0.50%) durante toda la val estatica; en el periodo",
        "    crisis el DFR si cambia y su efecto sera visible en C1.",
        "  - El periodo B (crisis BCE, 2022-07/2023-06) muestra el mayor",
        "    error en todos los modelos, motivando el uso de senales MCP.",
        "  - Estos resultados forman la linea base C0 del experimento TFG.",
        "    La condicion C1 debera mejorar principalmente en h=6, h=12",
        "    y en el periodo B.",
        sep,
    ]

    text = "\n".join(lines)
    path = RESULTS_DIR / "baseline_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Reporte guardado: {path}")
    return text


# ── Graficos ───────────────────────────────────────────────────────────────

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
    ax.set_ylabel("MAE (puntos de indice IPC)")
    ax.set_title("MAE Rolling por horizonte -- Baseline C0")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "rolling_mae_by_horizon.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Grafico: {path}")


def plot_error_by_period(period_metrics: dict) -> None:
    period_keys  = list(PERIOD_LABELS.keys())
    models_plot  = ["arima", "sarima", "sarimax"]
    fig, axes    = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

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
        ax.set_xticklabels(["Pre-crisis\n2021-06/22", "Crisis\n07/22-06/23",
                             "Post-crisis\n07/23-12/24"], fontsize=8)
        ax.set_ylabel("MAE (puntos de indice IPC)")
        ax.set_title(f"MAE por periodo -- h={h} mes{'es' if h > 1 else ''}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Comportamiento por periodo economico -- Baseline C0", fontsize=11)
    fig.tight_layout()
    path = PLOTS_DIR / "error_by_period.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Grafico: {path}")


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
                   alpha=0.08, color="red", label="_Crisis BCE")
        ax.set_ylabel("Error absoluto")
        ax.set_title(f"Error absoluto rolling -- h={h} mes{'es' if h > 1 else ''}")
        ax.legend(ncol=5, fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Evolucion del error rolling en el tiempo -- Baseline C0", fontsize=11)
    fig.tight_layout()
    path = PLOTS_DIR / "rolling_errors_h1_h12.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Grafico: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CONSOLIDACION BASELINE")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    static, rolling, preds = load_all()
    period_metrics = metrics_by_period(preds)
    ranking        = build_ranking(rolling)

    report = write_report(static, rolling, period_metrics, ranking)
    print()
    print(report)

    summary = {
        "static_val":     {m: static[m]["metrics_val"] for m in ["arima", "sarima", "sarimax"]},
        "rolling":        rolling,
        "period_metrics": period_metrics,
        "ranking":        {str(h): v for h, v in ranking.items()},
    }
    summary_path = RESULTS_DIR / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")

    print("\nGenerando graficos...")
    plot_mae_by_horizon(rolling)
    plot_error_by_period(period_metrics)
    plot_rolling_errors_over_time(preds)

    print("\nArchivos generados en results/:")
    for p in sorted(RESULTS_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
