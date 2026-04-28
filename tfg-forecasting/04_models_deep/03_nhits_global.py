"""
03_nhits_global.py -- N-HiTS para CPI Global (horizontes largos)

Analogo a 03_nhits.py (Espana) sobre la serie global.
El muestreo jerarquico multi-resolucion de N-HiTS es especialmente
util para h=12 donde ARIMA converge a la media.

Configuracion:
  - input_size = 24
  - n_pool_kernel_size = [2, 2, 1]
  - max_steps = 500

Entrada:  data/processed/cpi_global_monthly.parquet
Salida:   08_results/nhits_global_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers_global import RESULTS_DIR, load_nf_format_global, evaluate_forecast, print_comparison

HORIZONS = [1, 3, 6, 12]


def train_and_evaluate(horizon):
    df_train, df_val, _, y_train_vals = load_nf_format_global()

    model = NHITS(
        h=horizon,
        input_size=24,
        max_steps=500,
        scaler_type="standard",
        learning_rate=1e-3,
        n_pool_kernel_size=[2, 2, 1],
        random_seed=42,
        enable_progress_bar=False,
    )

    nf = NeuralForecast(models=[model], freq="MS")
    nf.fit(df=df_train)

    fc = nf.predict().reset_index()

    y_pred = fc["NHITS"].values[:horizon]
    y_true = df_val["y"].values[:horizon]
    dates  = df_val["ds"].values[:horizon]

    metrics = evaluate_forecast(y_true, y_pred, y_train_vals, f"nhits_global_h{horizon}")
    return metrics, dates, y_true, y_pred


def main():
    print("=" * 60)
    print("N-HiTS — CPI Global (horizontes largos)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for h in HORIZONS:
        print(f"\n--- Horizonte h={h} ---")
        metrics, dates, y_true, y_pred = train_and_evaluate(h)
        all_metrics[f"h{h}"] = metrics
        print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
        print_comparison(dates, y_true, y_pred)

    out_path = RESULTS_DIR / "nhits_global_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetricas guardadas: {out_path}")

    print(f"\n{'=' * 60}")
    print("RESUMEN N-HiTS GLOBAL")
    print(f"{'h':>4} {'MAE':>8} {'RMSE':>8} {'MASE':>8}")
    print("-" * 30)
    for h in HORIZONS:
        m = all_metrics[f"h{h}"]
        print(f"{h:>4} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MASE']:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
