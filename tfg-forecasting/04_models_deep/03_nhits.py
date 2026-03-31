"""
03_nhits.py -- N-HiTS para horizontes largos

N-HiTS (Challu et al., 2023): extension de N-BEATS con muestreo
jerarquico multi-resolucion. Diseñado especificamente para capturar
patrones a distintas escalas temporales, lo que lo hace especialmente
adecuado para h=12.

Si N-HiTS gana al LSTM en h=12, refuerza el argumento de que los
foundation models (diseñados para largo plazo) son el escalon natural.

Configuracion:
  - input_size = 24
  - n_pool_kernel_size = [2, 2, 1]  (pooling jerarquico)
  - max_steps = 500

Entrada:  data/processed/ipc_spain_index.parquet
Salida:   04_models_deep/results/nhits_metrics.json
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

warnings.filterwarnings("ignore")

from _helpers import (
    RESULTS_DIR, load_nf_format, evaluate_forecast, print_comparison
)

HORIZONS = [1, 3, 6, 12]


def train_and_evaluate(horizon):
    df_train, df_val, _, y_train_vals = load_nf_format()

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

    fc = nf.predict()
    fc = fc.reset_index()

    y_pred = fc["NHITS"].values[:horizon]
    y_true = df_val["y"].values[:horizon]
    dates  = df_val["ds"].values[:horizon]

    metrics = evaluate_forecast(y_true, y_pred, y_train_vals, f"nhits_h{horizon}")
    return metrics, dates, y_true, y_pred


def main():
    print("=" * 60)
    print("N-HiTS -- Modelo deep para horizontes largos")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for h in HORIZONS:
        print(f"\n--- Horizonte h={h} ---")
        metrics, dates, y_true, y_pred = train_and_evaluate(h)
        all_metrics[f"h{h}"] = metrics

        print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
        print_comparison(dates, y_true, y_pred)

    out_path = RESULTS_DIR / "nhits_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetricas guardadas: {out_path}")

    print(f"\n{'=' * 60}")
    print("RESUMEN N-HiTS")
    print(f"{'h':>4} {'MAE':>8} {'RMSE':>8} {'MASE':>8}")
    print("-" * 30)
    for h in HORIZONS:
        m = all_metrics[f"h{h}"]
        print(f"{h:>4} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MASE']:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
