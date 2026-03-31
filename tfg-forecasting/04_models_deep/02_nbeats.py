"""
02_nbeats.py -- N-BEATS como baseline deep principal

N-BEATS (Oreshkin et al., 2020): arquitectura puramente feedforward con
bloques residuales. Mas estable y reproducible que LSTM, rendimiento
competitivo sin necesidad de tuning extensivo.

Se usa la variante 'generic' (sin interpretabilidad), que es la que
mejor funciona para forecasting puro.

Configuracion:
  - input_size = 24 (2 anos de contexto)
  - n_blocks = [1, 1]  (stacks trend + seasonality)
  - mlp_units = [[256, 256], [256, 256]]
  - max_steps = 500

Entrada:  data/processed/ipc_spain_index.parquet
Salida:   04_models_deep/results/nbeats_metrics.json
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

warnings.filterwarnings("ignore")

from _helpers import (
    RESULTS_DIR, load_nf_format, evaluate_forecast, print_comparison
)

HORIZONS = [1, 3, 6, 12]


def train_and_evaluate(horizon):
    df_train, df_val, _, y_train_vals = load_nf_format()

    # Para h < 4 los stacks trend/seasonality no son compatibles (requieren
    # h >= 2*n_harmonics), asi que usamos stacks genericos (identity).
    if horizon < 4:
        stacks = ["identity", "identity", "identity"]
    else:
        stacks = ["identity", "trend", "seasonality"]

    model = NBEATS(
        h=horizon,
        input_size=24,
        stack_types=stacks,
        max_steps=500,
        scaler_type="standard",
        learning_rate=1e-3,
        random_seed=42,
        enable_progress_bar=False,
    )

    nf = NeuralForecast(models=[model], freq="MS")
    nf.fit(df=df_train)

    fc = nf.predict()
    fc = fc.reset_index()

    y_pred = fc["NBEATS"].values[:horizon]
    y_true = df_val["y"].values[:horizon]
    dates  = df_val["ds"].values[:horizon]

    metrics = evaluate_forecast(y_true, y_pred, y_train_vals, f"nbeats_h{horizon}")
    return metrics, dates, y_true, y_pred


def main():
    print("=" * 60)
    print("N-BEATS -- Baseline deep principal")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for h in HORIZONS:
        print(f"\n--- Horizonte h={h} ---")
        metrics, dates, y_true, y_pred = train_and_evaluate(h)
        all_metrics[f"h{h}"] = metrics

        print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
        print_comparison(dates, y_true, y_pred)

    out_path = RESULTS_DIR / "nbeats_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetricas guardadas: {out_path}")

    print(f"\n{'=' * 60}")
    print("RESUMEN N-BEATS")
    print(f"{'h':>4} {'MAE':>8} {'RMSE':>8} {'MASE':>8}")
    print("-" * 30)
    for h in HORIZONS:
        m = all_metrics[f"h{h}"]
        print(f"{h:>4} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MASE']:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
