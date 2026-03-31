"""
01_lstm_univariate.py -- LSTM univariante como referencia historica

Incluido porque es el baseline deep mas citado en la literatura de
prediccion de series temporales. Usa NeuralForecast para consistencia
con N-BEATS y N-HiTS.

Configuracion:
  - input_size = 24 (2 anos de contexto)
  - hidden_size = 64
  - max_steps = 500
  - horizons evaluados: 1, 3, 6, 12

Entrada:  data/processed/ipc_spain_index.parquet
Salida:   04_models_deep/results/lstm_metrics.json
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM

warnings.filterwarnings("ignore")

from _helpers import (
    RESULTS_DIR, load_nf_format, evaluate_forecast, print_comparison
)

HORIZONS = [1, 3, 6, 12]


def train_and_evaluate(horizon):
    """Entrena LSTM para un horizonte especifico y evalua sobre validacion."""
    df_train, df_val, _, y_train_vals = load_nf_format()

    model = LSTM(
        h=horizon,
        input_size=24,
        encoder_hidden_size=64,
        decoder_hidden_size=64,
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

    # Alinear con actuals
    y_pred = fc["LSTM"].values[:horizon]
    y_true = df_val["y"].values[:horizon]
    dates  = df_val["ds"].values[:horizon]

    metrics = evaluate_forecast(y_true, y_pred, y_train_vals, f"lstm_h{horizon}")
    return metrics, dates, y_true, y_pred


def main():
    print("=" * 60)
    print("LSTM UNIVARIANTE — Referencia historica deep")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for h in HORIZONS:
        print(f"\n--- Horizonte h={h} ---")
        metrics, dates, y_true, y_pred = train_and_evaluate(h)
        all_metrics[f"h{h}"] = metrics

        print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
        print_comparison(dates, y_true, y_pred)

    # Guardar
    out_path = RESULTS_DIR / "lstm_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetricas guardadas: {out_path}")

    print(f"\n{'=' * 60}")
    print("RESUMEN LSTM")
    print(f"{'h':>4} {'MAE':>8} {'RMSE':>8} {'MASE':>8}")
    print("-" * 30)
    for h in HORIZONS:
        m = all_metrics[f"h{h}"]
        print(f"{h:>4} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MASE']:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
