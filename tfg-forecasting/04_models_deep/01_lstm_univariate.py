"""LSTM univariate as historical deep reference.

Included because LSTM is the most cited deep baseline in the time-series
forecasting literature. Uses NeuralForecast for consistency with N-BEATS and N-HiTS.

Configuration:
  - input_size = 24 (2 years of context)
  - hidden_size = 64
  - max_steps = 500
  - horizons: 1, 3, 6, 12

Input:  data/processed/ipc_spain_index.parquet
Output: 04_models_deep/results/lstm_metrics.json
"""

import json
import warnings
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM

warnings.filterwarnings("ignore")

from _helpers import (
    RESULTS_DIR, load_nf_format, evaluate_forecast, print_comparison
)
from shared.logger import get_logger

logger = get_logger(__name__)

HORIZONS = [1, 3, 6, 12]


def train_and_evaluate(horizon):
    """Train LSTM for a given horizon and evaluate on validation set."""
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

    y_pred = fc["LSTM"].values[:horizon]
    y_true = df_val["y"].values[:horizon]
    dates  = df_val["ds"].values[:horizon]

    metrics = evaluate_forecast(y_true, y_pred, y_train_vals, f"lstm_h{horizon}")
    return metrics, dates, y_true, y_pred


def main():
    logger.info("=" * 60)
    logger.info("LSTM UNIVARIATE — Historical deep reference")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for h in HORIZONS:
        logger.info(f"\n--- Horizon h={h} ---")
        metrics, dates, y_true, y_pred = train_and_evaluate(h)
        all_metrics[f"h{h}"] = metrics

        logger.info(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
        print_comparison(dates, y_true, y_pred)

    out_path = RESULTS_DIR / "lstm_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nMetrics saved: {out_path}")

    logger.info(f"\n{'=' * 60}")
    logger.info("LSTM SUMMARY")
    logger.info(f"{'h':>4} {'MAE':>8} {'RMSE':>8} {'MASE':>8}")
    logger.info("-" * 30)
    for h in HORIZONS:
        m = all_metrics[f"h{h}"]
        logger.info(f"{h:>4} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MASE']:>8.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
