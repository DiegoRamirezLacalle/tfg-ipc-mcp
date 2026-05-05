"""LSTM univariate on HICP Eurozone.

Static evaluation on validation set (2021-01 to 2022-06).
Output: 08_results/lstm_europe_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(Path(__file__).parent))
from _helpers_europe import load_nf_format_europe, evaluate_forecast, RESULTS_DIR
from shared.logger import get_logger

logger = get_logger(__name__)

HORIZONS = [1, 3, 6, 12]


def main():
    logger.info("=" * 55)
    logger.info("LSTM UNIVARIATE - HICP Eurozone")
    logger.info("=" * 55)
    train, val, df_full, y_train_vals = load_nf_format_europe()
    logger.info(f"Train: {len(train)} obs  |  Val: {len(val)} obs")

    all_metrics = {}
    for h in HORIZONS:
        logger.info(f"\n  h={h}...")
        model = LSTM(h=h, input_size=24, encoder_hidden_size=64, decoder_hidden_size=64,
                     max_steps=300, scaler_type="standard", learning_rate=1e-3,
                     random_seed=42, enable_progress_bar=False)
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=train)
        fc = nf.predict()
        y_pred = fc["LSTM"].values[:h]
        y_true = val["y"].values[:h]
        m = evaluate_forecast(y_true, y_pred, y_train_vals, "lstm")
        all_metrics[f"h{h}"] = {k: v for k, v in m.items() if k != "model"}
        logger.info(f"MAE={m['MAE']:.4f}  MASE={m['MASE']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "lstm_europe_metrics.json", "w") as f:
        json.dump({"lstm": all_metrics}, f, indent=2)
    logger.info("\nSaved: lstm_europe_metrics.json")


if __name__ == "__main__":
    main()
