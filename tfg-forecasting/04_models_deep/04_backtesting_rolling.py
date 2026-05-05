"""Rolling expanding-window backtesting for deep models.

Same methodology as 03_models_baseline/04_backtesting_rolling.py:
  - Expanding window, fixed orders, monthly origins
  - Models: LSTM, N-BEATS, N-HiTS
  - Horizons: h = 1, 3, 6, 12

Performance note: training deep models is ~50x slower than ARIMA.
To keep rolling viable:
  - Quarterly origins instead of monthly
  - max_steps reduced to 300 (sufficient convergence for CPI)

Output:
  results/deep_rolling_predictions.parquet
  results/deep_rolling_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NBEATS, NHITS
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END
from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

HORIZONS = [1, 3, 6, 12]
MODEL_NAMES = ["lstm", "nbeats", "nhits"]
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

# Quarterly origins for computational viability
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
ORIGIN_FREQ   = "3MS"


def load_full():
    """Load full series in NeuralForecast long format."""
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df[["indice_general"]].copy()
    y.index.name = "ds"
    y = y.reset_index()
    y.columns = ["ds", "y"]
    y["unique_id"] = "IPC_ESP"
    y = y[["unique_id", "ds", "y"]]
    return y


def build_models(horizon):
    """Instantiate the three deep models for a given horizon."""
    if horizon < 4:
        nbeats_stacks = ["identity", "identity", "identity"]
    else:
        nbeats_stacks = ["identity", "trend", "seasonality"]

    return [
        LSTM(
            h=horizon,
            input_size=24,
            encoder_hidden_size=64,
            decoder_hidden_size=64,
            max_steps=300,
            scaler_type="standard",
            learning_rate=1e-3,
            random_seed=42,
            enable_progress_bar=False,
        ),
        NBEATS(
            h=horizon,
            input_size=24,
            stack_types=nbeats_stacks,
            max_steps=300,
            scaler_type="standard",
            learning_rate=1e-3,
            random_seed=42,
            enable_progress_bar=False,
        ),
        NHITS(
            h=horizon,
            input_size=24,
            max_steps=300,
            scaler_type="standard",
            learning_rate=1e-3,
            n_pool_kernel_size=[2, 2, 1],
            random_seed=42,
            enable_progress_bar=False,
        ),
    ]


def run_rolling(df_full):
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq=ORIGIN_FREQ)
    y_series = df_full.set_index("ds")["y"]

    # MASE scale (fixed, over initial train set)
    y_train_init = y_series.loc[:DATE_TRAIN_END].values
    mase_scale = float(np.mean(np.abs(y_train_init[12:] - y_train_init[:-12])))

    records = []
    total = len(origins) * len(HORIZONS)

    with tqdm(total=total, desc="Rolling deep") as pbar:
        for origin in origins:
            df_train = df_full[df_full["ds"] <= origin].copy()

            for h in HORIZONS:
                pbar.set_postfix(origin=str(origin.date()), h=h)

                horizon_end = origin + pd.DateOffset(months=h)
                if horizon_end > TEST_END_TS:
                    pbar.update(1)
                    continue

                fc_dates = pd.date_range(
                    start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
                )
                y_actual = y_series.reindex(fc_dates)
                if y_actual.isna().any():
                    pbar.update(1)
                    continue

                y_true = y_actual.values

                # Build and fit models for this horizon
                models = build_models(h)
                nf = NeuralForecast(models=models, freq="MS")
                nf.fit(df=df_train)
                fc = nf.predict().reset_index()

                for model_name, col_name in zip(MODEL_NAMES, ["LSTM", "NBEATS", "NHITS"]):
                    y_pred = fc[col_name].values[:h]

                    for i, (date, real, pred) in enumerate(
                        zip(fc_dates, y_true, y_pred), start=1
                    ):
                        records.append({
                            "origin":    origin,
                            "fc_date":   date,
                            "step":      i,
                            "horizon":   h,
                            "model":     model_name,
                            "y_true":    real,
                            "y_pred":    pred,
                            "error":     real - pred,
                            "abs_error": abs(real - pred),
                        })

                pbar.update(1)

    return pd.DataFrame(records), mase_scale


def compute_metrics(df_preds, mase_scale):
    results = {}
    for model in MODEL_NAMES:
        results[model] = {}
        m_df = df_preds[df_preds["model"] == model]
        for h in HORIZONS:
            h_df = m_df[m_df["horizon"] == h]
            if h_df.empty:
                continue
            y_true = h_df["y_true"].values
            y_pred = h_df["y_pred"].values
            results[model][f"h{h}"] = {
                "MAE":    round(float(np.mean(np.abs(y_true - y_pred))), 4),
                "RMSE":   round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
                "MASE":   round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
                "n_evals": int(len(h_df["origin"].unique())),
            }
    return results


def main():
    logger.info("=" * 60)
    logger.info("ROLLING BACKTESTING - Deep models (LSTM / N-BEATS / N-HiTS)")
    logger.info(f"Origins: {ORIGINS_START} - {ORIGINS_END} (every 3 months)")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_full = load_full()
    logger.info(f"Data: {len(df_full)} obs")

    df_preds, mase_scale = run_rolling(df_full)
    logger.info(f"\nTotal predictions: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info("DEEP ROLLING RESULTS")
    logger.info("=" * 60)
    for h in HORIZONS:
        key = f"h{h}"
        logger.info(f"\n  h={h}:")
        logger.info(f"  {'Model':<10} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
        logger.info(f"  {'-'*40}")
        for model in MODEL_NAMES:
            if key in metrics.get(model, {}):
                m = metrics[model][key]
                logger.info(f"  {model:<10} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                             f"{m['MASE']:>8.4f} {m['n_evals']:>5d}")

    preds_path = RESULTS_DIR / "deep_rolling_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    logger.info(f"\nPredictions: {preds_path}")

    metrics_path = RESULTS_DIR / "deep_rolling_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
