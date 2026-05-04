"""Rolling expanding-window backtesting — HICP Eurozone deep models.

Models: LSTM, N-BEATS, N-HiTS (NeuralForecast).

Differences from Global pipeline:
  - Series: hicp_europe_index.parquet (index level, base 2015=100)
  - NBEATS: identity stacks (Fs=0.664, significant seasonality)
  - Quarterly origins (3MS) for computational viability
  - max_steps = 300

MASE scale: fixed over initial train set 2002-01 to 2020-12.

Output:
  08_results/deep_rolling_predictions_europe.parquet
  08_results/deep_rolling_metrics_europe.json
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

RESULTS_DIR   = ROOT / "08_results"
HORIZONS      = [1, 3, 6, 12]
MODEL_NAMES   = ["lstm", "nbeats", "nhits"]
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
ORIGIN_FREQ   = "3MS"


def load_full():
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "ds", "hicp_index": "y"})
    df["unique_id"] = "HICP_EUROPE"
    return df[["unique_id", "ds", "y"]].sort_values("ds").reset_index(drop=True)


def build_models(horizon):
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
            # significant seasonality (Fs=0.664) — identity stacks
            stack_types=["identity", "identity", "identity"],
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
    origins  = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq=ORIGIN_FREQ)
    y_series = df_full.set_index("ds")["y"]

    y_train_init = y_series.loc[:DATE_TRAIN_END].values
    mase_scale   = float(np.mean(np.abs(y_train_init[12:] - y_train_init[:-12])))
    logger.info(f"  MASE scale: {mase_scale:.4f}")

    records = []
    total   = len(origins) * len(HORIZONS)

    with tqdm(total=total, desc="Rolling deep europe") as pbar:
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

                try:
                    models = build_models(h)
                    nf     = NeuralForecast(models=models, freq="MS")
                    nf.fit(df=df_train)
                    fc = nf.predict().reset_index()
                except Exception as e:
                    logger.warning(f"\n[!] {origin.date()} h={h}: {e}")
                    pbar.update(1)
                    continue

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
            yt = h_df["y_true"].values
            yp = h_df["y_pred"].values
            results[model][f"h{h}"] = {
                "MAE":     round(float(np.mean(np.abs(yt - yp))), 4),
                "RMSE":    round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
                "MASE":    round(float(np.mean(np.abs(yt - yp)) / mase_scale), 4),
                "n_evals": int(len(h_df["origin"].unique())),
            }
    return results


def main():
    logger.info("=" * 60)
    logger.info("ROLLING BACKTESTING DEEP — HICP Eurozone")
    logger.info(f"  Origins: {ORIGINS_START} - {ORIGINS_END} (every 3 months)")
    logger.info(f"  Horizons: {HORIZONS}")
    logger.info(f"  Models: {MODEL_NAMES}")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_full = load_full()
    logger.info(f"HICP Europe: {len(df_full)} obs  "
                f"({df_full['ds'].min().date()} - {df_full['ds'].max().date()})\n")

    df_preds, mase_scale = run_rolling(df_full)
    logger.info(f"\nTotal predictions: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    logger.info("\n" + "=" * 60)
    logger.info("DEEP ROLLING RESULTS — HICP Eurozone")
    logger.info("=" * 60)

    baseline_path = RESULTS_DIR / "rolling_metrics_europe.json"
    baselines = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    logger.info(f"\n  {'Model':<10}", )
    header = f"  {'Model':<10}"
    for h in HORIZONS:
        header += f"   h={h:>2} MAE"
    logger.info(header)
    logger.info(f"  {'-'*55}")

    all_models = list(baselines.keys()) + MODEL_NAMES
    for model in all_models:
        src = metrics if model in MODEL_NAMES else baselines
        row = f"  {model:<10}"
        for h in HORIZONS:
            key = f"h{h}"
            v = src.get(model, {}).get(key, {}).get("MAE")
            ref = baselines.get("naive", {}).get(key, {}).get("MAE")
            if v is not None:
                mark = "*" if ref and v < ref else " "
                row += f"  {v:>7.4f}{mark}"
            else:
                row += f"  {'N/A':>8}"
        logger.info(row)
    logger.info("  (* = beats lag-12 naive)")

    df_preds.to_parquet(RESULTS_DIR / "deep_rolling_predictions_europe.parquet", index=False)
    with open(RESULTS_DIR / "deep_rolling_metrics_europe.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nSaved to {RESULTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
