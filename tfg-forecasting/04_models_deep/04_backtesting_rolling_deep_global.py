"""
04_backtesting_rolling_deep_global.py — Backtesting expanding-window deep CPI Global

Analogo a 04_backtesting_rolling.py (Espana) sobre la serie global.
Modelos: LSTM, N-BEATS, N-HiTS.

Diferencias clave respecto a Espana:
  - Serie: cpi_global_monthly.parquet (cpi_global_rate)
  - NBEATS stacks = identity x3 siempre (Fs=-0.08, sin estacionalidad)
  - Origenes trimestrales (3MS) para viabilidad computacional
  - max_steps = 300

Salida:
  08_results/deep_rolling_predictions_global.parquet
  08_results/deep_rolling_metrics_global.json
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

RESULTS_DIR = ROOT / "08_results"

HORIZONS    = [1, 3, 6, 12]
MODEL_NAMES = ["lstm", "nbeats", "nhits"]
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
ORIGIN_FREQ   = "3MS"


def load_full():
    df = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df[["cpi_global_rate"]].copy()
    y.index.name = "ds"
    y = y.reset_index()
    y.columns = ["ds", "y"]
    y["ds"] = pd.to_datetime(y["ds"])
    y["unique_id"] = "CPI_GLOBAL"
    return y[["unique_id", "ds", "y"]]


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
    print(f"  MASE scale (naive lag-12, train): {mase_scale:.4f} pp")

    records = []
    total   = len(origins) * len(HORIZONS)

    with tqdm(total=total, desc="Rolling deep global") as pbar:
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

                models = build_models(h)
                nf     = NeuralForecast(models=models, freq="MS")
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
    print("=" * 60)
    print("BACKTESTING ROLLING — Deep models CPI Global")
    print(f"  Origenes: {ORIGINS_START} - {ORIGINS_END} (cada 3 meses)")
    print(f"  Horizontes: {HORIZONS}")
    print(f"  Modelos: {MODEL_NAMES}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_full = load_full()
    print(f"CPI Global: {len(df_full)} obs  "
          f"({df_full['ds'].min().date()} - {df_full['ds'].max().date()})\n")

    df_preds, mase_scale = run_rolling(df_full)
    print(f"\nTotal predicciones: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print("RESULTADOS ROLLING DEEP — CPI Global")
    print("=" * 60)
    for h in HORIZONS:
        key = f"h{h}"
        print(f"\n  h={h}:")
        print(f"  {'Modelo':<10} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
        print(f"  {'-'*42}")
        for model in MODEL_NAMES:
            if key in metrics.get(model, {}):
                m = metrics[model][key]
                print(f"  {model:<10} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                      f"{m['MASE']:>8.4f} {m['n_evals']:>5d}")

    preds_path = RESULTS_DIR / "deep_rolling_predictions_global.parquet"
    df_preds.to_parquet(preds_path, index=False)
    print(f"\nPredicciones: {preds_path}")

    metrics_path = RESULTS_DIR / "deep_rolling_metrics_global.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metricas:     {metrics_path}")

    print("\n" + "=" * 60)
    print("BACKTESTING DEEP COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
