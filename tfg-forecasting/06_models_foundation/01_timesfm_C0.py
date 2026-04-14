"""
01_timesfm_C0.py — TimesFM 2.5 condicion C0 (solo historico)

Rolling-origin backtesting con el mismo protocolo que baseline:
  - 48 origenes: 2021-01 a 2024-12
  - Horizontes: h=1, 3, 6, 12
  - Metricas: MAE, RMSE, MASE (naive estacional lag-12)

Modelo: google/timesfm-2.5-200m-pytorch (200M params, PyTorch backend)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
MAX_H = max(HORIZONS)
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END  # 2024-12-01
MODEL_NAME = "timesfm_C0"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)


# ── Datos ────────────────────────────────────────────────────────

def load_data() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_exog.parquet")
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df["indice_general"]


# ── Modelo ───────────────────────────────────────────────────────

def load_model():
    import timesfm

    print("[timesfm] Cargando modelo google/timesfm-2.5-200m-pytorch ...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
    )
    # max_context debe ser multiplo del patch_size (32)
    # 512 = 16*32, suficiente para ~42 anios mensuales
    tfm.compile(
        forecast_config=timesfm.ForecastConfig(
            max_context=512,
            max_horizon=MAX_H,
            per_core_batch_size=1,
        )
    )
    print("[timesfm] Modelo cargado y compilado (max_horizon=12)")
    return tfm


# ── Rolling backtesting ─────────────────────────────────────────

def run_rolling(y: pd.Series, model) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    # Escala MASE fija sobre train set inicial
    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="TimesFM C0 rolling"):
        context = y.loc[:origin].values.astype(np.float32)

        # Forecast h=12 de una vez, slice para cada horizonte
        point_out, _ = model.forecast(horizon=MAX_H, inputs=[context])
        full_pred = point_out[0]  # shape: (12,)

        for h in HORIZONS:
            horizon_end = origin + pd.DateOffset(months=h)
            if horizon_end > TEST_END_TS:
                continue

            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue

            y_true = y_actual.values
            y_pred = full_pred[:h]

            for i, (date, real, pred) in enumerate(
                zip(fc_dates, y_true, y_pred), start=1
            ):
                records.append({
                    "origin": origin,
                    "fc_date": date,
                    "step": i,
                    "horizon": h,
                    "model": MODEL_NAME,
                    "y_true": float(real),
                    "y_pred": float(pred),
                    "error": float(real - pred),
                    "abs_error": float(abs(real - pred)),
                })

    return pd.DataFrame(records), mase_scale


# ── Metricas ─────────────────────────────────────────────────────

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for h in HORIZONS:
        h_df = df_preds[df_preds["horizon"] == h]
        if h_df.empty:
            continue
        y_true = h_df["y_true"].values
        y_pred = h_df["y_pred"].values
        results[f"h{h}"] = {
            "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
            "n_evals": int(len(h_df["origin"].unique())),
        }
    return results


def print_table(metrics: dict) -> None:
    print(f"\n{'Horizonte':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    print("-" * 45)
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics:
            m = metrics[key]
            print(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                  f"{m['MASE']:8.4f} {m['n_evals']:5d}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"BACKTESTING ROLLING — {MODEL_NAME}")
    print(f"Origenes: {ORIGINS_START} - {ORIGINS_END}")
    print(f"Horizontes: {HORIZONS}")
    print("=" * 60)

    y = load_data()
    print(f"Datos: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")

    model = load_model()

    df_preds, mase_scale = run_rolling(y, model)
    print(f"\nPredicciones generadas: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print(f"RESULTADOS {MODEL_NAME}")
    print("=" * 60)
    print_table(metrics)

    # Guardar
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    print(f"\nPredicciones: {preds_path}")

    metrics_path = RESULTS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    print(f"Metricas:     {metrics_path}")


if __name__ == "__main__":
    main()
