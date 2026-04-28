"""
20_timegpt_C0_europe.py — TimeGPT condicion C0 (solo historico) HICP Eurozona

Rolling-origin backtesting:
  - 48 origenes: 2021-01 a 2024-12
  - Horizontes: h=1, 3, 6, 12
  - Metricas: MAE, RMSE, MASE (naive estacional lag-12)

API Nixtla: client.forecast(df, h=MAX_H, freq='MS')
Serie: hicp_europe_index.parquet (indice en nivel, base 2015=100)

Control de costes:
  --test-run    ejecuta solo 5 origenes para verificar funcionamiento
  --full        lanza las 48 origenes completas (default)

API key desde .env en la raiz del monorepo (NIXTLA_API_KEY).

Salida:
  08_results/timegpt_C0_europe_predictions.parquet
  08_results/timegpt_C0_europe_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
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
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "timegpt_C0_europe"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SERIES_ID = "HICP_EUROPE"


# -- API client -------------------------------------------------------

def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError(
            "NIXTLA_API_KEY no configurada. "
            "Edita el fichero .env en la raiz del monorepo."
        )
    from nixtla import NixtlaClient
    client = NixtlaClient(api_key=api_key)
    return client


# -- Datos -----------------------------------------------------------

def load_data() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    return df["hicp_index"]


def to_nixtla_df(y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds":        y.index,
        "y":         y.values,
    })


# -- Rolling backtesting ---------------------------------------------

def run_rolling(
    y: pd.Series,
    client,
    test_run: bool = False,
) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run:
        origins = origins[:5]
        print(f"[test-run] Probando con {len(origins)} origenes")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))
    print(f"  MASE scale: {mase_scale:.4f}")

    records = []

    for origin in tqdm(origins, desc="TimeGPT C0 europe rolling"):
        context = y.loc[:origin]
        df_input = to_nixtla_df(context)

        try:
            fc = client.forecast(
                df=df_input,
                h=MAX_H,
                freq="MS",
                time_col="ds",
                target_col="y",
                id_col="unique_id",
            )
            fc = fc.sort_values("ds").reset_index(drop=True)
            pred_values = fc["TimeGPT"].values  # shape (MAX_H,)
        except Exception as e:
            print(f"\n[!] Error en {origin.date()}: {e}")
            continue

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
            y_pred = pred_values[:h]

            for i, (date, real, pred) in enumerate(
                zip(fc_dates, y_true, y_pred), start=1
            ):
                records.append({
                    "origin":    origin,
                    "fc_date":   date,
                    "step":      i,
                    "horizon":   h,
                    "model":     MODEL_NAME,
                    "y_true":    float(real),
                    "y_pred":    float(pred),
                    "error":     float(real - pred),
                    "abs_error": float(abs(real - pred)),
                })

    return pd.DataFrame(records), mase_scale


# -- Metricas --------------------------------------------------------

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for h in HORIZONS:
        h_df = df_preds[df_preds["horizon"] == h]
        if h_df.empty:
            continue
        y_true = h_df["y_true"].values
        y_pred = h_df["y_pred"].values
        results[f"h{h}"] = {
            "MAE":     round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE":    round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MASE":    round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
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


# -- Main ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TimeGPT C0 europe rolling backtesting")
    parser.add_argument("--test-run", action="store_true",
                        help="Ejecutar solo 5 origenes para verificar coste/funcionamiento")
    args = parser.parse_args()

    print("=" * 60)
    print(f"BACKTESTING ROLLING — {MODEL_NAME}")
    print(f"Origenes: {ORIGINS_START} - {ORIGINS_END} "
          f"({'5 TEST' if args.test_run else '48 COMPLETO'})")
    print(f"Horizontes: {HORIZONS}")
    print("=" * 60)

    y = load_data()
    print(f"HICP Europa: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")

    client = get_client()
    print("[timegpt] Cliente Nixtla inicializado")

    df_preds, mase_scale = run_rolling(y, client, test_run=args.test_run)
    print(f"\nPredicciones generadas: {len(df_preds)}")

    if df_preds.empty:
        print("[!] No se generaron predicciones.")
        return

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print(f"RESULTADOS {MODEL_NAME}")
    print("=" * 60)
    print_table(metrics)

    # Comparativa vs baseline y otros foundation
    baseline_path = RESULTS_DIR / "rolling_metrics_europe.json"
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text())
        print("\n--- vs SARIMA (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            tgpt = metrics.get(key, {}).get("MAE")
            sar  = baselines.get("sarima", {}).get(key, {}).get("MAE")
            if tgpt and sar:
                delta = tgpt - sar
                print(f"  h={h}: TimeGPT={tgpt:.4f}  SARIMA={sar:.4f}  "
                      f"delta={delta:+.4f} ({delta/sar*100:+.1f}%)")

    tfm_path = RESULTS_DIR / "timesfm_C0_europe_metrics.json"
    if tfm_path.exists():
        tfm = json.loads(tfm_path.read_text()).get("timesfm_C0_europe", {})
        print("\n--- vs TimesFM C0 europe (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            tgpt = metrics.get(key, {}).get("MAE")
            tfm_mae = tfm.get(key, {}).get("MAE")
            if tgpt and tfm_mae:
                delta = tgpt - tfm_mae
                print(f"  h={h}: TimeGPT={tgpt:.4f}  TimesFM={tfm_mae:.4f}  "
                      f"delta={delta:+.4f} ({delta/tfm_mae*100:+.1f}%)")

    if args.test_run:
        print("\n[test-run] Resultados de prueba. Lanzar sin --test-run para el backtesting completo.")
        return

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
