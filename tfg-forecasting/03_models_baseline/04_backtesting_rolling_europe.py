"""
04_backtesting_rolling_europe.py — Backtesting expanding-window HICP Eurozona

Diseno identico al pipeline de Espana y Global:
  - Ventana expandiente: entrena con todos los datos hasta el origen t
  - Ordenes FIJOS del auto_arima (script 01)
  - Horizontes: h = 1, 3, 6, 12 meses
  - Origenes: 2021-01 hasta 2024-12 (48 puntos)

Modelos:
  naive   — lag-12 estacional
  sarima  — SARIMA con orden del auto_arima
  sarimax — SARIMA + DFR (valores reales conocidos, sin leakage)

MASE scale: fijada sobre el train inicial (2002-01 a 2020-12).

Salida:
  08_results/rolling_predictions_europe.parquet
  08_results/rolling_metrics_europe.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END

RESULTS_DIR   = ROOT / "08_results"
HORIZONS      = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)
MODELS        = ["naive", "sarima", "sarimax"]


def load_orders():
    path = RESULTS_DIR / "arima_europe_metrics.json"
    if path.exists():
        saved = json.loads(path.read_text())
        order          = tuple(saved["order"])
        seasonal_order = tuple(saved["seasonal_order"])
        print(f"  Orden auto_arima: SARIMA{order}x{seasonal_order}")
    else:
        order          = (2, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        print(f"  Orden fallback:   SARIMA{order}x{seasonal_order}")
    return order, seasonal_order


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    y = df["hicp_index"]

    ecb = pd.read_parquet(ROOT / "data" / "processed" / "ecb_rates_monthly.parquet")
    dfr = ecb["dfr"].reindex(y.index).ffill()

    return y, dfr


def forecast_naive(y_train: pd.Series, h: int) -> np.ndarray:
    preds = []
    for s in range(1, h + 1):
        idx = -12 + ((s - 1) % 12)
        preds.append(float(y_train.iloc[idx]))
    return np.array(preds)


def run_rolling(y: pd.Series, dfr: pd.Series, order: tuple, seasonal_order: tuple):
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))
    print(f"  MASE scale: {mase_scale:.4f}")

    records = []
    for origin in tqdm(origins, desc="Origenes"):
        y_train   = y.loc[:origin]
        dfr_train = dfr.loc[:origin].values.reshape(-1, 1)

        try:
            res_sarima = SM_SARIMAX(
                y_train, order=order, seasonal_order=seasonal_order, trend="n"
            ).fit(disp=False)

            res_sarimax = SM_SARIMAX(
                y_train, exog=dfr_train,
                order=order, seasonal_order=seasonal_order, trend="n"
            ).fit(disp=False)
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

            y_true    = y_actual.values
            dfr_fut   = dfr.reindex(fc_dates).values.reshape(-1, 1)

            preds = {
                "naive":   forecast_naive(y_train, h),
                "sarima":  res_sarima.get_forecast(steps=h).predicted_mean.values,
                "sarimax": res_sarimax.get_forecast(
                    steps=h, exog=dfr_fut
                ).predicted_mean.values,
            }

            for model_name, y_pred in preds.items():
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

    return pd.DataFrame(records), mase_scale


def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for model in MODELS:
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


def print_results(metrics: dict) -> None:
    print(f"\n  {'Modelo':<10}", end="")
    for h in HORIZONS:
        print(f"   h={h:>2} MAE", end="")
    print()
    print(f"  {'-'*55}")
    for model in MODELS:
        print(f"  {model:<10}", end="")
        for h in HORIZONS:
            key = f"h{h}"
            if key in metrics.get(model, {}):
                v = metrics[model][key]["MAE"]
                ref = metrics["naive"][key]["MAE"]
                mark = "*" if v < ref else " "
                print(f"  {v:>7.4f}{mark}", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        print()
    print("  (* = bate al naive lag-12)")

    # DFR benefit
    print(f"\n  BENEFICIO DFR (SARIMAX vs SARIMA):")
    print(f"  {'h':>4}  {'MAE SARIMA':>12}  {'MAE SARIMAX':>13}  {'Delta%':>8}")
    print(f"  {'-'*42}")
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics.get("sarima", {}) and key in metrics.get("sarimax", {}):
            ms = metrics["sarima"][key]["MAE"]
            mx = metrics["sarimax"][key]["MAE"]
            pct = (mx - ms) / ms * 100
            mark = " <-- mejora" if pct < 0 else ""
            print(f"  {h:>4}  {ms:>12.4f}  {mx:>13.4f}  {pct:>+7.1f}%{mark}")


def main():
    print("=" * 60)
    print("BACKTESTING ROLLING — HICP Eurozona")
    print(f"  Origenes: {ORIGINS_START} - {ORIGINS_END}")
    print(f"  Horizontes: {HORIZONS}")
    print("=" * 60)

    order, seasonal_order = load_orders()
    y, dfr = load_data()
    print(f"\nHICP: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")
    print(f"DFR:  {dfr.index.min().date()} - {dfr.index.max().date()}\n")

    df_preds, mase_scale = run_rolling(y, dfr, order, seasonal_order)
    print(f"\nRegistros: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print_results(metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR / "rolling_predictions_europe.parquet", index=False)
    with open(RESULTS_DIR / "rolling_metrics_europe.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nGuardado en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
