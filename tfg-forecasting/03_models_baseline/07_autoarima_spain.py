"""
07_autoarima_spain.py — AutoSARIMA rolling backtesting (IPC Espana)

Diferencia clave frente a SARIMA fijo (04_backtesting_rolling.py):
  - En cada origen rolling, pmdarima.auto_arima re-selecciona los ordenes
    (p,d,q)(P,D,Q) optimos via criterio AIC + stepwise.
  - El SARIMA fijo usa ordenes determinados UNA sola vez en el train inicial.

Diseno:
  - Ventana expandiente: en cada origen t se entrena con todos los datos hasta t
  - auto_arima estacional (m=12) re-ajustado en cada origen
  - Horizontes: h = 1, 3, 6, 12 meses
  - Origenes: 2021-01 hasta 2024-12 (48 puntos)
  - Sin exogena (pure AutoSARIMA, comparable al SARIMA baseline)

Salida:
  08_results/autoarima_spain_predictions.parquet
  08_results/autoarima_spain_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END

RESULTS_DIR   = ROOT / "08_results"
HORIZONS      = [1, 3, 6, 12]
ORIGINS_START = "2021-01-01"
ORIGINS_END   = DATE_TEST_END
TEST_END_TS   = pd.Timestamp(DATE_TEST_END)


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_exog.parquet")
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df["indice_general"]


def forecast_naive(y_train: pd.Series, h: int) -> np.ndarray:
    preds = []
    for s in range(1, h + 1):
        idx = -12 + ((s - 1) % 12)
        preds.append(float(y_train.iloc[idx]))
    return np.array(preds)


def run_rolling(y: pd.Series):
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))
    print(f"  MASE scale: {mase_scale:.4f} pp")

    records = []
    orders_log = []

    for origin in tqdm(origins, desc="Origenes AutoARIMA Spain"):
        y_train = y.loc[:origin]

        try:
            model = auto_arima(
                y_train,
                seasonal=True,
                m=12,
                stepwise=True,
                information_criterion="aic",
                max_p=3, max_q=3,
                max_P=2, max_Q=2,
                max_d=2, max_D=1,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
            )
            orders_log.append({
                "origin": str(origin.date()),
                "order": list(model.order),
                "seasonal_order": list(model.seasonal_order),
            })
        except Exception as e:
            print(f"\n[!] Error auto_arima en {origin.date()}: {e}")
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

            try:
                y_pred_auto = model.predict(n_periods=h)
            except Exception:
                continue

            y_pred_naive = forecast_naive(y_train, h)

            for model_name, y_pred in [("auto_arima", y_pred_auto), ("naive", y_pred_naive)]:
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

    return pd.DataFrame(records), mase_scale, orders_log


def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for model in ["auto_arima", "naive"]:
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
    print("AutoARIMA ROLLING — IPC Espana")
    print(f"  Origenes: {ORIGINS_START} - {ORIGINS_END}")
    print(f"  Horizontes: {HORIZONS}")
    print(f"  Metodo: pmdarima.auto_arima re-ajustado en cada origen")
    print("=" * 60)

    y = load_data()
    print(f"\nIPC Espana: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)\n")

    df_preds, mase_scale, orders_log = run_rolling(y)
    print(f"\nTotal registros: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print("RESULTADOS AutoARIMA Spain")
    print("=" * 60)
    ref = metrics.get("naive", {})
    print(f"\n  {'h':>4}  {'MAE':>8}  {'RMSE':>8}  {'MASE':>8}  {'vs naive':>9}")
    print(f"  {'-'*48}")
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics.get("auto_arima", {}):
            m = metrics["auto_arima"][key]
            n_mae = ref.get(key, {}).get("MAE", float("nan"))
            ratio = m["MAE"] / n_mae if n_mae else float("nan")
            mark = " *" if ratio < 1.0 else ""
            print(f"  {h:>4}  {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  "
                  f"{m['MASE']:>8.4f}  {ratio:>8.3f}x{mark}")

    print("\n  (* = bate al naive estacional lag-12)")

    # Muestra ordenes seleccionados
    print("\n  Muestra de ordenes auto_arima (cada 12 origenes):")
    for entry in orders_log[::12]:
        print(f"    {entry['origin']}: SARIMA{tuple(entry['order'])}x{tuple(entry['seasonal_order'])}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(RESULTS_DIR / "autoarima_spain_predictions.parquet", index=False)
    with open(RESULTS_DIR / "autoarima_spain_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_DIR / "autoarima_spain_orders.json", "w") as f:
        json.dump(orders_log, f, indent=2)

    print(f"\nGuardado en {RESULTS_DIR}")
    print("=" * 60)
    print("COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
