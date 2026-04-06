"""
04_backtesting_rolling.py — Backtesting expanding-window

Evaluacion rigurosa de los modelos baseline mediante rolling origin.

Diseno:
  - Ventana expandiente: en cada origen t se entrena con todos los datos hasta t
  - Ordenes fijos determinados por auto_arima en 01/02/03 (sin re-seleccion en
    cada paso, evita look-ahead bias y reduce tiempo de computo)
  - Modelos: ARIMA(1,1,2), SARIMA(0,1,1)(0,1,1)12, SARIMAX con dfr, naive estacional
  - Horizontes: h = 1, 3, 6, 12 meses
  - Origenes: 2021-01 hasta 2024-12 (48 puntos; para h=12 el ultimo util es 2023-12)

Nota SARIMAX: el DFR es publico en tiempo real (decisiones del BCE se publican
el mismo dia), por lo que pasar los valores reales del DFR como exogena futura
no introduce look-ahead bias.

Salida:
  results/rolling_predictions.parquet  — predicciones tidy (origin, horizon, model)
  results/rolling_metrics.json         — MAE/RMSE/MASE por modelo x horizonte
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
from shared.metrics import mae, rmse, mase

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Ordenes fijados por auto_arima ─────────────────────────────────────────
ARIMA_ORDER   = (1, 1, 2)
SARIMA_ORDER  = (0, 1, 1)
SARIMA_SORDER = (0, 1, 1, 12)
EXOG_COL      = "dfr"

HORIZONS       = [1, 3, 6, 12]
ORIGINS_START  = "2021-01-01"
ORIGINS_END    = DATE_TEST_END     # "2024-12-01"
MODELS         = ["naive", "arima", "sarima", "sarimax"]
TEST_END_TS    = pd.Timestamp(DATE_TEST_END)


# ── Carga de datos ─────────────────────────────────────────────────────────

def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_exog.parquet")
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    return df


# ── Ajuste statsmodels (ordenes fijos, rapido) ─────────────────────────────

def fit_arima(y_train: pd.Series):
    mod = SM_SARIMAX(y_train, order=ARIMA_ORDER, trend="c")
    return mod.fit(disp=False)


def fit_sarima(y_train: pd.Series):
    mod = SM_SARIMAX(y_train, order=SARIMA_ORDER,
                     seasonal_order=SARIMA_SORDER, trend="c")
    return mod.fit(disp=False)


def fit_sarimax(y_train: pd.Series, x_train: pd.DataFrame):
    mod = SM_SARIMAX(y_train, exog=x_train, order=SARIMA_ORDER,
                     seasonal_order=SARIMA_SORDER, trend="c")
    return mod.fit(disp=False)


def forecast_fixed(result, h: int, x_future=None) -> np.ndarray:
    """Prediccion h pasos desde el final del train."""
    fc = result.forecast(steps=h, exog=x_future)
    return fc.values


def forecast_naive(y_train: pd.Series, h: int) -> np.ndarray:
    """Naive estacional: y[t+s] = y[t+s-12] para s=1..h."""
    preds = []
    for s in range(1, h + 1):
        # s=1 -> t-11, ..., s=12 -> t (estacionalidad mensual de 12)
        idx = -12 + ((s - 1) % 12)
        preds.append(float(y_train.iloc[idx]))
    return np.array(preds)


# ── Bucle principal ────────────────────────────────────────────────────────

def run_rolling(df: pd.DataFrame) -> pd.DataFrame:
    y = df["indice_general"]
    X = df[[EXOG_COL]]

    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    # Escala MASE: naive estacional sobre el train set inicial (fija)
    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="Origenes"):
        # Datos disponibles hasta el origen (inclusive)
        y_train = y.loc[:origin]
        x_train = X.loc[:origin]

        # Ajuste de los tres modelos parametricos
        try:
            res_arima   = fit_arima(y_train)
            res_sarima  = fit_sarima(y_train)
            res_sarimax = fit_sarimax(y_train, x_train)
        except Exception as e:
            print(f"\n[!] Error en {origin.date()}: {e}")
            continue

        for h in HORIZONS:
            # Horizonte completo debe quedar dentro del test end.
            horizon_end = origin + pd.DateOffset(months=h)
            if horizon_end > TEST_END_TS:
                continue

            # Fechas del horizonte
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )

            # Actuals disponibles para este horizonte
            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue   # actuals aun no disponibles

            y_true = y_actual.values
            x_future = X.reindex(fc_dates)  # DFR real (sin look-ahead bias)

            # Predicciones de cada modelo
            preds = {
                "naive":   forecast_naive(y_train, h),
                "arima":   forecast_fixed(res_arima, h),
                "sarima":  forecast_fixed(res_sarima, h),
                "sarimax": forecast_fixed(res_sarimax, h, x_future=x_future),
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


# ── Metricas agregadas ─────────────────────────────────────────────────────

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    """MAE, RMSE, MASE por modelo x horizonte."""
    results = {}
    for model in MODELS:
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


# ── Impresion de resultados ────────────────────────────────────────────────

def print_table(metrics: dict) -> None:
    for h in HORIZONS:
        key = f"h{h}"
        print(f"\n--- Horizonte h={h} ---")
        print(f"{'Modelo':<10} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
        print("-" * 42)
        for model in MODELS:
            if key in metrics.get(model, {}):
                m = metrics[model][key]
                print(f"{model:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                      f"{m['MASE']:8.4f} {m['n_evals']:5d}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BACKTESTING ROLLING — Modelos baseline")
    print(f"Origenes: {ORIGINS_START} - {ORIGINS_END}")
    print(f"Horizontes: {HORIZONS}")
    print(f"Modelos: {MODELS}")
    print("=" * 60)

    df = load_data()
    print(f"Datos cargados: {df.index.min().date()} - {df.index.max().date()} ({len(df)} obs)")

    df_preds, mase_scale = run_rolling(df)

    print(f"\nTotal predicciones generadas: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print("RESULTADOS ROLLING BACKTESTING")
    print("=" * 60)
    print_table(metrics)

    # Comparativa MASE vs naive (benchmark relativo)
    print("\n--- MASE relativo al naive estacional (naive=1.00) ---")
    print(f"{'Modelo':<10}", end="")
    for h in HORIZONS:
        print(f"  h={h:>2}", end="")
    print()
    print("-" * 42)
    for model in ["arima", "sarima", "sarimax"]:
        print(f"{model:<10}", end="")
        for h in HORIZONS:
            key = f"h{h}"
            if key in metrics.get(model, {}) and key in metrics.get("naive", {}):
                ratio = metrics[model][key]["MASE"] / metrics["naive"][key]["MASE"]
                print(f"  {ratio:>5.3f}", end="")
            else:
                print(f"  {'N/A':>5}", end="")
        print()

    # Guardar
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = RESULTS_DIR / "rolling_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    print(f"\nPredicciones guardadas: {preds_path}")

    metrics_path = RESULTS_DIR / "rolling_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metricas guardadas:     {metrics_path}")


if __name__ == "__main__":
    main()
