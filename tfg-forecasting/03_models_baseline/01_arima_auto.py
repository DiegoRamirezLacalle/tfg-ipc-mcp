"""
01_arima_auto.py -- ARIMA no estacional via auto_arima (pmdarima)

Rangos de busqueda informados por el analisis ACF/PACF (notebook 04):
  - d=1 confirmado por ADF/KPSS/PP
  - p candidato: 0-3 (PACF significativo en lag 1)
  - q candidato: 0-3 (ACF significativo en lag 1)

Seleccion final por AIC. Sin componente estacional (eso va en 02_sarima).

Entrada:  data/processed/ipc_spain_index.parquet
Salida:   03_models_baseline/results/arima_summary.txt
          03_models_baseline/results/arima_metrics.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parents[1]       # tfg-forecasting/
MONOREPO = ROOT.parent                           # tfg-ipc-mcp/
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END, FORECAST_HORIZON
from shared.metrics import mae, rmse, mase

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df["indice_general"]
    y.index.freq = "MS"

    train = y.loc[:DATE_TRAIN_END]
    val = y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    test = y.loc[DATE_VAL_END:].iloc[1:]

    return y, train, val, test


def fit_arima(train):
    """auto_arima con rangos informados por ACF/PACF."""
    model = pm.auto_arima(
        train,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=1,                    # fijado por tests de estacionariedad
        seasonal=False,         # sin componente estacional (va en 02_sarima)
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )
    return model


def diagnose_residuals(model, model_name):
    """Diagnostico de residuos: Ljung-Box y estadisticas basicas."""
    resid = model.resid()
    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)

    print(f"\n--- Diagnostico de residuos ({model_name}) ---")
    print(f"  Media:     {resid.mean():.6f}")
    print(f"  Std:       {resid.std():.4f}")
    print(f"  Ljung-Box:")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "AUTOCORRELACION"
        print(f"    Lag {lag:2d}: stat={row['lb_stat']:.2f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return lb


def forecast_and_evaluate(model, train, val):
    """Prediccion sobre validacion y calculo de metricas."""
    n_val = len(val)
    fc, ci = model.predict(n_periods=n_val, return_conf_int=True, alpha=0.05)

    fc_series = pd.Series(fc, index=val.index, name="forecast")

    y_true = val.values
    y_pred = fc_series.values
    y_train = train.values

    metrics = {
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MASE": round(mase(y_true, y_pred, y_train, m=12), 4),
    }

    return fc_series, ci, metrics


def save_results(model, metrics, model_name):
    """Guarda summary e metricas."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Summary del modelo
    summary_path = RESULTS_DIR / f"{model_name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    print(f"\nSummary guardado en: {summary_path}")

    # Metricas JSON
    metrics_out = {
        "model": model_name,
        "order": list(model.order),
        "aic": round(model.aic(), 4),
        "bic": round(model.bic(), 4),
        "n_train": int(model.nobs_),
        "metrics_val": metrics,
    }
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metricas guardadas en: {metrics_path}")

    return metrics_out


def main():
    print("=" * 60)
    print("ARIMA AUTO — Modelo baseline sin estacionalidad")
    print("=" * 60)

    y, train, val, test = load_data()
    print(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    print(f"Val:   {val.index.min().date()} -> {val.index.max().date()} ({len(val)} obs)")
    print(f"Test:  {test.index.min().date()} -> {test.index.max().date()} ({len(test)} obs)")

    # Ajuste
    print("\n--- Busqueda auto_arima ---")
    model = fit_arima(train)
    print(f"\nModelo seleccionado: ARIMA{model.order}")
    print(f"AIC: {model.aic():.2f}  |  BIC: {model.bic():.2f}")
    print(model.summary())

    # Diagnostico
    diagnose_residuals(model, "arima")

    # Prediccion sobre validacion
    print(f"\n--- Prediccion sobre validacion ({len(val)} meses) ---")
    fc, ci, metrics = forecast_and_evaluate(model, train, val)

    print(f"\nMetricas sobre validacion:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Comparacion punto a punto
    print(f"\n{'Fecha':>12} {'Real':>10} {'Pred':>10} {'Error':>10}")
    print("-" * 45)
    for date, real, pred in zip(val.index, val.values, fc.values):
        print(f"{str(date.date()):>12} {real:10.3f} {pred:10.3f} {real - pred:10.3f}")

    # Guardar
    result = save_results(model, metrics, "arima")

    print(f"\n{'=' * 60}")
    print(f"RESUMEN: ARIMA{model.order}")
    print(f"  AIC={result['aic']}  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
