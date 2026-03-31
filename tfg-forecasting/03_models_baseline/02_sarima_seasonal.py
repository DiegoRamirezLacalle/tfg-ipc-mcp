"""
02_sarima_seasonal.py -- SARIMA con componente estacional via auto_arima

Rangos informados por el analisis EDA completo:
  - d=1, D=1 confirmados por tests de estacionariedad
  - m=12 (frecuencia mensual)
  - p candidato: 0-3, q candidato: 0-3 (ACF/PACF regular)
  - P candidato: 0-2, Q candidato: 0-2 (ACF/PACF estacional)
  - Modelo esperado: SARIMA(1,1,1)(0-1,1,1)_12

Compara resultados con ARIMA (01_arima_auto.py) para cuantificar
el beneficio de incorporar estacionalidad.

Entrada:  data/processed/ipc_spain_index.parquet
Salida:   03_models_baseline/results/sarima_summary.txt
          03_models_baseline/results/sarima_metrics.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
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


def fit_sarima(train):
    """auto_arima estacional con rangos informados por ACF/PACF."""
    model = pm.auto_arima(
        train,
        # Ordenes regulares
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=1,                        # fijado por tests
        # Ordenes estacionales
        seasonal=True,
        m=12,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        D=1,                        # fijado por tests + Fs
        # Criterio y busqueda
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


def compare_with_arima():
    """Carga metricas de ARIMA para comparar."""
    arima_path = RESULTS_DIR / "arima_metrics.json"
    if arima_path.exists():
        with open(arima_path, "r") as f:
            return json.load(f)
    return None


def save_results(model, metrics, model_name):
    """Guarda summary y metricas."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / f"{model_name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    print(f"\nSummary guardado en: {summary_path}")

    seasonal_order = list(model.seasonal_order) if hasattr(model, "seasonal_order") else None

    metrics_out = {
        "model": model_name,
        "order": list(model.order),
        "seasonal_order": seasonal_order,
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
    print("SARIMA — Modelo baseline con estacionalidad")
    print("=" * 60)

    y, train, val, test = load_data()
    print(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    print(f"Val:   {val.index.min().date()} -> {val.index.max().date()} ({len(val)} obs)")
    print(f"Test:  {test.index.min().date()} -> {test.index.max().date()} ({len(test)} obs)")

    # Ajuste
    print("\n--- Busqueda auto_arima estacional ---")
    model = fit_sarima(train)

    order = model.order
    s_order = model.seasonal_order
    print(f"\nModelo seleccionado: SARIMA{order}x{s_order}")
    print(f"AIC: {model.aic():.2f}  |  BIC: {model.bic():.2f}")
    print(model.summary())

    # Diagnostico
    diagnose_residuals(model, "sarima")

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
    result = save_results(model, metrics, "sarima")

    # Comparacion con ARIMA
    arima = compare_with_arima()
    if arima:
        print(f"\n{'=' * 60}")
        print("COMPARATIVA ARIMA vs SARIMA (validacion)")
        print(f"{'=' * 60}")
        print(f"{'Metrica':<8} {'ARIMA':>12} {'SARIMA':>12} {'Mejora':>12}")
        print("-" * 48)
        for m_name in ["MAE", "RMSE", "MASE"]:
            a_val = arima["metrics_val"][m_name]
            s_val = metrics[m_name]
            mejora = ((a_val - s_val) / a_val) * 100
            print(f"{m_name:<8} {a_val:12.4f} {s_val:12.4f} {mejora:+11.1f}%")
        print(f"\n{'AIC':<8} {arima['aic']:12.2f} {result['aic']:12.2f}")
        print(f"{'BIC':<8} {arima.get('bic', 'N/A'):>12} {result['bic']:12.2f}")
    else:
        print("\n(Ejecuta primero 01_arima_auto.py para ver la comparativa)")

    print(f"\n{'=' * 60}")
    print(f"RESUMEN: SARIMA{order}x{s_order}")
    print(f"  AIC={result['aic']}  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
