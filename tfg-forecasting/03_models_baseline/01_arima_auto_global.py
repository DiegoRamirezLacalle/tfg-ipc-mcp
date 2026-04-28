"""
01_arima_auto_global.py -- ARIMA no estacional via auto_arima (pmdarima) — CPI Global

Serie objetivo: cpi_global_rate (tasa interanual YoY %, mediana de 186 paises)

Parametros fijados por el EDA (notebooks 02-04 _global):
  d = 1   (ADF p=0.010 marginal, PP p=0.145 no rechaza, KPSS rechaza estac. en nivel)
  D = 0   (Fs = -0.079: sin estacionalidad, mediana global cancela patrones nacionales)
  p in [0..4], q in [0..4]  (PACF corta en lag 3, ACF cola hasta lag 4 sobre diff(1))

Diferencia clave con Espana:
  - Espana:  indice en nivel -> SARIMA con D=1
  - Global:  tasa YoY ya calculada -> ARIMA sin componente estacional (D=0)

Seleccion final por AIC.

Entrada:  data/processed/cpi_global_monthly.parquet
Salida:   08_results/arima_global_summary.txt
          08_results/arima_global_metrics.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT     = Path(__file__).resolve().parents[1]   # tfg-forecasting/
MONOREPO = ROOT.parent                            # tfg-ipc-mcp/
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.metrics import mae, rmse, mase

RESULTS_DIR = ROOT / "08_results"


# ── Carga ──────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df["cpi_global_rate"]
    y.index.freq = "MS"

    train = y.loc[:DATE_TRAIN_END]
    val   = y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    test  = y.loc[DATE_VAL_END:].iloc[1:]

    return y, train, val, test


# ── Ajuste ─────────────────────────────────────────────────────────────────

def fit_arima(train: pd.Series):
    """
    auto_arima con d=1, D=0, seasonal=False.
    Rangos p, q hasta 4 segun ACF/PACF sobre diff(1).
    """
    model = pm.auto_arima(
        train,
        start_p=0, max_p=4,
        start_q=0, max_q=4,
        d=1,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )
    return model


# ── Diagnostico de residuos ────────────────────────────────────────────────

def diagnose_residuals(model, name="arima_global"):
    resid = model.resid()
    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)

    print(f"\n--- Diagnostico de residuos ({name}) ---")
    print(f"  Media:    {resid.mean():.6f}")
    print(f"  Std:      {resid.std():.4f}")
    print(f"  Min:      {resid.min():.4f}   Max: {resid.max():.4f}")
    print(f"  Ljung-Box (H0: sin autocorrelacion):")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "AUTOCORRELACION RESIDUAL"
        print(f"    Lag {lag:2d}: stat={row['lb_stat']:7.3f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return resid, lb


# ── Prediccion sobre validacion ────────────────────────────────────────────

def forecast_and_evaluate(model, train: pd.Series, val: pd.Series):
    n_val = len(val)
    fc, ci = model.predict(n_periods=n_val, return_conf_int=True, alpha=0.05)

    fc_series = pd.Series(fc, index=val.index, name="forecast")

    metrics = {
        "MAE":  round(float(mae(val.values, fc)), 4),
        "RMSE": round(float(rmse(val.values, fc)), 4),
        "MASE": round(float(mase(val.values, fc, train.values, m=12)), 4),
    }
    return fc_series, ci, metrics


# ── Guardar resultados ─────────────────────────────────────────────────────

def save_results(model, metrics: dict, resid: pd.Series, lb: pd.DataFrame):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Summary textual
    summary_path = RESULTS_DIR / "arima_global_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    print(f"\nSummary guardado: {summary_path}")

    # Metricas JSON
    lb_dict = {
        f"lag_{lag}": {"stat": round(row["lb_stat"], 4), "pvalue": round(row["lb_pvalue"], 4)}
        for lag, row in lb.iterrows()
    }
    out = {
        "model":    "arima_global",
        "order":    list(model.order),
        "aic":      round(float(model.aic()), 4),
        "bic":      round(float(model.bic()), 4),
        "n_train":  int(model.nobs_),
        "residuals": {
            "mean": round(float(resid.mean()), 6),
            "std":  round(float(resid.std()), 4),
            "ljung_box": lb_dict,
        },
        "metrics_val": metrics,
    }
    metrics_path = RESULTS_DIR / "arima_global_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Metricas guardadas: {metrics_path}")

    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ARIMA AUTO GLOBAL — Baseline CPI Global (d=1, D=0)")
    print("=" * 60)

    y, train, val, test = load_data()
    print(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    print(f"Val:   {val.index.min().date()} -> {val.index.max().date()} ({len(val)} obs)")
    print(f"Test:  {test.index.min().date()} -> {test.index.max().date()} ({len(test)} obs)")
    print(f"\nEstadisticas train:  media={train.mean():.4f}  std={train.std():.4f}  "
          f"min={train.min():.4f}  max={train.max():.4f}")

    # Ajuste
    print("\n--- Busqueda auto_arima (d=1, D=0, seasonal=False, p/q max=4) ---")
    model = fit_arima(train)

    order = model.order
    print(f"\nModelo seleccionado: ARIMA{order}")
    print(f"AIC: {model.aic():.4f}  |  BIC: {model.bic():.4f}")
    print(f"Parametros:\n{model.summary()}")

    # Diagnostico residuos
    resid, lb = diagnose_residuals(model, f"ARIMA{order}")

    # Prediccion sobre validacion
    print(f"\n--- Prediccion sobre validacion ({len(val)} meses) ---")
    fc, ci, metrics = forecast_and_evaluate(model, train, val)

    print(f"\nMetricas sobre validacion (2021-01 a 2022-06):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n{'Fecha':>12} {'Real':>10} {'Pred':>10} {'Error':>10} {'CI_low':>8} {'CI_hi':>8}")
    print("-" * 65)
    for date, real, pred, lo, hi in zip(val.index, val.values, fc.values, ci[:, 0], ci[:, 1]):
        err = real - pred
        flag = " <--" if abs(err) > 1.5 else ""
        print(f"{str(date.date()):>12} {real:10.4f} {pred:10.4f} {err:10.4f} "
              f"{lo:8.4f} {hi:8.4f}{flag}")

    # Guardar
    result = save_results(model, metrics, resid, lb)

    print(f"\n{'=' * 60}")
    print(f"RESUMEN ARIMA GLOBAL")
    print(f"  Modelo:  ARIMA{order}")
    print(f"  AIC:     {result['aic']}")
    print(f"  BIC:     {result['bic']}")
    print(f"  MAE val: {metrics['MAE']}")
    print(f"  RMSE val:{metrics['RMSE']}")
    print(f"  MASE val:{metrics['MASE']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
