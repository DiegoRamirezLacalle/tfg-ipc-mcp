"""
03_sarimax_with_exog.py — SARIMAX con tipo de deposito BCE (DFR) como exogena

Variable exogena principal: dfr (Deposit Facility Rate)
Justificacion economica: es el instrumento de politica monetaria principal del
BCE desde 2014 y tiene relacion directa con la inflacion al consumo.

Nota sobre la evaluacion estatica:
  En la prediccion sobre validacion se usan los valores REALES del DFR durante
  2021-2022 (supuesto oraculo). Esto es correcto para la evaluacion estatica
  del baseline; el backtesting rolling usara los valores conocidos en cada
  origen de prediccion, que tambien son reales (el DFR es publico el mismo dia).

Entrada:  data/processed/features_exog.parquet
Salida:   03_models_baseline/results/sarimax_summary.txt
          03_models_baseline/results/sarimax_metrics.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.metrics import mae, rmse, mase

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Variable exogena principal
EXOG_COL = "dfr"


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_exog.parquet")
    df.index.freq = "MS"

    y = df["indice_general"]
    X = df[[EXOG_COL]]

    train_mask = y.index <= DATE_TRAIN_END
    val_mask   = (y.index > DATE_TRAIN_END) & (y.index <= DATE_VAL_END)

    y_train, y_val = y[train_mask], y[val_mask]
    X_train, X_val = X[train_mask], X[val_mask]

    return y, X, y_train, y_val, X_train, X_val


def fit_sarimax(y_train, X_train):
    """auto_arima con variable exogena. Mismos rangos que SARIMA para comparabilidad."""
    model = pm.auto_arima(
        y_train,
        exogenous=X_train,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=1,
        seasonal=True,
        m=12,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        D=1,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )
    return model


def diagnose_residuals(model):
    resid = model.resid()
    lb = acorr_ljungbox(resid, lags=[6, 12, 24], return_df=True)

    print("\n--- Diagnostico de residuos (sarimax) ---")
    print(f"  Media:     {resid.mean():.6f}")
    print(f"  Std:       {resid.std():.4f}")
    print("  Ljung-Box:")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "AUTOCORRELACION"
        print(f"    Lag {lag:2d}: stat={row['lb_stat']:.2f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return lb


def forecast_and_evaluate(model, y_train, y_val, X_val):
    n_val = len(y_val)
    fc, ci = model.predict(
        n_periods=n_val,
        exogenous=X_val,
        return_conf_int=True,
        alpha=0.05,
    )

    fc_series = pd.Series(fc, index=y_val.index, name="forecast")
    metrics = {
        "MAE":  round(mae(y_val.values, fc), 4),
        "RMSE": round(rmse(y_val.values, fc), 4),
        "MASE": round(mase(y_val.values, fc, y_train.values, m=12), 4),
    }
    return fc_series, ci, metrics


def load_previous_metrics():
    """Carga metricas de ARIMA y SARIMA para comparativa."""
    results = {}
    for name in ["arima", "sarima"]:
        path = RESULTS_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def save_results(model, metrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "sarimax_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    print(f"\nSummary guardado en: {summary_path}")

    out = {
        "model": "sarimax",
        "exog": EXOG_COL,
        "order": list(model.order),
        "seasonal_order": list(model.seasonal_order),
        "aic": round(model.aic(), 4),
        "bic": round(model.bic(), 4),
        "n_train": int(model.nobs_),
        "metrics_val": metrics,
    }
    metrics_path = RESULTS_DIR / "sarimax_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Metricas guardadas en: {metrics_path}")

    return out


def main():
    print("=" * 60)
    print(f"SARIMAX — Baseline con exogena: {EXOG_COL}")
    print("=" * 60)

    y, X, y_train, y_val, X_train, X_val = load_data()
    print(f"Train: {y_train.index.min().date()} -> {y_train.index.max().date()} ({len(y_train)} obs)")
    print(f"Val:   {y_val.index.min().date()} -> {y_val.index.max().date()} ({len(y_val)} obs)")
    print(f"DFR train: min={X_train[EXOG_COL].min():.2f}  max={X_train[EXOG_COL].max():.2f}")
    print(f"DFR val:   min={X_val[EXOG_COL].min():.2f}  max={X_val[EXOG_COL].max():.2f}")

    print("\n--- Busqueda auto_arima con exogena ---")
    model = fit_sarimax(y_train, X_train)

    order   = model.order
    s_order = model.seasonal_order
    print(f"\nModelo seleccionado: SARIMAX{order}x{s_order}")
    print(f"AIC: {model.aic():.2f}  |  BIC: {model.bic():.2f}")
    print(model.summary())

    diagnose_residuals(model)

    print(f"\n--- Prediccion sobre validacion ({len(y_val)} meses) ---")
    fc, ci, metrics = forecast_and_evaluate(model, y_train, y_val, X_val)

    print("\nMetricas sobre validacion:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n{'Fecha':>12} {'Real':>10} {'Pred':>10} {'Error':>10} {'DFR':>8}")
    print("-" * 50)
    for date, real, pred, dfr_val in zip(y_val.index, y_val.values, fc.values, X_val[EXOG_COL].values):
        print(f"{str(date.date()):>12} {real:10.3f} {pred:10.3f} {real - pred:10.3f} {dfr_val:8.2f}")

    result = save_results(model, metrics)

    # Comparativa con ARIMA y SARIMA
    prev = load_previous_metrics()
    if prev:
        print(f"\n{'=' * 60}")
        print("COMPARATIVA ARIMA / SARIMA / SARIMAX (validacion)")
        print(f"{'=' * 60}")
        header = f"{'Metrica':<8}"
        for name in ["arima", "sarima", "sarimax"]:
            if name in prev or name == "sarimax":
                header += f" {name.upper():>10}"
        print(header)
        print("-" * 50)

        all_metrics = {**prev, "sarimax": result}
        for m_name in ["MAE", "RMSE", "MASE"]:
            row = f"{m_name:<8}"
            for name in ["arima", "sarima", "sarimax"]:
                if name in all_metrics:
                    row += f" {all_metrics[name]['metrics_val'][m_name]:10.4f}"
            print(row)

        print("\nAIC:")
        for name in ["arima", "sarima", "sarimax"]:
            if name in all_metrics:
                print(f"  {name.upper():<8}: {all_metrics[name]['aic']:.2f}")

    print(f"\n{'=' * 60}")
    print(f"RESUMEN: SARIMAX{order}x{s_order}  exog={EXOG_COL}")
    print(f"  AIC={result['aic']}  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  MASE={metrics['MASE']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
