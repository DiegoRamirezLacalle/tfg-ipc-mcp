"""
02_sarima_global.py -- ARIMA(1,1,1) fijo como modelo de referencia simple

Proposito: comparar el ARIMA(3,1,0) auto-seleccionado (script 01) contra
un modelo de menor complejidad ARIMA(1,1,1), que es el punto de partida
canonico de Box-Jenkins y sirve como referencia de parsimonia.

El nombre del fichero es sarima_global para mantener simetria con Espana,
donde el script 02 es el SARIMA estacional. Aqui no hay estacionalidad
(D=0, Fs=-0.08), por lo que la "version estacional" del global es
directamente un ARIMA mas simple.

Modelos ajustados:
  - ARIMA(1,1,1): modelo de referencia simple
  - ARIMA(3,1,0): mejor por AIC (recuperado del JSON del script 01)

Entrada:  data/processed/cpi_global_monthly.parquet
          08_results/arima_global_metrics.json
Salida:   08_results/arima111_global_summary.txt
          08_results/arima111_global_metrics.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.metrics import mae, rmse, mase

RESULTS_DIR = ROOT / "08_results"

# Orden fijo de referencia
ARIMA_REF_ORDER = (1, 1, 1)


# ── Carga ──────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df["cpi_global_rate"]
    y.index.freq = "MS"
    train = y.loc[:DATE_TRAIN_END]
    val   = y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    test  = y.loc[DATE_VAL_END:].iloc[1:]
    return y, train, val, test


# ── Ajuste con statsmodels (orden fijo) ───────────────────────────────────

def fit_arima_fixed(y_train: pd.Series, order: tuple):
    """Ajuste con statsmodels SARIMAX en modo ARIMA puro (sin seasonal)."""
    mod = SARIMAX(y_train, order=order, trend="n")
    return mod.fit(disp=False)


# ── Diagnostico ────────────────────────────────────────────────────────────

def diagnose_residuals(result, name: str):
    resid = result.resid
    lb = acorr_ljungbox(resid.dropna(), lags=[6, 12, 24], return_df=True)

    print(f"\n--- Diagnostico de residuos ({name}) ---")
    print(f"  Media:    {resid.mean():.6f}")
    print(f"  Std:      {resid.std():.4f}")
    print(f"  Min:      {resid.min():.4f}   Max: {resid.max():.4f}")
    print(f"  Ljung-Box (H0: sin autocorrelacion):")
    for lag, row in lb.iterrows():
        status = "OK" if row["lb_pvalue"] > 0.05 else "AUTOCORRELACION RESIDUAL"
        print(f"    Lag {lag:2d}: stat={row['lb_stat']:7.3f}  p={row['lb_pvalue']:.4f}  [{status}]")

    return resid, lb


# ── Prediccion rolling multi-paso ─────────────────────────────────────────

def forecast_val(result, n_steps: int, val_index):
    """Prediccion de n_steps pasos desde el final del train."""
    fc_obj = result.get_forecast(steps=n_steps)
    fc   = fc_obj.predicted_mean.values
    ci   = fc_obj.conf_int(alpha=0.05).values
    return pd.Series(fc, index=val_index), ci


# ── Metricas ───────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_train):
    return {
        "MAE":  round(float(mae(y_true, y_pred)),              4),
        "RMSE": round(float(rmse(y_true, y_pred)),             4),
        "MASE": round(float(mase(y_true, y_pred, y_train, m=12)), 4),
    }


# ── Guardar ────────────────────────────────────────────────────────────────

def save_results(result, metrics: dict, lb: pd.DataFrame, order: tuple):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"arima{''.join(str(x) for x in order)}_global"

    summary_path = RESULTS_DIR / f"{name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(result.summary()))
    print(f"Summary guardado: {summary_path}")

    lb_dict = {
        f"lag_{lag}": {"stat": round(row["lb_stat"], 4), "pvalue": round(row["lb_pvalue"], 4)}
        for lag, row in lb.iterrows()
    }
    out = {
        "model":    name,
        "order":    list(order),
        "aic":      round(float(result.aic), 4),
        "bic":      round(float(result.bic), 4),
        "n_train":  int(result.nobs),
        "residuals": {
            "mean": round(float(result.resid.mean()), 6),
            "std":  round(float(result.resid.std()),  4),
            "ljung_box": lb_dict,
        },
        "metrics_val": metrics,
    }
    metrics_path = RESULTS_DIR / f"{name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Metricas guardadas: {metrics_path}")
    return out


# ── Comparativa con ARIMA(3,1,0) ──────────────────────────────────────────

def compare_with_auto():
    """Lee el JSON del script 01 si existe."""
    path = RESULTS_DIR / "arima_global_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"ARIMA{ARIMA_REF_ORDER} GLOBAL — Modelo de referencia simple")
    print("=" * 60)

    y, train, val, test = load_data()
    print(f"Train: {train.index.min().date()} -> {train.index.max().date()} ({len(train)} obs)")
    print(f"Val:   {val.index.min().date()} -> {val.index.max().date()} ({len(val)} obs)")

    # Ajuste
    print(f"\n--- Ajustando ARIMA{ARIMA_REF_ORDER} (orden fijo, sin constante) ---")
    result = fit_arima_fixed(train, ARIMA_REF_ORDER)
    print(result.summary())

    aic = result.aic
    bic = result.bic
    print(f"\nAIC: {aic:.4f}  |  BIC: {bic:.4f}")

    # Diagnostico
    resid, lb = diagnose_residuals(result, f"ARIMA{ARIMA_REF_ORDER}")

    # Prediccion sobre validacion
    print(f"\n--- Prediccion sobre validacion ({len(val)} meses) ---")
    fc, ci = forecast_val(result, len(val), val.index)
    metrics = compute_metrics(val.values, fc.values, train.values)

    print(f"\nMetricas sobre validacion:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n{'Fecha':>12} {'Real':>10} {'Pred':>10} {'Error':>10}")
    print("-" * 45)
    for date, real, pred in zip(val.index, val.values, fc.values):
        flag = " <--" if abs(real - pred) > 1.5 else ""
        print(f"{str(date.date()):>12} {real:10.4f} {pred:10.4f} {real-pred:10.4f}{flag}")

    # Guardar
    result_dict = save_results(result, metrics, lb, ARIMA_REF_ORDER)

    # Comparativa con ARIMA(3,1,0) auto
    auto = compare_with_auto()
    if auto:
        auto_order = tuple(auto["order"])
        print(f"\n{'=' * 60}")
        print(f"COMPARATIVA ARIMA{auto_order} (auto) vs ARIMA{ARIMA_REF_ORDER} (ref)")
        print(f"{'=' * 60}")
        print(f"{'Metrica':<8} {'ARIMA'+str(auto_order):>14} {'ARIMA'+str(ARIMA_REF_ORDER):>14} {'Dif':>10}")
        print("-" * 50)
        for m_name in ["MAE", "RMSE", "MASE"]:
            a_val = auto["metrics_val"][m_name]
            r_val = metrics[m_name]
            diff  = r_val - a_val
            print(f"{m_name:<8} {a_val:14.4f} {r_val:14.4f} {diff:+10.4f}")
        print(f"\n{'AIC':<8} {auto['aic']:14.4f} {aic:14.4f} {aic - auto['aic']:+10.4f}")
        print(f"{'BIC':<8} {auto['bic']:14.4f} {bic:14.4f} {bic - auto['bic']:+10.4f}")
        print()
        # Decision
        if auto["aic"] < aic:
            margin = aic - auto["aic"]
            print(f"=> ARIMA{auto_order} es preferible por AIC ({margin:.2f} puntos mejor).")
            print(f"   Se usara ARIMA{auto_order} en el backtesting.")
        else:
            margin = aic - auto["aic"]
            print(f"=> ARIMA{ARIMA_REF_ORDER} es preferible por AIC ({-margin:.2f} puntos mejor).")
            print(f"   Se revisara el modelo para el backtesting.")
    else:
        print("\n(Ejecuta primero 01_arima_auto_global.py para ver la comparativa)")

    print(f"\n{'=' * 60}")
    print(f"RESUMEN ARIMA{ARIMA_REF_ORDER} GLOBAL")
    print(f"  AIC: {aic:.4f}  BIC: {bic:.4f}")
    print(f"  MAE val: {metrics['MAE']}  RMSE: {metrics['RMSE']}  MASE: {metrics['MASE']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
