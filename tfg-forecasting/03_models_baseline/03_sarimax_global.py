"""
03_sarimax_global.py -- ARIMAX con Fed Funds Rate (FEDFUNDS) como exogena

Variable exogena: FEDFUNDS (Federal Funds Rate, % anual)
Fuente: FRED (Federal Reserve Bank of St. Louis)
URL:    https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS

Justificacion economica:
  El tipo de interes de referencia de la Fed es el instrumento de politica
  monetaria dominante a nivel global. Un aumento de tipos reduce la demanda
  agregada y, con retardo, la inflacion. Analogia exacta con el DFR del BCE
  en el modelo de Espana.

Nota sobre look-ahead bias:
  Las decisiones de la Fed se publican el mismo dia del FOMC meeting.
  Pasar los valores reales del FEDFUNDS como exogena futura en la evaluacion
  estatica (validation oracle) es correcto; en el rolling backtesting se
  usaran los valores conocidos en cada origen de prediccion.

Datos guardados en:
  data/raw/fedfunds_raw.csv
  data/processed/fedfunds_monthly.parquet

Salida:
  08_results/arimax_global_summary.txt
  08_results/arimax_global_metrics.json
"""

import io
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
import requests
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.metrics import mae, rmse, mase

RESULTS_DIR  = ROOT / "08_results"
RAW_DIR      = ROOT / "data" / "raw"
PROC_DIR     = ROOT / "data" / "processed"
FEDFUNDS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
EXOG_COL     = "fedfunds"


# ── Descarga y procesamiento de FEDFUNDS ──────────────────────────────────

def download_fedfunds() -> pd.Series:
    """
    Descarga FEDFUNDS de FRED (CSV publico, sin API key).
    Devuelve serie mensual indexada por MS desde 2001-01.
    """
    raw_path = RAW_DIR / "fedfunds_raw.csv"

    print(f"  Descargando FEDFUNDS desde FRED...")
    r = requests.get(FEDFUNDS_URL, timeout=60)
    r.raise_for_status()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(r.text, encoding="utf-8")
    print(f"  Guardado raw: {raw_path}")

    df = pd.read_csv(io.StringIO(r.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "FEDFUNDS": EXOG_COL})
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthBegin(0)
    df = df.set_index("date").sort_index()
    df.index.freq = "MS"

    series = df[EXOG_COL].astype(float)
    return series


def prepare_fedfunds(series: pd.Series, date_start="2001-01-01",
                     date_end="2025-01-01") -> pd.Series:
    """
    Filtra al rango del proyecto y guarda en processed/.
    No se aplica shift: el tipo de la Fed se conoce el mismo dia
    de su anuncio (mismo supuesto que DFR en Espana).
    """
    s = series.loc[date_start:date_end]
    # Rellenar posibles gaps con forward fill (tipo no cambia entre reuniones)
    target_idx = pd.date_range(date_start, date_end, freq="MS")
    s = s.reindex(target_idx).ffill().dropna()
    s.index.freq = "MS"

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "fedfunds_monthly.parquet"
    s.to_frame().to_parquet(out_path)
    print(f"  Guardado processed: {out_path}")

    return s


def load_or_download_fedfunds() -> pd.Series:
    proc_path = PROC_DIR / "fedfunds_monthly.parquet"
    if proc_path.exists():
        print(f"  Cargando FEDFUNDS desde cache: {proc_path}")
        s = pd.read_parquet(proc_path)[EXOG_COL]
        s.index.freq = "MS"
        return s
    raw = download_fedfunds()
    return prepare_fedfunds(raw)


# ── Carga de datos ─────────────────────────────────────────────────────────

def load_data():
    # Serie objetivo
    df_y = pd.read_parquet(ROOT / "data" / "processed" / "cpi_global_monthly.parquet")
    y = df_y["cpi_global_rate"]
    y.index.freq = "MS"

    # Exogena
    fedfunds = load_or_download_fedfunds()
    print(f"  FEDFUNDS: {fedfunds.index.min().date()} -> {fedfunds.index.max().date()} "
          f"({len(fedfunds)} obs)  min={fedfunds.min():.2f}%  max={fedfunds.max():.2f}%")

    # Alinear al indice de y
    fedfunds = fedfunds.reindex(y.index).ffill()

    # Construir DataFrame combinado
    df = pd.DataFrame({"cpi_global_rate": y, EXOG_COL: fedfunds}).dropna()

    # Splits
    train_mask = df.index <= DATE_TRAIN_END
    val_mask   = (df.index > DATE_TRAIN_END) & (df.index <= DATE_VAL_END)

    y_train = df.loc[train_mask, "cpi_global_rate"]
    y_val   = df.loc[val_mask,   "cpi_global_rate"]
    X_train = df.loc[train_mask, [EXOG_COL]]
    X_val   = df.loc[val_mask,   [EXOG_COL]]

    return y_train, y_val, X_train, X_val


# ── Ajuste ─────────────────────────────────────────────────────────────────

def fit_arimax(y_train: pd.Series, X_train: pd.DataFrame):
    """
    auto_arima con exogena. Mismos rangos que script 01 (p/q max=4).
    d=1, D=0, seasonal=False — confirmado por EDA.
    """
    model = pm.auto_arima(
        y_train,
        exogenous=X_train,
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


# ── Diagnostico ────────────────────────────────────────────────────────────

def diagnose_residuals(model, name="arimax_global"):
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


# ── Prediccion ─────────────────────────────────────────────────────────────

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
        "MAE":  round(float(mae(y_val.values, fc)), 4),
        "RMSE": round(float(rmse(y_val.values, fc)), 4),
        "MASE": round(float(mase(y_val.values, fc, y_train.values, m=12)), 4),
    }
    return fc_series, ci, metrics


# ── Guardar ────────────────────────────────────────────────────────────────

def save_results(model, metrics, resid, lb):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "arimax_global_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
    print(f"\nSummary guardado: {summary_path}")

    lb_dict = {
        f"lag_{lag}": {"stat": round(row["lb_stat"], 4), "pvalue": round(row["lb_pvalue"], 4)}
        for lag, row in lb.iterrows()
    }
    out = {
        "model":         "arimax_global",
        "exog":          EXOG_COL,
        "order":         list(model.order),
        "aic":           round(float(model.aic()), 4),
        "bic":           round(float(model.bic()), 4),
        "n_train":       int(model.nobs_),
        "residuals": {
            "mean":      round(float(resid.mean()), 6),
            "std":       round(float(resid.std()),  4),
            "ljung_box": lb_dict,
        },
        "metrics_val":   metrics,
    }
    metrics_path = RESULTS_DIR / "arimax_global_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Metricas guardadas: {metrics_path}")
    return out


# ── Comparativa con scripts anteriores ────────────────────────────────────

def load_prev_metrics():
    prev = {}
    for name in ["arima_global", "arima111_global"]:
        path = RESULTS_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                prev[name] = json.load(f)
    return prev


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"ARIMAX GLOBAL — Baseline con exogena: {EXOG_COL.upper()}")
    print("=" * 60)

    print("\nCargando datos...")
    y_train, y_val, X_train, X_val = load_data()

    print(f"\nTrain: {y_train.index.min().date()} -> {y_train.index.max().date()} ({len(y_train)} obs)")
    print(f"Val:   {y_val.index.min().date()} -> {y_val.index.max().date()} ({len(y_val)} obs)")
    print(f"FEDFUNDS train: min={X_train[EXOG_COL].min():.2f}%  max={X_train[EXOG_COL].max():.2f}%")
    print(f"FEDFUNDS val:   min={X_val[EXOG_COL].min():.2f}%  max={X_val[EXOG_COL].max():.2f}%")

    # Ajuste
    print("\n--- Busqueda auto_arima con exogena FEDFUNDS ---")
    model = fit_arimax(y_train, X_train)

    order = model.order
    print(f"\nModelo seleccionado: ARIMAX{order} + {EXOG_COL.upper()}")
    print(f"AIC: {model.aic():.4f}  |  BIC: {model.bic():.4f}")
    print(model.summary())

    # Diagnostico
    resid, lb = diagnose_residuals(model, f"ARIMAX{order}")

    # Prediccion
    print(f"\n--- Prediccion sobre validacion ({len(y_val)} meses) ---")
    fc, ci, metrics = forecast_and_evaluate(model, y_train, y_val, X_val)

    print(f"\nMetricas sobre validacion:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n{'Fecha':>12} {'Real':>10} {'Pred':>10} {'Error':>10} {'FedFunds':>10}")
    print("-" * 58)
    for date, real, pred, ff in zip(y_val.index, y_val.values, fc.values, X_val[EXOG_COL].values):
        flag = " <--" if abs(real - pred) > 1.5 else ""
        print(f"{str(date.date()):>12} {real:10.4f} {pred:10.4f} {real-pred:10.4f} {ff:10.2f}{flag}")

    # Guardar
    result_dict = save_results(model, metrics, resid, lb)

    # Comparativa completa
    prev = load_prev_metrics()
    all_m = {**prev, "arimax_global": result_dict}

    if prev:
        print(f"\n{'=' * 70}")
        print("COMPARATIVA ARIMA(3,1,0) vs ARIMA(1,1,1) vs ARIMAX + FEDFUNDS")
        print(f"{'=' * 70}")
        names = [k for k in ["arima_global", "arima111_global", "arimax_global"] if k in all_m]
        labels = {"arima_global": "ARIMA(3,1,0)", "arima111_global": "ARIMA(1,1,1)",
                  "arimax_global": f"ARIMAX+{EXOG_COL.upper()}"}
        header = f"{'Metrica':<8}" + "".join(f" {labels[n]:>16}" for n in names)
        print(header)
        print("-" * (8 + 17 * len(names)))
        for m_name in ["MAE", "RMSE", "MASE"]:
            row = f"{m_name:<8}"
            for n in names:
                row += f" {all_m[n]['metrics_val'][m_name]:>16.4f}"
            print(row)
        print(f"\n{'AIC':<8}" + "".join(f" {all_m[n]['aic']:>16.4f}" for n in names))
        print(f"{'BIC':<8}" + "".join(f" {all_m[n]['bic']:>16.4f}" for n in names))
        print()

        # Beneficio de la exogena
        if "arima_global" in all_m:
            base_mae = all_m["arima_global"]["metrics_val"]["MAE"]
            ax_mae   = metrics["MAE"]
            mejora   = (base_mae - ax_mae) / base_mae * 100
            print(f"Beneficio FEDFUNDS sobre ARIMA(3,1,0): {mejora:+.1f}% en MAE val")
            if mejora > 0:
                print("=> La exogena monetaria mejora la prediccion en el periodo de validacion.")
            else:
                print("=> La exogena no mejora la prediccion en val estatica.")
                print("   Nota: FEDFUNDS estuvo cerca de 0% durante 2021 (val period).")
                print("   Su efecto sera mas visible en el rolling backtesting (2022-2024).")

    print(f"\n{'=' * 60}")
    print(f"RESUMEN ARIMAX GLOBAL")
    print(f"  Modelo:  ARIMAX{order} + {EXOG_COL.upper()}")
    print(f"  AIC: {result_dict['aic']}  BIC: {result_dict['bic']}")
    print(f"  MAE val: {metrics['MAE']}  RMSE: {metrics['RMSE']}  MASE: {metrics['MASE']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
