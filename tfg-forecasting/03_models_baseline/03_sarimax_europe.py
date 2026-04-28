"""
03_sarimax_europe.py — SARIMAX con ECB DFR como exogena (HICP Eurozona)

Variable exogena: DFR (Deposit Facility Rate del BCE)
Justificacion: el HICP Eurozona es exactamente la serie que el BCE intenta
controlar con el DFR. Correlacion directa y sin intermediarios.

Modelo: ordenes del auto_arima + exog=DFR

Salida:
  08_results/arimax_europe_summary.txt
  08_results/arimax_europe_metrics.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_VAL_END
from shared.metrics import mae, rmse, mase

RESULTS_DIR = ROOT / "08_results"
EXOG_COL    = "dfr"


def load_data():
    # Serie objetivo
    df = pd.read_parquet(ROOT / "data" / "processed" / "hicp_europe_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    y = df["hicp_index"]

    # Exogena DFR
    ecb = pd.read_parquet(ROOT / "data" / "processed" / "ecb_rates_monthly.parquet")
    dfr = ecb[EXOG_COL].reindex(y.index).ffill()

    train_y   = y.loc[:DATE_TRAIN_END]
    val_y     = y.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]
    train_dfr = dfr.loc[:DATE_TRAIN_END]
    val_dfr   = dfr.loc[DATE_TRAIN_END:DATE_VAL_END].iloc[1:]

    return train_y, val_y, train_dfr, val_dfr


def main():
    # Leer orden ganador de auto_arima (o usar fallback)
    metrics_path = RESULTS_DIR / "arima_europe_metrics.json"
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        order          = tuple(saved["order"])
        seasonal_order = tuple(saved["seasonal_order"])
        print(f"Orden cargado de auto_arima: SARIMA{order}x{seasonal_order}")
    else:
        order          = (2, 1, 1)   # fallback razonable
        seasonal_order = (1, 1, 1, 12)
        print(f"Orden fallback: SARIMA{order}x{seasonal_order}")

    print("=" * 60)
    print(f"SARIMAX{order}x{seasonal_order} + DFR — HICP Eurozona")
    print("=" * 60)

    train_y, val_y, train_dfr, val_dfr = load_data()
    print(f"Train: {train_y.index.min().date()} -> {train_y.index.max().date()}")
    print(f"DFR range: {train_dfr.min():.2f}% -> {train_dfr.max():.2f}%")

    mod = SARIMAX(train_y, exog=train_dfr.values.reshape(-1, 1),
                  order=order, seasonal_order=seasonal_order, trend="n")
    res = mod.fit(disp=False)
    print(res.summary())

    lb = acorr_ljungbox(res.resid, lags=[6, 12, 24], return_df=True)
    print("\nLjung-Box:")
    print(lb[["lb_stat", "lb_pvalue"]].round(4))

    fc = res.get_forecast(steps=len(val_y),
                          exog=val_dfr.values.reshape(-1, 1)).predicted_mean
    m_val = {
        "MAE":  round(mae(val_y.values, fc.values), 4),
        "RMSE": round(rmse(val_y.values, fc.values), 4),
        "MASE": round(mase(val_y.values, fc.values, train_y.values), 4),
    }
    print(f"\nMetricas validacion: {m_val}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "arimax_europe_summary.txt").write_text(
        f"SARIMAX{order}x{seasonal_order}+DFR\n"
        f"AIC={res.aic:.2f}  BIC={res.bic:.2f}\n"
        f"MAE_val={m_val['MAE']}  MASE_val={m_val['MASE']}\n"
    )
    with open(RESULTS_DIR / "arimax_europe_metrics.json", "w") as f:
        json.dump({"order": list(order), "seasonal_order": list(seasonal_order),
                   "exog": EXOG_COL, "aic": round(res.aic, 2), "metrics_val": m_val}, f, indent=2)

    print(f"\nGuardado en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
