"""
07_timegpt_C1_energy.py — TimeGPT C1 con variables energeticas + MCP seleccionadas

Covariables (8 total, subconjunto optimo para evitar overfitting):
  Energia (datos reales desde 2002, sin NaN):
    brent_ma3       # corr 0.715 con IPC(t+1) en 2015+
    ttf_ma3         # corr 0.541
    brent_ret       # captura shocks energeticos rapidos

  MCP (NaN pre-2015, datos reales post-2015):
    bce_shock_score, bce_tone_numeric, bce_cumstance,
    gdelt_tone_ma6, signal_available

Contexto Fix v2: IPC completo desde 2002 + separacion energia/MCP para NaN.

Control de costes:
  --test-run    ejecuta solo 5 origenes
  --full        lanza las 48 origenes completas (default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END, DATE_TEST_END

RESULTS_DIR = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
MAX_H = max(HORIZONS)
ORIGINS_START = "2021-01-01"
ORIGINS_END = DATE_TEST_END
MODEL_NAME = "timegpt_C1_energy"
TEST_END_TS = pd.Timestamp(DATE_TEST_END)
SERIES_ID = "ipc_spain"
SIGNAL_START = "2015-01-01"

# Energia: datos reales desde 2002 (Brent via proxy WTI), sin NaN pre-2015
ENERGY_COLS = ["brent_ma3", "brent_ret", "ttf_ma3"]

# MCP: NaN pre-2015 para evitar regimen de ceros espurio
MCP_COLS = [
    "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
    "gdelt_tone_ma6", "signal_available",
]

EXOG_COLS = ENERGY_COLS + MCP_COLS


# ── API client ───────────────────────────────────────────────────

def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError(
            "NIXTLA_API_KEY no configurada. "
            "Edita el fichero .env en la raiz del monorepo."
        )
    from nixtla import NixtlaClient
    client = NixtlaClient(api_key=api_key)
    return client


# ── Datos ────────────────────────────────────────────────────────

def load_data() -> tuple[pd.Series, pd.DataFrame]:
    # Serie objetivo
    ipc_df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = ipc_df["indice_general"]
    y.index = pd.to_datetime(y.index)
    y.index.freq = "MS"

    # Covariables C1 (ahora con 31 cols incluyendo energia)
    c1 = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    c1["date"] = pd.to_datetime(c1["date"])
    c1 = c1.set_index("date")
    c1.index.freq = "MS"
    for col in EXOG_COLS:
        if col in c1.columns:
            c1[col] = c1[col].fillna(0.0)

    return y, c1


def build_nixtla_df(
    y: pd.Series,
    exog: pd.DataFrame,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara df (historico con exogenas) y futuros_df (solo exogenas del horizonte).

    Tratamiento diferenciado:
      - IPC: contexto COMPLETO desde 2002 (tendencia largo plazo)
      - ENERGY_COLS: datos reales desde 2002 (Brent tiene proxy desde 2001)
      - MCP_COLS: NaN pre-2015, datos reales post-2015
    """
    # IPC completo desde 2002 (igual que C0)
    context_y = y.loc[:origin]

    # Historico: merge serie + exogenas
    hist_df = pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": context_y.index,
        "y": context_y.values,
    })

    # Energia: datos reales desde 2002, sin NaN
    for col in ENERGY_COLS:
        if col in exog.columns:
            col_vals = exog.loc[:origin, col].reindex(context_y.index)
            hist_df[col] = col_vals.values
        else:
            hist_df[col] = 0.0

    # MCP: NaN pre-2015 para evitar regimen de ceros espurio
    for col in MCP_COLS:
        if col in exog.columns:
            col_vals = exog.loc[:origin, col].reindex(context_y.index)
            col_vals.loc[col_vals.index < SIGNAL_START] = np.nan
            hist_df[col] = col_vals.values
        else:
            hist_df[col] = np.nan

    # Futuro: forward-fill del ultimo valor conocido en el origen
    last_row = exog.loc[:origin, EXOG_COLS].iloc[-1]
    fc_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
    )

    future_df = pd.DataFrame({"unique_id": SERIES_ID, "ds": fc_dates})
    for col in EXOG_COLS:
        if col == "signal_available":
            future_df[col] = 1.0
        else:
            future_df[col] = float(last_row[col])

    return hist_df, future_df


# ── Rolling backtesting ─────────────────────────────────────────

def run_rolling(
    y: pd.Series,
    exog: pd.DataFrame,
    client,
    test_run: bool = False,
) -> tuple[pd.DataFrame, float]:
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")
    if test_run:
        origins = origins[:5]
        print(f"[test-run] Probando con {len(origins)} origenes")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="TimeGPT C1 energy rolling"):
        hist_df, future_df = build_nixtla_df(y, exog, origin)

        try:
            fc = client.forecast(
                df=hist_df,
                X_df=future_df,
                h=MAX_H,
                freq="MS",
                time_col="ds",
                target_col="y",
                id_col="unique_id",
            )
            fc = fc.sort_values("ds").reset_index(drop=True)
            pred_values = fc["TimeGPT"].values
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

            y_true = y_actual.values
            y_pred = pred_values[:h]

            for i, (date, real, pred) in enumerate(
                zip(fc_dates, y_true, y_pred), start=1
            ):
                records.append({
                    "origin": origin,
                    "fc_date": date,
                    "step": i,
                    "horizon": h,
                    "model": MODEL_NAME,
                    "y_true": float(real),
                    "y_pred": float(pred),
                    "error": float(real - pred),
                    "abs_error": float(abs(real - pred)),
                })

    return pd.DataFrame(records), mase_scale


# ── Metricas ─────────────────────────────────────────────────────

def compute_metrics(df_preds: pd.DataFrame, mase_scale: float) -> dict:
    results = {}
    for h in HORIZONS:
        h_df = df_preds[df_preds["horizon"] == h]
        if h_df.empty:
            continue
        y_true = h_df["y_true"].values
        y_pred = h_df["y_pred"].values
        results[f"h{h}"] = {
            "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MASE": round(float(np.mean(np.abs(y_true - y_pred)) / mase_scale), 4),
            "n_evals": int(len(h_df["origin"].unique())),
        }
    return results


def print_table(metrics: dict) -> None:
    print(f"\n{'Horizonte':<12} {'MAE':>8} {'RMSE':>8} {'MASE':>8} {'N':>5}")
    print("-" * 45)
    for h in HORIZONS:
        key = f"h{h}"
        if key in metrics:
            m = metrics[key]
            print(f"h={h:<10} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
                  f"{m['MASE']:8.4f} {m['n_evals']:5d}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimeGPT C1 energy rolling backtesting")
    parser.add_argument("--test-run", action="store_true",
                        help="Ejecutar solo 5 origenes para verificar coste/funcionamiento")
    args = parser.parse_args()

    print("=" * 60)
    print(f"BACKTESTING ROLLING — {MODEL_NAME}")
    print(f"Origenes: {ORIGINS_START} - {ORIGINS_END} ({'5 TEST' if args.test_run else '48 COMPLETO'})")
    print(f"Horizontes: {HORIZONS}")
    print(f"Energy cols: {ENERGY_COLS}")
    print(f"MCP cols: {MCP_COLS}")
    print("=" * 60)

    y, exog = load_data()
    print(f"Serie IPC: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")
    print(f"Exogenas disponibles: {[c for c in EXOG_COLS if c in exog.columns]}")

    client = get_client()
    print("[timegpt] Cliente Nixtla inicializado")

    df_preds, mase_scale = run_rolling(y, exog, client, test_run=args.test_run)
    print(f"\nPredicciones generadas: {len(df_preds)}")

    if df_preds.empty:
        print("[!] No se generaron predicciones.")
        return

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print(f"RESULTADOS {MODEL_NAME}")
    print("=" * 60)
    print_table(metrics)

    # Comparativa C0 vs C1_energy
    c0_path = RESULTS_DIR / "timegpt_C0_metrics.json"
    if c0_path.exists():
        with open(c0_path) as f:
            c0_m = json.load(f).get("timegpt_C0", {})
        print("\n--- C0 vs C1_energy (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            c0_mae = c0_m.get(key, {}).get("MAE")
            c1_mae = metrics.get(key, {}).get("MAE")
            if c0_mae and c1_mae:
                delta = c1_mae - c0_mae
                print(f"  h={h}: C0={c0_mae:.4f}  C1_energy={c1_mae:.4f}  "
                      f"delta={delta:+.4f} ({delta/c0_mae*100:+.1f}%)")

    if args.test_run:
        print("\n[test-run] OK. Lanzar sin --test-run para el backtesting completo.")
        return

    # Guardar
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = RESULTS_DIR / f"{MODEL_NAME}_predictions.parquet"
    df_preds.to_parquet(preds_path, index=False)
    print(f"\nPredicciones: {preds_path}")

    metrics_path = RESULTS_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({MODEL_NAME: metrics}, f, indent=2)
    print(f"Metricas:     {metrics_path}")


if __name__ == "__main__":
    main()
