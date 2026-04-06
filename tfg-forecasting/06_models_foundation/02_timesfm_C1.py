"""
02_timesfm_C1.py — TimesFM 2.5 condicion C1 (historico + senales MCP)

Rolling-origin backtesting con covariables del pipeline MCP:
  - Numericas: gdelt_avg_tone, gdelt_goldstein, gdelt_n_articles,
               bce_shock_score, bce_uncertainty, ine_surprise_score,
               dfr, mrr, dfr_diff, dfr_lag3, dfr_lag6, dfr_lag12
  - Categoricas: bce_tone, dominant_topic

Para el horizonte futuro se usa forward-fill del ultimo valor conocido
(en un contexto real no conocemos senales de noticias futuras).

Las senales MCP ya tienen shift +1 aplicado en features_c1.parquet,
por lo que no hay leakage temporal.

Modelo: google/timesfm-2.5-200m-pytorch + forecast_with_covariates()
Requiere: return_backcast=True en ForecastConfig
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
MODEL_NAME = "timesfm_C1"

# Covariables numericas de features_c1.parquet
NUM_COVARIATES = [
    "dfr", "mrr", "dfr_diff", "dfr_lag3", "dfr_lag6", "dfr_lag12",
    "gdelt_avg_tone", "gdelt_goldstein", "gdelt_n_articles",
    "bce_shock_score", "bce_uncertainty", "ine_surprise_score",
]

# Covariables categoricas
CAT_COVARIATES = ["bce_tone", "dominant_topic"]


# ── Datos ────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"

    # Rellenar NaN en covariables (meses antes de 2015 sin senales MCP)
    for col in NUM_COVARIATES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    for col in CAT_COVARIATES:
        if col in df.columns:
            df[col] = df[col].fillna("neutral")

    # ine_topic no es covariable directa — se integra via ine_surprise_score
    return df


# ── Modelo ───────────────────────────────────────────────────────

def load_model():
    import timesfm

    print("[timesfm] Cargando modelo google/timesfm-2.5-200m-pytorch ...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
    )
    tfm.compile(
        forecast_config=timesfm.ForecastConfig(
            max_context=512,
            max_horizon=MAX_H,
            per_core_batch_size=1,
            return_backcast=True,  # Requerido para forecast_with_covariates
        )
    )
    print("[timesfm] Modelo cargado (max_horizon=12, return_backcast=True)")
    return tfm


# ── Preparar covariables para un origen dado ─────────────────────

def prepare_covariates(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    h: int,
) -> tuple[dict, dict]:
    """
    Prepara covariables para forecast_with_covariates().

    Las covariables deben tener longitud = len(context) + horizon.
    Para el horizonte futuro, forward-fill del ultimo valor conocido.
    """
    context = df.loc[:origin]
    ctx_len = len(context)

    # Valores futuros: forward-fill
    last_row = context.iloc[-1]

    dyn_num = {}
    for col in NUM_COVARIATES:
        if col not in df.columns:
            continue
        hist = context[col].values.astype(np.float64)
        future = np.full(h, float(last_row[col]))
        full = np.concatenate([hist, future])
        dyn_num[col] = [full.tolist()]

    dyn_cat = {}
    for col in CAT_COVARIATES:
        if col not in df.columns:
            continue
        hist = context[col].values.tolist()
        future = [str(last_row[col])] * h
        full = hist + future
        dyn_cat[col] = [full]

    return dyn_num, dyn_cat


# ── Rolling backtesting ─────────────────────────────────────────

def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["indice_general"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    # Escala MASE fija
    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="TimesFM C1 rolling"):
        context = y.loc[:origin].values.astype(np.float32)

        # Preparar covariables para h=12 (maximo horizonte)
        dyn_num, dyn_cat = prepare_covariates(df, origin, MAX_H)

        # Forecast con covariables
        try:
            point_out, _ = model.forecast_with_covariates(
                inputs=[context.tolist()],
                dynamic_numerical_covariates=dyn_num if dyn_num else None,
                dynamic_categorical_covariates=dyn_cat if dyn_cat else None,
                xreg_mode="xreg + timesfm",
                normalize_xreg_target_per_input=True,
                force_on_cpu=True,
            )
            full_pred = np.array(point_out[0])
        except Exception as e:
            print(f"\n[!] Error en {origin.date()}: {e}")
            continue

        for h in HORIZONS:
            fc_dates = pd.date_range(
                start=origin + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            y_actual = y.reindex(fc_dates)
            if y_actual.isna().any():
                continue

            y_true = y_actual.values
            y_pred = full_pred[:h]

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
    print("=" * 60)
    print(f"BACKTESTING ROLLING — {MODEL_NAME}")
    print(f"Origenes: {ORIGINS_START} - {ORIGINS_END}")
    print(f"Horizontes: {HORIZONS}")
    print(f"Covariables num: {NUM_COVARIATES}")
    print(f"Covariables cat: {CAT_COVARIATES}")
    print("=" * 60)

    df = load_data()
    print(f"Datos: {df.index.min().date()} - {df.index.max().date()} ({len(df)} obs)")

    model = load_model()

    df_preds, mase_scale = run_rolling(df, model)
    print(f"\nPredicciones generadas: {len(df_preds)}")

    metrics = compute_metrics(df_preds, mase_scale)

    print("\n" + "=" * 60)
    print(f"RESULTADOS {MODEL_NAME}")
    print("=" * 60)
    print_table(metrics)

    # Comparativa C0 vs C1
    c0_path = RESULTS_DIR / "timesfm_C0_metrics.json"
    if c0_path.exists():
        with open(c0_path) as f:
            c0_metrics = json.load(f).get("timesfm_C0", {})
        print("\n--- C0 vs C1 (MAE) ---")
        for h in HORIZONS:
            key = f"h{h}"
            c0_mae = c0_metrics.get(key, {}).get("MAE", "N/A")
            c1_mae = metrics.get(key, {}).get("MAE", "N/A")
            if isinstance(c0_mae, float) and isinstance(c1_mae, float):
                delta = c1_mae - c0_mae
                print(f"  h={h}: C0={c0_mae:.4f}  C1={c1_mae:.4f}  delta={delta:+.4f}")

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
