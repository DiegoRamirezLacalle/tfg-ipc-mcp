"""
02_timesfm_C1.py — TimesFM 2.5 condicion C1 (historico + senales MCP)

Fix 1 — XReg restringido al periodo con senales reales (2015+):
  El modelo base de TimesFM recibe el contexto COMPLETO de IPC (282 obs,
  identico a C0). La correccion por senales MCP se calcula mediante un
  Ridge externo ajustado SOLO sobre df.loc['2015':origin], donde todas
  las covariables tienen valores reales. La correccion es la diferencia
  entre el nivel predicho por el Ridge con las senales actuales y el
  nivel predicho con senales neutrales (ceros).
  Esto implementa correctamente la separacion base-TimesFM / XReg-MCP.

Fix 2 — Seleccion de covariables:
  Columnas de entrada al Ridge: gdelt_avg_tone, gdelt_tone_ma3,
  gdelt_tone_ma6, bce_shock_score, bce_tone_numeric, bce_cumstance,
  ine_surprise_score, ine_inflacion, signal_available.
  Eliminadas: bce_uncertainty (5 valores), gdelt_goldstein/n_articles
  (correladas/ruidosas), dfr_diff/lag3/6/12 (redundantes con tasas
  ya en el contexto IPC).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
TEST_END_TS = pd.Timestamp(DATE_TEST_END)

# Fix 1: XReg se ajusta SOLO desde esta fecha en adelante
SIGNAL_START = "2015-01-01"

# Fix 2: covariables del XReg externo (exactamente las especificadas)
XREG_COVS = [
    "gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6",
    "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
    "ine_surprise_score", "ine_inflacion",
    "signal_available",
]

# Ridge regularization (evita overfitting con ~60-120 obs y 9 features)
RIDGE_ALPHA = 1.0


# ── Datos ────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    for col in XREG_COVS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
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
        )
    )
    print("[timesfm] Modelo base cargado (sin XReg interno, identico a C0)")
    return tfm


# ── Fix 1: Ridge XReg externo ajustado sobre 2015:origin ─────────

def compute_xreg_correction(
    df: pd.DataFrame,
    origin: pd.Timestamp,
) -> float:
    """
    Ajusta Ridge sobre df.loc[SIGNAL_START:origin] SOLO con senales reales.
    Target: cambio mensual del IPC (primera diferencia, estacionario ±1.5 pp).
    Devuelve la correccion marginal:
      correction = beta @ current_signals
                 = Ridge.predict(current) - Ridge.predict(zeros)

    Usar primera diferencia evita que el Ridge sobreajuste la tendencia de
    nivel del IPC (rango 79-100) a las senales, lo que produciria
    correcciones enormes (±5-10 pp).
    Si no hay suficientes datos (< 13 meses con senales), devuelve 0.0.
    """
    signal_start_ts = pd.Timestamp(SIGNAL_START)
    window = df.loc[signal_start_ts:origin].copy()

    # Solo filas con senales reales (post-shift)
    window = window[window["signal_available"] > 0]
    if len(window) < 13:
        return 0.0

    # Cambio mensual del IPC como target (estacionario, rango ±1.5 pp)
    ipc_mom = window["indice_general"].diff(1)
    valid = ~ipc_mom.isna()
    X = window.loc[valid, XREG_COVS].values.astype(np.float64)
    y_diff = ipc_mom[valid].values.astype(np.float64)

    reg = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    reg.fit(X, y_diff)

    current = df.loc[origin:origin, XREG_COVS].values.astype(np.float64)
    neutral = np.zeros_like(current)

    # Efecto marginal de las senales actuales sobre el cambio mensual esperado
    correction = float(reg.predict(current)[0] - reg.predict(neutral)[0])
    return correction


# ── Rolling backtesting ─────────────────────────────────────────

def run_rolling(df: pd.DataFrame, model) -> tuple[pd.DataFrame, float]:
    y = df["indice_general"]
    origins = pd.date_range(start=ORIGINS_START, end=ORIGINS_END, freq="MS")

    y_train_init = y.loc[:DATE_TRAIN_END]
    mase_scale = float(np.mean(np.abs(
        y_train_init.values[12:] - y_train_init.values[:-12]
    )))

    records = []

    for origin in tqdm(origins, desc="TimesFM C1 rolling"):
        # Fix diagnostico: contexto 2015+ (protocolo consistente con TimeGPT C1)
        context = y.loc[SIGNAL_START:origin].values.astype(np.float32)
        try:
            point_out, _ = model.forecast(horizon=MAX_H, inputs=[context])
            base_pred = np.array(point_out[0])  # shape (MAX_H,)
        except Exception as e:
            print(f"\n[!] Error base en {origin.date()}: {e}")
            continue

        # Fix 1 — Correccion Ridge: ajustada SOLO sobre 2015:origin
        xreg_correction = compute_xreg_correction(df, origin)

        # C1 = TimesFM_base + correccion_MCP (misma para todos los h)
        full_pred = base_pred + xreg_correction

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
    print(f"Fix 1: base TimesFM C0 (282 obs) + Ridge externo sobre 2015:origin")
    print(f"Fix 2: XReg covariables = {XREG_COVS}")
    print(f"Origenes: {ORIGINS_START} - {ORIGINS_END}")
    print(f"Horizontes: {HORIZONS}")
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
                pct = (delta / c0_mae) * 100
                print(f"  h={h}: C0={c0_mae:.4f}  C1={c1_mae:.4f}  "
                      f"delta={delta:+.4f} ({pct:+.1f}%)")

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
