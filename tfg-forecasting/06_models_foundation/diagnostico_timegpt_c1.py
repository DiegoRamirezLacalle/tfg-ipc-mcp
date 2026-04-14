"""
diagnostico_timegpt_c1.py — Diagnostico completo antes de aplicar fix a TimeGPT C1

4 diagnosticos con presupuesto total de 20 llamadas API:
  D1: Regimen de ceros (3 variantes x 1 llamada = 3 llamadas)
  D2: Subconjuntos de senales (5 subconjuntos x 1 llamada = 5 llamadas)
  D3: Estrategia forward-fill (3 estrategias x 1 llamada = 3 llamadas)
  D4: C0 baseline (1 llamada)
  Total: 12 llamadas (margen de 8 para errores/retries)

Origenes de prueba: 2022-07 a 2022-11 (5 origenes, pleno shock BCE)
Metrica: MAE h=1 (la mas sensible a cambios en covariables)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.constants import DATE_TRAIN_END

SERIES_ID = "ipc_spain"
MAX_H = 12
TEST_ORIGINS = pd.to_datetime([
    "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01",
])

ALL_EXOG = [
    "gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6",
    "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
    "ine_surprise_score", "ine_inflacion",
    "signal_available",
]

api_calls = 0
MAX_API_CALLS = 35


# ── Datos ──────────────────────────────────────────────────────

def load_ipc() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "processed" / "ipc_spain_index.parquet")
    y = df["indice_general"]
    y.index = pd.to_datetime(y.index)
    y.index.freq = "MS"
    return y


def load_features() -> pd.DataFrame:
    c1 = pd.read_parquet(ROOT / "data" / "processed" / "features_c1.parquet")
    c1["date"] = pd.to_datetime(c1["date"])
    c1 = c1.set_index("date")
    c1.index.freq = "MS"
    for col in ALL_EXOG:
        if col in c1.columns:
            c1[col] = c1[col].fillna(0.0)
    return c1


def get_client():
    load_dotenv(MONOREPO / ".env")
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError("NIXTLA_API_KEY no configurada.")
    from nixtla import NixtlaClient
    return NixtlaClient(api_key=api_key)


# ── Helpers comunes ────────────────────────────────────────────

def build_hist_df(
    y: pd.Series,
    exog: pd.DataFrame | None,
    origin: pd.Timestamp,
    exog_cols: list[str] | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """Construye df historico para Nixtla. Si exog=None, solo serie."""
    context_y = y.loc[:origin]
    if start_date:
        context_y = context_y.loc[start_date:]

    hist = pd.DataFrame({
        "unique_id": SERIES_ID,
        "ds": context_y.index,
        "y": context_y.values,
    })

    if exog is not None and exog_cols:
        context_exog = exog.loc[context_y.index[0]:origin, exog_cols]
        # Reindex to match exactly
        context_exog = context_exog.reindex(context_y.index)
        for col in exog_cols:
            if col in context_exog.columns:
                hist[col] = context_exog[col].values
            else:
                hist[col] = 0.0

    return hist


def build_future_df(
    origin: pd.Timestamp,
    exog: pd.DataFrame | None,
    exog_cols: list[str] | None = None,
    fill_strategy: str = "forward",
) -> pd.DataFrame | None:
    """Construye future_df para Nixtla. Estrategias: forward, zero, mean3."""
    if exog is None or not exog_cols:
        return None

    fc_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1), periods=MAX_H, freq="MS"
    )
    future = pd.DataFrame({"unique_id": SERIES_ID, "ds": fc_dates})

    last_row = exog.loc[:origin, exog_cols].iloc[-1]

    if fill_strategy == "forward":
        for col in exog_cols:
            if col == "signal_available":
                future[col] = 1.0
            else:
                future[col] = float(last_row[col])

    elif fill_strategy == "zero":
        for col in exog_cols:
            future[col] = 0.0

    elif fill_strategy == "mean3":
        last3 = exog.loc[:origin, exog_cols].tail(3).mean()
        for col in exog_cols:
            if col == "signal_available":
                future[col] = 1.0
            else:
                future[col] = float(last3[col])

    return future


def forecast_one_origin(
    client,
    y: pd.Series,
    origin: pd.Timestamp,
    exog: pd.DataFrame | None = None,
    exog_cols: list[str] | None = None,
    start_date: str | None = None,
    fill_strategy: str = "forward",
) -> float | None:
    """
    Hace UNA llamada API para un origen.
    Devuelve la prediccion h=1 (un solo paso adelante).
    """
    global api_calls
    if api_calls >= MAX_API_CALLS:
        print(f"  [!] Limite de {MAX_API_CALLS} llamadas alcanzado, saltando.")
        return None

    hist = build_hist_df(y, exog, origin, exog_cols, start_date)
    future = build_future_df(origin, exog, exog_cols, fill_strategy)

    kwargs = dict(
        df=hist,
        h=MAX_H,
        freq="MS",
        time_col="ds",
        target_col="y",
        id_col="unique_id",
    )
    if future is not None:
        kwargs["X_df"] = future

    try:
        fc = client.forecast(**kwargs)
        api_calls += 1
        fc = fc.sort_values("ds").reset_index(drop=True)
        return float(fc["TimeGPT"].iloc[0])  # h=1 prediction
    except Exception as e:
        api_calls += 1
        print(f"  [!] Error en {origin.date()}: {e}")
        return None


def compute_mae_h1(
    preds: list[float | None],
    actuals: list[float],
) -> float | None:
    """MAE h=1 sobre pares validos."""
    pairs = [(p, a) for p, a in zip(preds, actuals) if p is not None]
    if not pairs:
        return None
    return float(np.mean([abs(p - a) for p, a in pairs]))


# ── Diagnostico 1: Regimen de ceros ────────────────────────────

def diag1_regimen_ceros(client, y, exog):
    print("\n" + "=" * 60)
    print("DIAGNOSTICO 1 — Regimen de ceros pre-2015")
    print("=" * 60)

    actuals = []
    for origin in TEST_ORIGINS:
        target_date = origin + pd.DateOffset(months=1)
        actuals.append(float(y.loc[target_date]))

    results = {}

    # Variante A: dataset completo (282 filas, con ceros pre-2015)
    print("\n[A] Dataset completo (282 filas, ceros pre-2015)...")
    preds_a = []
    for origin in TEST_ORIGINS:
        p = forecast_one_origin(client, y, origin, exog, ALL_EXOG)
        preds_a.append(p)
    results["A_completo"] = compute_mae_h1(preds_a, actuals)

    # Variante B: recortado desde 2015 (~120 filas, solo senales reales)
    print("[B] Dataset desde 2015 (~120 filas, solo senales reales)...")
    preds_b = []
    for origin in TEST_ORIGINS:
        p = forecast_one_origin(client, y, origin, exog, ALL_EXOG,
                                start_date="2015-01-01")
        preds_b.append(p)
    results["B_desde2015"] = compute_mae_h1(preds_b, actuals)

    # Variante C: NaN pre-2015 (TimeGPT interpola)
    # Crear copia de exog con NaN en lugar de 0 para pre-2015
    print("[C] Dataset completo, NaN pre-2015 (TimeGPT interpola)...")
    exog_nan = exog.copy()
    mask_pre2015 = exog_nan.index < "2015-01-01"
    for col in ALL_EXOG:
        if col in exog_nan.columns and col != "signal_available":
            exog_nan.loc[mask_pre2015, col] = np.nan
    # signal_available ya deberia ser 0 pre-2015, pero forzamos NaN tambien
    # para que TimeGPT no vea ningun patron artificial
    # Nota: Nixtla puede no aceptar NaN en exogenas, en ese caso usamos -999
    # como missing indicator
    preds_c = []
    for origin in TEST_ORIGINS:
        p = forecast_one_origin(client, y, origin, exog_nan, ALL_EXOG)
        preds_c.append(p)
    results["C_nan_pre2015"] = compute_mae_h1(preds_c, actuals)

    return results, actuals


# ── Diagnostico 2: Subconjuntos de senales ─────────────────────

def diag2_subconjuntos(client, y, exog):
    print("\n" + "=" * 60)
    print("DIAGNOSTICO 2 — Subconjuntos de senales")
    print("=" * 60)

    actuals = []
    for origin in TEST_ORIGINS:
        target_date = origin + pd.DateOffset(months=1)
        actuals.append(float(y.loc[target_date]))

    subsets = {
        "GDELT": ["gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6"],
        "BCE": ["bce_shock_score", "bce_tone_numeric", "bce_cumstance"],
        "INE": ["ine_surprise_score", "ine_inflacion"],
        "cumstance+avail": ["bce_cumstance", "signal_available"],
        "ALL_9": ALL_EXOG,
    }

    results = {}
    for name, cols in subsets.items():
        print(f"\n[{name}] Covariables: {cols}")
        preds = []
        for origin in TEST_ORIGINS:
            p = forecast_one_origin(client, y, origin, exog, cols)
            preds.append(p)
        results[name] = compute_mae_h1(preds, actuals)

    return results, actuals


# ── Diagnostico 3: Estrategia forward-fill ─────────────────────

def diag3_fill_strategy(client, y, exog):
    print("\n" + "=" * 60)
    print("DIAGNOSTICO 3 — Estrategia de relleno del horizonte futuro")
    print("=" * 60)

    actuals = []
    for origin in TEST_ORIGINS:
        target_date = origin + pd.DateOffset(months=1)
        actuals.append(float(y.loc[target_date]))

    strategies = ["forward", "zero", "mean3"]
    results = {}

    for strat in strategies:
        print(f"\n[{strat}] Rellenando horizonte futuro con estrategia '{strat}'...")
        preds = []
        for origin in TEST_ORIGINS:
            p = forecast_one_origin(client, y, origin, exog, ALL_EXOG,
                                    fill_strategy=strat)
            preds.append(p)
        results[strat] = compute_mae_h1(preds, actuals)

    return results, actuals


# ── Diagnostico 4: C0 baseline ────────────────────────────────

def diag4_c0_baseline(client, y):
    print("\n" + "=" * 60)
    print("DIAGNOSTICO 4 — C0 baseline (sin exogenas)")
    print("=" * 60)

    actuals = []
    for origin in TEST_ORIGINS:
        target_date = origin + pd.DateOffset(months=1)
        actuals.append(float(y.loc[target_date]))

    preds = []
    for origin in TEST_ORIGINS:
        p = forecast_one_origin(client, y, origin)
        preds.append(p)

    mae = compute_mae_h1(preds, actuals)
    return mae, actuals


# ── Main ──────────────────────────────────────────────────────

def main():
    global api_calls

    print("=" * 60)
    print("DIAGNOSTICO TIMEGPT C1 — Identificar causa raiz")
    print(f"Origenes de prueba: {[o.strftime('%Y-%m') for o in TEST_ORIGINS]}")
    print(f"Presupuesto API: {MAX_API_CALLS} llamadas")
    print("=" * 60)

    y = load_ipc()
    exog = load_features()
    client = get_client()

    print(f"Serie IPC: {y.index.min().date()} - {y.index.max().date()} ({len(y)} obs)")
    print(f"Features C1: {exog.index.min().date()} - {exog.index.max().date()} ({len(exog)} obs)")

    # ── D4 primero (1 llamada, baseline necesario para todo) ────
    mae_c0, actuals = diag4_c0_baseline(client, y)
    print(f"\n  >>> C0 baseline MAE h=1: {mae_c0:.4f}")
    print(f"  >>> Llamadas API usadas: {api_calls}/{MAX_API_CALLS}")

    # ── D1: Regimen de ceros (3 variantes, ~3 llamadas por variante
    #    pero usamos un solo origin para ahorrar, o todos 5) ────
    # Optimizacion: para D1 usamos solo 1 llamada por variante (batch)
    # TimeGPT acepta multiples series, pero aqui hacemos 1 por origen
    # Ajuste: usar todos los 5 origenes pero contar llamadas
    d1_results, _ = diag1_regimen_ceros(client, y, exog)
    print(f"\n  >>> Llamadas API usadas: {api_calls}/{MAX_API_CALLS}")

    # ── D2: Subconjuntos de senales ────
    # 5 subconjuntos x 5 origenes = 25 llamadas... demasiado
    # Optimizacion: usar solo 1 origen representativo (2022-09) para D2
    print("\n[!] D2 optimizado: usando 1 origen representativo (2022-09)")

    # Redifinir para usar solo 1 origen
    single_origin = pd.to_datetime(["2022-09-01"])
    target_date = single_origin[0] + pd.DateOffset(months=1)
    single_actual = [float(y.loc[target_date])]

    subsets = {
        "GDELT": ["gdelt_avg_tone", "gdelt_tone_ma3", "gdelt_tone_ma6"],
        "BCE": ["bce_shock_score", "bce_tone_numeric", "bce_cumstance"],
        "INE": ["ine_surprise_score", "ine_inflacion"],
        "cumstance+avail": ["bce_cumstance", "signal_available"],
        "ALL_9": ALL_EXOG,
    }

    print("\n" + "=" * 60)
    print("DIAGNOSTICO 2 — Subconjuntos de senales (origen: 2022-09)")
    print("=" * 60)

    d2_results = {}
    for name, cols in subsets.items():
        if api_calls >= MAX_API_CALLS:
            print(f"  [!] Limite alcanzado, saltando {name}")
            d2_results[name] = None
            continue
        print(f"  [{name}] {cols}")
        p = forecast_one_origin(client, y, single_origin[0], exog, cols)
        if p is not None:
            d2_results[name] = abs(p - single_actual[0])
        else:
            d2_results[name] = None
    print(f"\n  >>> Llamadas API usadas: {api_calls}/{MAX_API_CALLS}")

    # ── D3: Estrategia fill (3 estrategias x 1 origen) ────
    print("\n" + "=" * 60)
    print("DIAGNOSTICO 3 — Estrategia fill horizonte (origen: 2022-09)")
    print("=" * 60)

    d3_results = {}
    for strat in ["forward", "zero", "mean3"]:
        if api_calls >= MAX_API_CALLS:
            print(f"  [!] Limite alcanzado, saltando {strat}")
            d3_results[strat] = None
            continue
        print(f"  [{strat}]")
        p = forecast_one_origin(client, y, single_origin[0], exog, ALL_EXOG,
                                fill_strategy=strat)
        if p is not None:
            d3_results[strat] = abs(p - single_actual[0])
        else:
            d3_results[strat] = None
    print(f"\n  >>> Llamadas API usadas: {api_calls}/{MAX_API_CALLS}")

    # ── RESUMEN ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DIAGNOSTICO TIMEGPT C1")
    print("=" * 60)

    # D4 + D1
    print(f"\nC0 baseline MAE h=1 (5 origenes): {mae_c0:.4f}" if mae_c0 else
          "\nC0 baseline: ERROR")

    print("\nHipotesis regimen ceros:")
    for name, mae in d1_results.items():
        if mae is not None and mae_c0 is not None:
            delta_pct = (mae - mae_c0) / mae_c0 * 100
            print(f"  {name:20s}: MAE = {mae:.4f}  (vs C0: {delta_pct:+.1f}%)")
        elif mae is not None:
            print(f"  {name:20s}: MAE = {mae:.4f}")
        else:
            print(f"  {name:20s}: ERROR")

    # Determinar hipotesis
    a = d1_results.get("A_completo")
    b = d1_results.get("B_desde2015")
    if a is not None and b is not None:
        if b < a * 0.90:
            hyp1 = "CONFIRMADA"
            hyp1_detail = f"Recortar a 2015+ reduce MAE un {(a-b)/a*100:.1f}%"
        elif b > a * 1.10:
            hyp1 = "NO CONFIRMADA (recortar empeora)"
            hyp1_detail = f"Recortar a 2015+ AUMENTA MAE un {(b-a)/a*100:.1f}%"
        else:
            hyp1 = "INDETERMINADA (diferencia < 10%)"
            hyp1_detail = f"Diferencia: {(b-a)/a*100:+.1f}%"
    else:
        hyp1 = "ERROR en calculo"
        hyp1_detail = ""

    print(f"\n  >>> Hipotesis regimen ceros: {hyp1}")
    if hyp1_detail:
        print(f"      {hyp1_detail}")

    # D2
    print("\nSenales individuales (AE h=1, origen 2022-09):")
    d2_sorted = sorted(
        [(k, v) for k, v in d2_results.items() if v is not None],
        key=lambda x: x[1]
    )
    # C0 ref para este origen
    c0_ref_p = forecast_one_origin(client, y, single_origin[0])
    c0_ref_ae = abs(c0_ref_p - single_actual[0]) if c0_ref_p is not None else None

    if c0_ref_ae is not None:
        print(f"  {'C0 (sin exog)':20s}: AE = {c0_ref_ae:.4f}  (referencia)")
    for name, ae in d2_sorted:
        if c0_ref_ae is not None:
            delta = (ae - c0_ref_ae) / c0_ref_ae * 100
            marker = " <-- MEJOR" if ae < c0_ref_ae else ""
            print(f"  {name:20s}: AE = {ae:.4f}  (vs C0: {delta:+.1f}%){marker}")
        else:
            print(f"  {name:20s}: AE = {ae:.4f}")

    # Identificar mas daninas y neutras
    if d2_sorted:
        neutrals = [k for k, v in d2_sorted if c0_ref_ae and v <= c0_ref_ae * 1.05]
        harmful = [k for k, v in d2_sorted if c0_ref_ae and v > c0_ref_ae * 1.20]
        print(f"\n  Senales mas neutras (<=5% degradacion): {neutrals or ['ninguna']}")
        print(f"  Senales mas daninas (>20% degradacion): {harmful or ['ninguna']}")

    # D3
    print("\nEstrategia forward-fill horizonte (AE h=1, origen 2022-09):")
    best_strat = None
    best_ae = float("inf")
    for strat in ["forward", "zero", "mean3"]:
        ae = d3_results.get(strat)
        if ae is not None:
            marker = ""
            if ae < best_ae:
                best_ae = ae
                best_strat = strat
                marker = " <-- MEJOR"
            if c0_ref_ae is not None:
                delta = (ae - c0_ref_ae) / c0_ref_ae * 100
                print(f"  {strat:15s}: AE = {ae:.4f}  (vs C0: {delta:+.1f}%){marker}")
            else:
                print(f"  {strat:15s}: AE = {ae:.4f}{marker}")
        else:
            print(f"  {strat:15s}: ERROR")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION RECOMENDADA")
    print("=" * 60)

    # Determinar mejor config
    best_subset = d2_sorted[0][0] if d2_sorted else "ALL_9"
    best_subset_ae = d2_sorted[0][1] if d2_sorted else None

    recs = []
    if hyp1 == "CONFIRMADA":
        recs.append("Recortar contexto a 2015+ (eliminar regimen de ceros)")
    elif "NO CONFIRMADA" in hyp1:
        recs.append("Mantener contexto completo (los ceros no son el problema)")

    if best_strat and best_strat != "forward":
        recs.append(f"Cambiar estrategia fill a '{best_strat}'")
    else:
        recs.append("Mantener forward-fill (o la mejor de las 3)")

    if best_subset != "ALL_9" and best_subset_ae is not None and c0_ref_ae is not None:
        if best_subset_ae < c0_ref_ae * 1.05:
            recs.append(f"Usar solo covariables '{best_subset}' (neutrales)")
        else:
            recs.append("Todas las senales degradan; considerar reducir a 0 (C0 puro)")

    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")

    print(f"\n  Total llamadas API: {api_calls}/{MAX_API_CALLS}")


if __name__ == "__main__":
    main()
