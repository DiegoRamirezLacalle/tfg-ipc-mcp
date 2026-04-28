"""
13_build_features_c1_europe.py — Construccion features_c1_europe.parquet

Combina cuatro fuentes de datos mas la nueva senial Europa:

  hicp_europe_index.parquet     -> hicp_index (target)
  ecb_rates_monthly.parquet     -> dfr, mrr, dfr_ma3 (computado aqui)
  energy_prices_monthly.parquet -> brent_ma3, ttf_ma3
  institutional_signals_monthly -> epu_europe_ma3
  europe_signals_monthly.parq.  -> esi_eurozone, breakeven_5y_lag1, eurusd_ma3
  news_signals.parquet          -> bce_shock_score, bce_tone_numeric,
                                   bce_cumstance, gdelt_tone_ma6, signal_available

Rango final: 2002-01 a 2024-12 (276 filas).
Las columnas MCP (news_signals) son NaN antes de 2015-01.

Tambien imprime correlaciones con hicp_index(t+1) para orientar el diseño C1.

Salida: data/processed/features_c1_europe.parquet
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR    = ROOT / "data" / "processed"
OUTPUT_PATH = DATA_DIR / "features_c1_europe.parquet"
FREQ        = "MS"
DATE_START  = "2002-01-01"
DATE_END    = "2024-12-01"


def main():
    print("=" * 60)
    print("BUILD features_c1_europe.parquet")
    print("=" * 60)

    idx = pd.date_range(start=DATE_START, end=DATE_END, freq=FREQ)

    # ── 1. Target: HICP Europa ───────────────────────────────────────
    df_hicp = pd.read_parquet(DATA_DIR / "hicp_europe_index.parquet")
    df_hicp["date"] = pd.to_datetime(df_hicp["date"])
    df_hicp = df_hicp.set_index("date")
    hicp = df_hicp["hicp_index"].reindex(idx)
    print(f"HICP Europa:      {hicp.notna().sum()} obs  "
          f"(rango {hicp.min():.2f}-{hicp.max():.2f})")

    # ── 2. ECB rates: DFR, MRR, dfr_ma3 ────────────────────────────
    ecb = pd.read_parquet(DATA_DIR / "ecb_rates_monthly.parquet")
    ecb.index = pd.DatetimeIndex(ecb.index, freq=FREQ)
    dfr = ecb["dfr"].reindex(idx).ffill()
    mrr = ecb["mrr"].reindex(idx).ffill()
    dfr_ma3 = dfr.rolling(3).mean().shift(1)  # shift evita leakage
    dfr_s   = dfr.shift(1)                    # usar el valor del mes anterior
    print(f"DFR:              {dfr.notna().sum()} obs  "
          f"(rango {dfr.min():.2f}%-{dfr.max():.2f}%)")

    # ── 3. Energy: brent_ma3, ttf_ma3 ───────────────────────────────
    en = pd.read_parquet(DATA_DIR / "energy_prices_monthly.parquet")
    en.index = pd.DatetimeIndex(en.index, freq=FREQ)
    brent_ma3 = en["brent_ma3"].reindex(idx).ffill()
    ttf_ma3   = en["ttf_ma3"].reindex(idx).ffill()
    print(f"Brent ma3:        {brent_ma3.notna().sum()} obs")
    print(f"TTF ma3:          {ttf_ma3.notna().sum()} obs")

    # ── 4. Institutional: epu_europe_ma3 ────────────────────────────
    inst = pd.read_parquet(DATA_DIR / "institutional_signals_monthly.parquet")
    inst["date"] = pd.to_datetime(inst["date"])
    inst = inst.set_index("date")
    epu_europe_ma3 = inst["epu_europe_ma3"].reindex(idx).ffill()
    print(f"EPU Europe ma3:   {epu_europe_ma3.notna().sum()} obs")

    # ── 5. Europe signals: ESI, breakeven, EURUSD ───────────────────
    eur = pd.read_parquet(DATA_DIR / "europe_signals_monthly.parquet")
    eur.index = pd.DatetimeIndex(eur.index, freq=FREQ)
    esi_eurozone     = eur["esi_eurozone"].reindex(idx).ffill().bfill()
    brk_lag1         = eur["breakeven_5y_lag1"].reindex(idx).ffill().bfill()
    eurusd_ma3       = eur["eurusd_ma3"].reindex(idx).ffill().bfill()
    print(f"ESI Eurozone:     {esi_eurozone.notna().sum()} obs  "
          f"(rango {esi_eurozone.min():.1f}-{esi_eurozone.max():.1f})")
    print(f"Breakeven 5Y lag1:{brk_lag1.notna().sum()} obs")
    print(f"EUR/USD ma3:      {eurusd_ma3.notna().sum()} obs")

    # ── 6. News / MCP signals ───────────────────────────────────────
    news = pd.read_parquet(DATA_DIR / "news_signals.parquet")
    news["date"] = pd.to_datetime(news["date"])
    news = news.set_index("date")
    mcp_cols = ["bce_shock_score", "bce_tone_numeric", "bce_cumstance",
                "gdelt_tone_ma6", "signal_available"]
    # Sin shift: estas ya tienen leakage controlado en origen (el BCE habla sobre el pasado)
    bce_shock     = news["bce_shock_score"].reindex(idx)
    bce_tone_num  = news["bce_tone_numeric"].reindex(idx)
    bce_cumstance = news["bce_cumstance"].reindex(idx)
    gdelt_ma6     = news["gdelt_tone_ma6"].reindex(idx)
    sig_avail     = news["signal_available"].reindex(idx).fillna(0)
    print(f"BCE shock score:  {bce_shock.notna().sum()} obs  "
          f"(desde {news.index.min().date()})")

    # ── Ensamblar dataframe final ────────────────────────────────────
    df_out = pd.DataFrame({
        "hicp_index":       hicp,
        "dfr":              dfr_s,
        "dfr_ma3":          dfr_ma3,
        "mrr":              mrr.shift(1),
        "brent_ma3":        brent_ma3,
        "ttf_ma3":          ttf_ma3,
        "epu_europe_ma3":   epu_europe_ma3,
        "esi_eurozone":     esi_eurozone,
        "breakeven_5y_lag1": brk_lag1,
        "eurusd_ma3":       eurusd_ma3,
        "bce_shock_score":  bce_shock,
        "bce_tone_numeric": bce_tone_num,
        "bce_cumstance":    bce_cumstance,
        "gdelt_tone_ma6":   gdelt_ma6,
        "signal_available": sig_avail,
    }, index=idx)
    df_out.index.name = "date"
    df_out.index.freq = FREQ

    print(f"\nDataFrame final: {df_out.shape}")
    print(f"NaN por columna:")
    for c in df_out.columns:
        n_na = df_out[c].isna().sum()
        if n_na > 0:
            print(f"  {c:<22} {n_na:>4} NaN")

    # ── Correlaciones con hicp(t+1) ──────────────────────────────────
    print("\n" + "=" * 55)
    print("Correlaciones con hicp_index(t+1)")
    print("=" * 55)
    target_fwd = df_out["hicp_index"].shift(-1)
    feats = [c for c in df_out.columns if c != "hicp_index"]
    corrs = []
    for c in feats:
        r = df_out[c].corr(target_fwd)
        corrs.append((c, round(float(r), 4) if pd.notna(r) else float("nan")))
    corrs.sort(key=lambda x: abs(x[1]) if pd.notna(x[1]) else 0, reverse=True)

    print(f"{'Feature':<24} {'Corr':>8}")
    print("-" * 35)
    for name, r in corrs:
        mark = " ***" if abs(r) > 0.15 else (" **" if abs(r) > 0.10 else "")
        print(f"  {name:<22} {r:>8.4f}{mark}")
    print("  (*** >0.15, ** >0.10)")

    # ── Correlaciones MCP vs residuos Chronos2 C0 ────────────────────
    c0_preds_path = ROOT / "08_results" / "chronos2_C0_europe_predictions.parquet"
    if c0_preds_path.exists():
        print("\n" + "=" * 55)
        print("Correlaciones MCP vs residuos Chronos-2 C0 Europe")
        print("(criterio de parada: >0.15 => experimento MCP solido)")
        print("=" * 55)
        c0 = pd.read_parquet(c0_preds_path)
        c0_h12 = c0[c0["horizon"] == 12].copy()
        c0_h12 = c0_h12.groupby("fc_date").agg(
            residual=("error", "mean")
        ).reset_index()
        c0_h12["fc_date"] = pd.to_datetime(c0_h12["fc_date"])
        c0_h12 = c0_h12.set_index("fc_date")["residual"]

        for mc in ["bce_shock_score", "bce_tone_numeric", "bce_cumstance", "gdelt_tone_ma6"]:
            feat_aligned = df_out[mc].reindex(c0_h12.index)
            r = feat_aligned.corr(c0_h12)
            mark = " *** SOLIDO" if abs(r) > 0.15 else (" ** marginal" if abs(r) > 0.10 else "")
            print(f"  {mc:<24} corr={r:+.4f}{mark}")

    # ── Guardar ──────────────────────────────────────────────────────
    df_out.to_parquet(OUTPUT_PATH)
    print(f"\nGuardado: {OUTPUT_PATH}")
    print(f"  Shape: {df_out.shape}")
    print(f"  Rango: {df_out.index.min().date()} - {df_out.index.max().date()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
