"""
09_regime_context_overlay.py
----------------------------
Phase 1 + Phase 3 improvement over 08_validated_context_overlay.py.

Same leakage-safe protocol (select a Ridge context-correction recipe on
PRE-2021 validation origins only, then apply the frozen recipe to the stored
C0 foundation forecasts -- no retraining, no C0/C1 forecast overwritten), but
with two additions motivated by the Phase 0 diagnosis:

  Phase 1 -- change-correlated signal selection.
    Besides the hand-picked feature families, a `change_sel` family is built per
    series from the THREE signals with the highest |corr with the next-month
    target increment| measured on the TRAINING window only (<= 2020-12). This is
    a pre-registered rule (top-3 by training correlation), not a test-window
    search. It replaces level-correlated covariates (imf_comm_ma3, brent, epu)
    with change-informative ones (imf_comm_diff, vix, dfr_diff for Global;
    the BCE/GDELT semantic signals for Europe).

  Phase 3 -- regime gating.
    The subperiod DM evidence says signals help in the 2022 shock, not in calm
    periods. We pre-register a FIXED gate: the correction is applied at an origin
    only when the trailing-12m realised volatility of the target's MoM change
    (computed from data <= origin) exceeds the TRAINING-period 80th percentile of
    that same volatility. The threshold is fixed a priori on <= 2020-12 data; it
    is NOT tuned on the 2021-2024 test window. Validation (2017-2020) is calm, so
    the recipe is *selected* ungated and only *gated* at test.

Outputs (one per covered family/series):
  08_results/{family}_C1_regime{,_global,_europe}_predictions.parquet
  08_results/{family}_C1_regime{,_global,_europe}_metrics.json
  08_results/regime_context_overlay_selection.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "08_results"

HORIZONS = [1, 3, 6, 12]
VALIDATION_START = pd.Timestamp("2017-01-01")
VALIDATION_END = pd.Timestamp("2020-11-01")
TRAIN_END = pd.Timestamp("2020-12-01")
MIN_TRAIN_ROWS = 72
ALPHAS = [0.1, 1.0, 10.0, 100.0]
SCALES = [0.25, 0.5, 1.0]
TARGET_MODES = ["resid_12ma", "next_diff"]
CHANGE_SEL_K = 3            # top-k signals by training |corr with next increment|
REGIME_Q = 0.80            # training quantile of trailing vol that opens the gate
REGIME_VOL_WINDOW = 12


@dataclass(frozen=True)
class SeriesConfig:
    key: str
    target_col: str
    feature_file: Path
    feature_sets: dict[str, list[str]]
    overlays: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class RidgeFit:
    cols: list[str]
    fill: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    y_mean: float


SERIES = {
    "spain": SeriesConfig(
        key="spain",
        target_col="indice_general",
        feature_file=DATA / "features_c1.parquet",
        feature_sets={
            "epu_only": ["epu_europe_ma3"],
            "energy": ["brent_ma3", "brent_lag1", "ttf_ma3", "ttf_lag1"],
            "policy_energy": ["dfr", "dfr_lag3", "brent_ma3", "ttf_ma3"],
            "macro_small": ["dfr", "dfr_lag3", "epu_europe_ma3", "brent_ma3"],
        },
        overlays={
            "timesfm_C1_regime": ("timesfm_C0_predictions.parquet", "timesfm_C1_regime_predictions.parquet"),
            "chronos2_C1_regime": ("chronos2_C0_predictions.parquet", "chronos2_C1_regime_predictions.parquet"),
        },
    ),
    "global": SeriesConfig(
        key="global",
        target_col="cpi_global_rate",
        feature_file=DATA / "features_c1_global_institutional.parquet",
        feature_sets={
            "commodities": ["imf_comm_ma3", "imf_comm_lag1", "brent_log_ma3", "brent_log_lag1"],
            "supply": ["gscpi_ma3", "gscpi_lag1"],
            "risk_policy": ["gepu_ma3", "fedfunds_ma3", "vix_ma3"],
            "broad_small": ["imf_comm_ma3", "brent_log_ma3", "gscpi_ma3", "fedfunds_ma3", "gepu_ma3"],
        },
        overlays={
            "timesfm_C1_regime_global": ("timesfm_C0_global_predictions.parquet", "timesfm_C1_regime_global_predictions.parquet"),
            "chronos2_C1_regime_global": ("chronos2_C0_global_predictions.parquet", "chronos2_C1_regime_global_predictions.parquet"),
        },
    ),
    "europe": SeriesConfig(
        key="europe",
        target_col="hicp_index",
        feature_file=DATA / "features_c1_europe.parquet",
        feature_sets={
            "energy": ["brent_ma3", "ttf_ma3"],
            "macro": ["epu_europe_ma3", "esi_eurozone", "dfr_ma3", "breakeven_5y_lag1"],
            "mcp": ["bce_shock_score", "bce_tone_numeric", "bce_cumstance", "gdelt_tone_ma6"],
            "inst": ["epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3",
                     "dfr", "dfr_ma3", "ttf_ma3", "breakeven_5y_lag1"],
        },
        overlays={
            "timesfm_C1_regime_europe": ("timesfm_C0_europe_predictions.parquet", "timesfm_C1_regime_europe_predictions.parquet"),
            "chronos2_C1_regime_europe": ("chronos2_C0_europe_predictions.parquet", "chronos2_C1_regime_europe_predictions.parquet"),
        },
    ),
}


def load_feature_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def target_for_mode(df: pd.DataFrame, target_col: str, mode: str) -> pd.Series:
    y = df[target_col].astype(float)
    next_diff = y.shift(-1) - y
    if mode == "next_diff":
        return next_diff
    if mode == "resid_12ma":
        expected = y.diff().rolling(12, min_periods=6).mean()
        return next_diff - expected
    raise ValueError(f"Unknown target mode: {mode}")


def change_selected_cols(df: pd.DataFrame, target_col: str, k: int = CHANGE_SEL_K) -> list[str]:
    """Top-k signals by |corr with next-month increment| on the TRAINING window.

    The increment at row t is y[t+1] - y[t], so a row only counts as training if
    its FULL difference lies inside the pre-2021 window, i.e. t+1 <= TRAIN_END.
    Using t = TRAIN_END would pull y at 2021-01 (the first test month) into the
    selection -- a one-month leak. Restrict to t <= TRAIN_END - 1 month.
    """
    y = df[target_col].astype(float)
    nxt = y.shift(-1) - y
    tr = df.index <= (TRAIN_END - pd.DateOffset(months=1))
    scored = []
    for c in df.columns:
        if c == target_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            continue
        cc = s[tr].corr(nxt[tr])
        if pd.notna(cc):
            scored.append((c, abs(float(cc))))
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:k]]


def regime_threshold_and_vol(df: pd.DataFrame, target_col: str) -> tuple[float, pd.Series]:
    """Trailing realised vol of the target's MoM change, plus the training 80pct gate."""
    y = df[target_col].astype(float)
    vol = y.diff().rolling(REGIME_VOL_WINDOW, min_periods=6).std()
    thr = float(vol[df.index <= TRAIN_END].quantile(REGIME_Q))
    return thr, vol


def fit_ridge_context(df: pd.DataFrame, cols: list[str], target: pd.Series, alpha: float) -> RidgeFit | None:
    train = df[cols].copy()
    y = target.astype(float)
    valid = y.notna()
    train = train.loc[valid]
    y = y.loc[valid]
    if len(train) < MIN_TRAIN_ROWS:
        return None

    numeric = train.apply(pd.to_numeric, errors="coerce")
    fill = numeric.median(axis=0).fillna(0.0).to_numpy(dtype=float)
    x = numeric.to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, fill)

    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    xs = (x - center) / scale

    yv = y.to_numpy(dtype=float)
    y_mean = float(np.mean(yv))
    yc = yv - y_mean
    penalty = float(alpha) * np.eye(xs.shape[1])
    try:
        coef = np.linalg.solve(xs.T @ xs + penalty, xs.T @ yc)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xs.T @ xs + penalty) @ xs.T @ yc
    return RidgeFit(cols=list(cols), fill=fill, center=center, scale=scale, coef=coef, y_mean=y_mean)


def predict_context_deviation(fit: RidgeFit, row: pd.Series) -> float:
    x = pd.to_numeric(row[fit.cols], errors="coerce").to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, fit.fill)
    xs = (x - fit.center) / fit.scale
    value = float(xs @ fit.coef)
    return value if math.isfinite(value) else 0.0


def validation_origins(df: pd.DataFrame) -> list[pd.Timestamp]:
    origins = []
    for origin in pd.date_range(VALIDATION_START, VALIDATION_END, freq="MS"):
        if origin in df.index and origin + pd.DateOffset(months=1) in df.index:
            origins.append(origin)
    return origins


def evaluate_candidate(df, cols, mode, alpha, scale, target_col) -> dict | None:
    """Ungated validation MAE of the correction predicting the change target."""
    target = target_for_mode(df, target_col, mode)
    rows = []
    for origin in validation_origins(df):
        train_cutoff = origin - pd.DateOffset(months=1)
        fit = fit_ridge_context(df.loc[:train_cutoff], cols, target.loc[:train_cutoff], alpha)
        actual = target.loc[origin]
        if fit is None or pd.isna(actual):
            continue
        correction = scale * predict_context_deviation(fit, df.loc[origin])
        rows.append((float(actual), float(correction)))
    if len(rows) < 18:
        return None
    actuals = np.array([r[0] for r in rows])
    corrections = np.array([r[1] for r in rows])
    mae = float(np.mean(np.abs(actuals - corrections)))
    zero_mae = float(np.mean(np.abs(actuals)))
    delta = (mae - zero_mae) / zero_mae * 100.0 if zero_mae else math.nan
    return {
        "target_mode": mode, "alpha": float(alpha), "scale": float(scale),
        "n_validation": int(len(rows)), "validation_mae": mae,
        "zero_correction_mae": zero_mae, "validation_delta_pct_vs_zero": delta,
        "validation_beats_zero": bool(mae < zero_mae),
    }


def select_recipe(config: SeriesConfig, df: pd.DataFrame) -> dict:
    sets = {name: [c for c in cols if c in df.columns] for name, cols in config.feature_sets.items()}
    sets = {name: cols for name, cols in sets.items() if cols}
    sets["change_sel"] = change_selected_cols(df, config.target_col)
    candidates = []
    for set_name, cols in sets.items():
        for mode in TARGET_MODES:
            for alpha in ALPHAS:
                for scale in SCALES:
                    res = evaluate_candidate(df, cols, mode, alpha, scale, config.target_col)
                    if res is None:
                        continue
                    res["feature_set"] = set_name
                    res["features"] = cols
                    candidates.append(res)
    if not candidates:
        raise RuntimeError(f"No valid validation candidate for {config.key}.")
    best = min(candidates, key=lambda it: (it["validation_mae"], len(it["features"])))
    ranked = sorted(candidates, key=lambda it: (it["validation_mae"], len(it["features"])))[:10]
    return {
        "series": config.key,
        "validation_window": f"{VALIDATION_START.date()} to {VALIDATION_END.date()}",
        "change_sel_features": sets["change_sel"],
        "best": best, "top10": ranked,
    }


def fit_final_recipe(config, df, recipe, origin) -> RidgeFit | None:
    target = target_for_mode(df, config.target_col, recipe["target_mode"])
    cutoff = origin - pd.DateOffset(months=1)
    return fit_ridge_context(df.loc[:cutoff], recipe["features"], target.loc[:cutoff], recipe["alpha"])


def apply_overlay(config, df, recipe, model_name, source_file, output_file) -> dict:
    source_path = RESULTS / source_file
    output_path = RESULTS / output_file
    if not source_path.exists():
        return {"model": model_name, "source": source_file, "output": output_file, "status": "missing_source"}

    thr, vol = regime_threshold_and_vol(df, config.target_col)
    pred = pd.read_parquet(source_path).copy()
    pred["origin"] = pd.to_datetime(pred["origin"])
    pred["fc_date"] = pd.to_datetime(pred["fc_date"])
    base_labels = sorted(pred["model"].astype(str).unique().tolist())

    # Only correct if the selected recipe beat the zero-correction baseline on
    # the pre-2021 validation window; otherwise emit a no-op (identical to C0).
    applied = bool(recipe.get("validation_beats_zero", False))

    corrections: dict[pd.Timestamp, float] = {}
    gate_open: dict[pd.Timestamp, int] = {}
    for origin in sorted(pred["origin"].drop_duplicates()):
        origin = pd.Timestamp(origin)
        opened = int(origin in vol.index and pd.notna(vol.loc[origin]) and vol.loc[origin] > thr)
        gate_open[origin] = opened
        if not applied or origin not in df.index or not opened:
            corrections[origin] = 0.0
            continue
        fit = fit_final_recipe(config, df, recipe, origin)
        corrections[origin] = (0.0 if fit is None
                               else float(recipe["scale"]) * predict_context_deviation(fit, df.loc[origin]))

    pred["context_correction"] = pred["origin"].map(corrections).astype(float)
    pred["regime_open"] = pred["origin"].map(gate_open).astype(int)
    pred["y_pred_base"] = pred["y_pred"].astype(float)
    pred["y_pred"] = pred["y_pred_base"] + pred["context_correction"]
    pred["model"] = model_name
    pred["error"] = pred["y_true"].astype(float) - pred["y_pred"].astype(float)
    pred["abs_error"] = pred["error"].abs()
    pred.to_parquet(output_path, index=False)

    metrics = compute_metrics(pred, config.target_col, df)
    metrics_path = output_path.with_name(output_path.name.replace("_predictions.parquet", "_metrics.json"))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    nz = np.array([c for c in corrections.values()], dtype=float)
    n_open = int(sum(gate_open.values()))
    return {
        "model": model_name, "source": source_file, "source_model_labels": base_labels,
        "output": output_file, "metrics": metrics_path.name, "status": "ok",
        "n_origins": int(len(corrections)), "n_regime_open": n_open,
        "regime_threshold": round(thr, 4),
        "applied": applied, "validation_beats_zero": applied,
        "mean_abs_correction": float(np.mean(np.abs(nz))) if len(nz) else 0.0,
        "max_abs_correction": float(np.max(np.abs(nz))) if len(nz) else 0.0,
    }


def compute_metrics(pred: pd.DataFrame, target_col: str, feature_df: pd.DataFrame) -> dict:
    y = feature_df[target_col].astype(float)
    # Seasonal lag-12 MASE scale, matching the foundation-model convention, so
    # overlay MASE is comparable with the foundation metrics.
    train_vals = y.loc[:TRAIN_END].to_numpy(dtype=float)
    mase_scale = (float(np.mean(np.abs(train_vals[12:] - train_vals[:-12])))
                  if len(train_vals) > 12 else math.nan)
    out = {}
    for h in HORIZONS:
        sub = pred[pred["horizon"] == h]
        if sub.empty:
            continue
        err = sub["error"].astype(float)
        ae = err.abs()
        out[f"h{h}"] = {
            "MAE": round(float(ae.mean()), 4),
            "RMSE": round(float(np.sqrt(np.mean(err.to_numpy() ** 2))), 4),
            "MASE": round(float(ae.mean() / mase_scale), 4) if mase_scale and math.isfinite(mase_scale) else None,
            "n_evals": int(sub["origin"].nunique()),
            "n_rows": int(len(sub)),
        }
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "pre-2021 Ridge context recipe (incl. change-correlated signal "
                  "family) selected ungated, then applied with a fixed training-80pct "
                  "trailing-volatility regime gate on the 2021-2024 test window",
        "alphas": ALPHAS, "scales": SCALES, "target_modes": TARGET_MODES,
        "change_sel_k": CHANGE_SEL_K, "regime_quantile": REGIME_Q,
        "series": {}, "outputs": [],
    }
    for config in SERIES.values():
        df = load_feature_frame(config.feature_file)
        selection = select_recipe(config, df)
        best = selection["best"]
        report["series"][config.key] = selection
        print(f"{config.key}: {best['feature_set']} {best['target_mode']} "
              f"alpha={best['alpha']} scale={best['scale']} "
              f"feats={best['features']} val_delta={best['validation_delta_pct_vs_zero']:+.2f}%")
        for model_name, (src, out) in config.overlays.items():
            result = apply_overlay(config, df, best, model_name, src, out)
            report["outputs"].append(result)
            extra = (f" gate_open={result['n_regime_open']}/{result['n_origins']}"
                     if result["status"] == "ok" else "")
            print(f"  {model_name}: {result['status']} -> {out}{extra}")

    out = RESULTS / "regime_context_overlay_selection.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selection report: {out}")


if __name__ == "__main__":
    main()
