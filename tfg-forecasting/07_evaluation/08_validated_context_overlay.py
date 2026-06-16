"""
08_validated_context_overlay.py
--------------------------------
Causal, low-risk contextual correction experiment for the thesis-critical
foundation forecasts.

This script does not rerun the foundation models and does not overwrite any
existing prediction file. It:

1. Selects a small Ridge context-correction recipe using only pre-2021
   validation origins.
2. Applies the frozen recipe to existing C0 prediction Parquets.
3. Writes separate C1_validated prediction and metric artifacts.

The goal is to test whether noisy context bundles can be improved by
pre-period feature-family selection and stronger shrinkage, without selecting
features or horizons from the 2021-2024 thesis test window.
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
MIN_TRAIN_ROWS = 72
ALPHAS = [0.1, 1.0, 10.0, 100.0]
SCALES = [0.25, 0.5, 1.0]
TARGET_MODES = ["resid_12ma", "next_diff"]


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
            "epu_all": ["epu_europe_ma3", "epu_europe_log", "epu_europe_lag1"],
            "energy": ["brent_ma3", "brent_lag1", "ttf_ma3", "ttf_lag1"],
            "policy_energy": ["dfr", "dfr_lag3", "brent_ma3", "ttf_ma3"],
            "macro_small": ["dfr", "dfr_lag3", "epu_europe_ma3", "brent_ma3"],
        },
        overlays={
            "timesfm_C1_validated": ("timesfm_C0_predictions.parquet", "timesfm_C1_validated_predictions.parquet"),
            "chronos2_C1_validated": ("chronos2_C0_predictions.parquet", "chronos2_C1_validated_predictions.parquet"),
        },
    ),
    "global": SeriesConfig(
        key="global",
        target_col="cpi_global_rate",
        feature_file=DATA / "features_c1_global_institutional.parquet",
        feature_sets={
            "top3": ["imf_comm_ma3", "brent_log_ma3", "gscpi_ma3"],
            "commodities": ["imf_comm_ma3", "imf_comm_lag1", "brent_log_ma3", "brent_log_lag1"],
            "supply": ["gscpi_ma3", "gscpi_lag1"],
            "risk_policy": ["gepu_ma3", "fedfunds_ma3", "vix_ma3"],
            "broad_small": ["imf_comm_ma3", "brent_log_ma3", "gscpi_ma3", "fedfunds_ma3", "gepu_ma3"],
        },
        overlays={
            "timesfm_C1_validated_global": ("timesfm_C0_global_predictions.parquet", "timesfm_C1_validated_global_predictions.parquet"),
            "chronos2_C1_validated_global": ("chronos2_C0_global_predictions.parquet", "chronos2_C1_validated_global_predictions.parquet"),
        },
    ),
    "europe": SeriesConfig(
        key="europe",
        target_col="hicp_index",
        feature_file=DATA / "features_c1_europe.parquet",
        feature_sets={
            "inst": [
                "epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3",
                "dfr", "dfr_ma3", "ttf_ma3", "breakeven_5y_lag1",
            ],
            "energy": ["brent_ma3", "ttf_ma3"],
            "macro": ["epu_europe_ma3", "esi_eurozone", "dfr_ma3", "breakeven_5y_lag1"],
            "mcp": ["bce_shock_score", "bce_tone_numeric", "bce_cumstance", "gdelt_tone_ma6", "signal_available"],
            "full": [
                "epu_europe_ma3", "brent_ma3", "esi_eurozone", "eurusd_ma3",
                "dfr", "dfr_ma3", "ttf_ma3", "breakeven_5y_lag1",
                "bce_shock_score", "bce_tone_numeric", "bce_cumstance",
                "gdelt_tone_ma6", "signal_available",
            ],
        },
        overlays={
            "timesfm_C1_validated_europe": ("timesfm_C0_europe_predictions.parquet", "timesfm_C1_validated_europe_predictions.parquet"),
        },
    ),
}


def load_feature_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def target_for_mode(df: pd.DataFrame, target_col: str, mode: str) -> pd.Series:
    y = df[target_col].astype(float)
    next_diff = y.shift(-1) - y
    if mode == "next_diff":
        return next_diff
    if mode == "resid_12ma":
        expected = y.diff().rolling(12, min_periods=6).mean()
        return next_diff - expected
    raise ValueError(f"Unknown target mode: {mode}")


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
    if not math.isfinite(value):
        return 0.0
    return value


def validation_origins(df: pd.DataFrame) -> list[pd.Timestamp]:
    origins = []
    for origin in pd.date_range(VALIDATION_START, VALIDATION_END, freq="MS"):
        if origin in df.index and origin + pd.DateOffset(months=1) in df.index:
            origins.append(origin)
    return origins


def evaluate_candidate(
    df: pd.DataFrame,
    cols: list[str],
    mode: str,
    alpha: float,
    scale: float,
    target_col: str,
) -> dict | None:
    target = target_for_mode(df, target_col, mode)
    rows = []
    for origin in validation_origins(df):
        train_cutoff = origin - pd.DateOffset(months=1)
        train_df = df.loc[:train_cutoff]
        train_target = target.loc[:train_cutoff]
        fit = fit_ridge_context(train_df, cols, train_target, alpha)
        actual = target.loc[origin]
        if fit is None or pd.isna(actual):
            continue
        correction = scale * predict_context_deviation(fit, df.loc[origin])
        rows.append((float(actual), float(correction)))

    if len(rows) < 18:
        return None

    actuals = np.array([r[0] for r in rows], dtype=float)
    corrections = np.array([r[1] for r in rows], dtype=float)
    mae = float(np.mean(np.abs(actuals - corrections)))
    zero_mae = float(np.mean(np.abs(actuals)))
    delta = (mae - zero_mae) / zero_mae * 100.0 if zero_mae else math.nan
    return {
        "feature_set": None,
        "target_mode": mode,
        "alpha": float(alpha),
        "scale": float(scale),
        "n_validation": int(len(rows)),
        "validation_mae": mae,
        "zero_correction_mae": zero_mae,
        "validation_delta_pct_vs_zero": delta,
        "validation_beats_zero": bool(mae < zero_mae),
    }


def select_recipe(config: SeriesConfig, df: pd.DataFrame) -> dict:
    available_sets = {
        name: [col for col in cols if col in df.columns]
        for name, cols in config.feature_sets.items()
    }
    available_sets = {name: cols for name, cols in available_sets.items() if cols}
    candidates = []
    for set_name, cols in available_sets.items():
        for mode in TARGET_MODES:
            for alpha in ALPHAS:
                for scale in SCALES:
                    result = evaluate_candidate(df, cols, mode, alpha, scale, config.target_col)
                    if result is None:
                        continue
                    result["feature_set"] = set_name
                    result["features"] = cols
                    candidates.append(result)

    if not candidates:
        raise RuntimeError(f"No valid validation candidate for {config.key}.")

    best = min(candidates, key=lambda item: (item["validation_mae"], len(item["features"])))
    ranked = sorted(candidates, key=lambda item: (item["validation_mae"], len(item["features"])))[:10]
    return {
        "series": config.key,
        "validation_window": f"{VALIDATION_START.date()} to {VALIDATION_END.date()}",
        "best": best,
        "top10": ranked,
    }


def fit_final_recipe(config: SeriesConfig, df: pd.DataFrame, recipe: dict, origin: pd.Timestamp) -> RidgeFit | None:
    target = target_for_mode(df, config.target_col, recipe["target_mode"])
    train_cutoff = origin - pd.DateOffset(months=1)
    train_df = df.loc[:train_cutoff]
    train_target = target.loc[:train_cutoff]
    return fit_ridge_context(train_df, recipe["features"], train_target, recipe["alpha"])


def apply_overlay(config: SeriesConfig, df: pd.DataFrame, recipe: dict, model_name: str, source_file: str, output_file: str) -> dict:
    source_path = RESULTS / source_file
    output_path = RESULTS / output_file
    if not source_path.exists():
        return {
            "model": model_name,
            "source": source_file,
            "output": output_file,
            "status": "missing_source",
        }

    pred = pd.read_parquet(source_path).copy()
    pred["origin"] = pd.to_datetime(pred["origin"])
    pred["fc_date"] = pd.to_datetime(pred["fc_date"])
    base_labels = sorted(pred["model"].astype(str).unique().tolist())
    base_label = base_labels[0] if len(base_labels) == 1 else None

    # Only apply a correction if the selected recipe actually beat the
    # zero-correction baseline on the pre-2021 validation window. Otherwise the
    # "validated" overlay emits a no-op (identical to C0): a method that cannot
    # validate a correction must not invent one.
    applied = bool(recipe.get("validation_beats_zero", False))

    corrections: dict[pd.Timestamp, float] = {}
    for origin in sorted(pred["origin"].drop_duplicates()):
        origin = pd.Timestamp(origin)
        if not applied or origin not in df.index:
            corrections[origin] = 0.0
            continue
        fit = fit_final_recipe(config, df, recipe, origin)
        if fit is None:
            corrections[origin] = 0.0
            continue
        corrections[origin] = float(recipe["scale"]) * predict_context_deviation(fit, df.loc[origin])

    pred["context_correction"] = pred["origin"].map(corrections).astype(float)
    pred["y_pred_base"] = pred["y_pred"].astype(float)
    pred["y_pred"] = pred["y_pred_base"] + pred["context_correction"]
    pred["model"] = model_name
    pred["error"] = pred["y_true"].astype(float) - pred["y_pred"].astype(float)
    pred["abs_error"] = pred["error"].abs()
    pred.to_parquet(output_path, index=False)

    metrics = compute_metrics(pred, config.target_col, df)
    metrics_path = output_path.with_name(output_path.name.replace("_predictions.parquet", "_metrics.json"))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    nonzero = np.array(list(corrections.values()), dtype=float)
    return {
        "model": model_name,
        "source": source_file,
        "source_model_labels": base_labels,
        "output": output_file,
        "metrics": metrics_path.name,
        "status": "ok",
        "n_origins": int(len(corrections)),
        "applied": applied,
        "validation_beats_zero": applied,
        "mean_abs_correction": float(np.mean(np.abs(nonzero))) if len(nonzero) else 0.0,
        "max_abs_correction": float(np.max(np.abs(nonzero))) if len(nonzero) else 0.0,
        "base_label": base_label,
    }


def compute_metrics(pred: pd.DataFrame, target_col: str, feature_df: pd.DataFrame) -> dict:
    y = feature_df[target_col].astype(float)
    train_y = y.loc[:pd.Timestamp("2020-12-01")]
    # Seasonal lag-12 MASE scale, matching the foundation-model convention
    # (06_models_foundation/*: mean |y[t] - y[t-12]| over the training series),
    # so overlay MASE is comparable with the foundation metrics.
    train_vals = train_y.to_numpy(dtype=float)
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
    selection_report = {
        "method": "pre-2021 Ridge context overlay selected before applying to 2021-2024 test forecasts",
        "alphas": ALPHAS,
        "scales": SCALES,
        "target_modes": TARGET_MODES,
        "series": {},
        "outputs": [],
    }

    for config in SERIES.values():
        df = load_feature_frame(config.feature_file)
        selection = select_recipe(config, df)
        best = selection["best"]
        selection_report["series"][config.key] = selection
        print(
            f"{config.key}: {best['feature_set']} {best['target_mode']} "
            f"alpha={best['alpha']} scale={best['scale']} "
            f"validation_delta={best['validation_delta_pct_vs_zero']:+.2f}%"
        )

        for model_name, (source_file, output_file) in config.overlays.items():
            result = apply_overlay(config, df, best, model_name, source_file, output_file)
            selection_report["outputs"].append(result)
            print(f"  {model_name}: {result['status']} -> {output_file}")

    out = RESULTS / "validated_context_overlay_selection.json"
    out.write_text(json.dumps(selection_report, indent=2), encoding="utf-8")
    print(f"Selection report: {out}")


if __name__ == "__main__":
    main()
