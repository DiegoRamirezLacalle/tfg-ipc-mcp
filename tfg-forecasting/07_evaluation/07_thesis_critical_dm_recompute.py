"""
07_thesis_critical_dm_recompute.py
----------------------------------
Independent recomputation of the thesis-critical forecast comparisons from
current prediction Parquets only.

This script intentionally does not read the existing Diebold-Mariano JSON,
CSV, or Markdown summaries. It recomputes errors from y_true/y_pred, aligns
forecast rows by (origin, fc_date, horizon, step), and reports both:

  * path-level: all rows with horizon == h
  * endpoint-only: rows with horizon == h and step == h

DM test: two-sided Harvey-Leybourne-Newbold adjusted Diebold-Mariano test,
with HAC lags h-1 and Student-t p-value (df=n-1). Both absolute-error and
squared-error loss are reported.

Outputs:
  08_results/thesis_critical_dm_recompute.csv
  08_results/thesis_critical_dm_recompute.md
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "08_results"
HORIZONS = [1, 3, 6, 12]
ALIGN_KEYS = ["origin", "fc_date", "horizon", "step"]
REQUIRED_COLUMNS = {"origin", "fc_date", "step", "horizon", "model", "y_true", "y_pred"}
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260616


@dataclass(frozen=True)
class ModelSpec:
    key: str
    path: Path
    model: str


@dataclass(frozen=True)
class Comparison:
    series: str
    label: str
    model_a: str
    model_b: str
    spec_a: str
    spec_b: str


SPECS = {
    "spain_timesfm_c0": ModelSpec("spain_timesfm_c0", RESULTS / "timesfm_C0_predictions.parquet", "timesfm_C0"),
    "spain_timesfm_c1_inst": ModelSpec("spain_timesfm_c1_inst", RESULTS / "timesfm_C1_inst_predictions.parquet", "timesfm_C1_inst"),
    "spain_timesfm_c1_validated": ModelSpec("spain_timesfm_c1_validated", RESULTS / "timesfm_C1_validated_predictions.parquet", "timesfm_C1_validated"),
    "spain_chronos_c0": ModelSpec("spain_chronos_c0", RESULTS / "chronos2_C0_predictions.parquet", "chronos2_C0"),
    "spain_chronos_c1_inst": ModelSpec("spain_chronos_c1_inst", RESULTS / "chronos2_C1_inst_predictions.parquet", "chronos2_C1_inst"),
    "spain_chronos_c1_fwd": ModelSpec("spain_chronos_c1_fwd", RESULTS / "chronos2_C1_fwd_spain_predictions.parquet", "chronos2_C1_fwd_spain"),
    "spain_chronos_c1_validated": ModelSpec("spain_chronos_c1_validated", RESULTS / "chronos2_C1_validated_predictions.parquet", "chronos2_C1_validated"),
    "spain_timesfm_c1_regime": ModelSpec("spain_timesfm_c1_regime", RESULTS / "timesfm_C1_regime_predictions.parquet", "timesfm_C1_regime"),
    "spain_chronos_c1_regime": ModelSpec("spain_chronos_c1_regime", RESULTS / "chronos2_C1_regime_predictions.parquet", "chronos2_C1_regime"),
    "global_chronos_c0": ModelSpec("global_chronos_c0", RESULTS / "chronos2_C0_global_predictions.parquet", "chronos2_C0_global"),
    "global_chronos_c1_inst": ModelSpec("global_chronos_c1_inst", RESULTS / "chronos2_C1_inst_global_predictions.parquet", "chronos2_C1_inst_global"),
    "global_chronos_c1_validated": ModelSpec("global_chronos_c1_validated", RESULTS / "chronos2_C1_validated_global_predictions.parquet", "chronos2_C1_validated_global"),
    "global_timesfm_c0": ModelSpec("global_timesfm_c0", RESULTS / "timesfm_C0_global_predictions.parquet", "timesfm_C0_global"),
    "global_timesfm_c1_inst": ModelSpec("global_timesfm_c1_inst", RESULTS / "timesfm_C1_inst_global_predictions.parquet", "timesfm_C1_inst_global"),
    "global_timesfm_c1_validated": ModelSpec("global_timesfm_c1_validated", RESULTS / "timesfm_C1_validated_global_predictions.parquet", "timesfm_C1_validated_global"),
    "global_chronos_c1_fwd": ModelSpec("global_chronos_c1_fwd", RESULTS / "chronos2_C1_fwd_global_predictions.parquet", "chronos2_C1_fwd_global"),
    "global_chronos_c1_regime": ModelSpec("global_chronos_c1_regime", RESULTS / "chronos2_C1_regime_global_predictions.parquet", "chronos2_C1_regime_global"),
    "global_timesfm_c1_regime": ModelSpec("global_timesfm_c1_regime", RESULTS / "timesfm_C1_regime_global_predictions.parquet", "timesfm_C1_regime_global"),
    "global_arima": ModelSpec("global_arima", RESULTS / "rolling_predictions_global.parquet", "arima"),
    "global_auto_arima": ModelSpec("global_auto_arima", RESULTS / "autoarima_global_predictions.parquet", "auto_arima"),
    "europe_timesfm_c0": ModelSpec("europe_timesfm_c0", RESULTS / "timesfm_C0_europe_predictions.parquet", "timesfm_C0_europe"),
    "europe_timesfm_c1_full": ModelSpec("europe_timesfm_c1_full", RESULTS / "timesfm_C1_full_europe_predictions.parquet", "timesfm_C1_full_europe"),
    "europe_timesfm_c1_validated": ModelSpec("europe_timesfm_c1_validated", RESULTS / "timesfm_C1_validated_europe_predictions.parquet", "timesfm_C1_validated_europe"),
    "europe_chronos_c0": ModelSpec("europe_chronos_c0", RESULTS / "chronos2_C0_europe_predictions.parquet", "chronos2_C0_europe"),
    "europe_chronos_c1_inst": ModelSpec("europe_chronos_c1_inst", RESULTS / "chronos2_C1_inst_europe_predictions.parquet", "chronos2_C1_inst_europe"),
    "europe_chronos_c1_fwd": ModelSpec("europe_chronos_c1_fwd", RESULTS / "chronos2_C1_fwd_europe_predictions.parquet", "chronos2_C1_fwd_europe"),
    "europe_timesfm_c1_regime": ModelSpec("europe_timesfm_c1_regime", RESULTS / "timesfm_C1_regime_europe_predictions.parquet", "timesfm_C1_regime_europe"),
    "europe_chronos_c1_regime": ModelSpec("europe_chronos_c1_regime", RESULTS / "chronos2_C1_regime_europe_predictions.parquet", "chronos2_C1_regime_europe"),
    "europe_sarima": ModelSpec("europe_sarima", RESULTS / "rolling_predictions_europe.parquet", "sarima"),
    "europe_auto_arima": ModelSpec("europe_auto_arima", RESULTS / "autoarima_europe_predictions.parquet", "auto_arima"),
}

COMPARISONS = [
    Comparison("Spain", "TimesFM context", "timesfm_C0", "timesfm_C1_inst", "spain_timesfm_c0", "spain_timesfm_c1_inst"),
    Comparison("Spain", "TimesFM validated context", "timesfm_C0", "timesfm_C1_validated", "spain_timesfm_c0", "spain_timesfm_c1_validated"),
    Comparison("Spain", "Chronos-2 context", "chronos2_C0", "chronos2_C1_inst", "spain_chronos_c0", "spain_chronos_c1_inst"),
    Comparison("Spain", "Chronos-2 validated context", "chronos2_C0", "chronos2_C1_validated", "spain_chronos_c0", "spain_chronos_c1_validated"),
    Comparison("Spain", "TimesFM regime context", "timesfm_C0", "timesfm_C1_regime", "spain_timesfm_c0", "spain_timesfm_c1_regime"),
    Comparison("Spain", "Chronos-2 regime context", "chronos2_C0", "chronos2_C1_regime", "spain_chronos_c0", "spain_chronos_c1_regime"),
    Comparison("Spain", "Chronos-2 forward-path context", "chronos2_C0", "chronos2_C1_fwd_spain", "spain_chronos_c0", "spain_chronos_c1_fwd"),
    Comparison("Spain", "Chronos-2 forward vs flat-hold", "chronos2_C1_inst", "chronos2_C1_fwd_spain", "spain_chronos_c1_inst", "spain_chronos_c1_fwd"),
    Comparison("Global", "Chronos-2 context", "chronos2_C0_global", "chronos2_C1_inst_global", "global_chronos_c0", "global_chronos_c1_inst"),
    Comparison("Global", "Chronos-2 validated context", "chronos2_C0_global", "chronos2_C1_validated_global", "global_chronos_c0", "global_chronos_c1_validated"),
    Comparison("Global", "TimesFM context", "timesfm_C0_global", "timesfm_C1_inst_global", "global_timesfm_c0", "global_timesfm_c1_inst"),
    Comparison("Global", "TimesFM validated context", "timesfm_C0_global", "timesfm_C1_validated_global", "global_timesfm_c0", "global_timesfm_c1_validated"),
    Comparison("Global", "Chronos-2 forward-path context", "chronos2_C0_global", "chronos2_C1_fwd_global", "global_chronos_c0", "global_chronos_c1_fwd"),
    Comparison("Global", "Chronos-2 forward vs flat-hold", "chronos2_C1_inst_global", "chronos2_C1_fwd_global", "global_chronos_c1_inst", "global_chronos_c1_fwd"),
    Comparison("Global", "Chronos-2 regime context", "chronos2_C0_global", "chronos2_C1_regime_global", "global_chronos_c0", "global_chronos_c1_regime"),
    Comparison("Global", "TimesFM regime context", "timesfm_C0_global", "timesfm_C1_regime_global", "global_timesfm_c0", "global_timesfm_c1_regime"),
    Comparison("Global", "Chronos-2 vs ARIMA", "chronos2_C1_inst_global", "arima", "global_chronos_c1_inst", "global_arima"),
    Comparison("Global", "Chronos-2 vs AutoARIMA", "chronos2_C1_inst_global", "auto_arima", "global_chronos_c1_inst", "global_auto_arima"),
    Comparison("Global", "Chronos-2 validated vs ARIMA", "chronos2_C1_validated_global", "arima", "global_chronos_c1_validated", "global_arima"),
    Comparison("Global", "Chronos-2 validated vs AutoARIMA", "chronos2_C1_validated_global", "auto_arima", "global_chronos_c1_validated", "global_auto_arima"),
    Comparison("Global", "TimesFM vs ARIMA", "timesfm_C1_inst_global", "arima", "global_timesfm_c1_inst", "global_arima"),
    Comparison("Global", "TimesFM vs AutoARIMA", "timesfm_C1_inst_global", "auto_arima", "global_timesfm_c1_inst", "global_auto_arima"),
    Comparison("Global", "TimesFM validated vs ARIMA", "timesfm_C1_validated_global", "arima", "global_timesfm_c1_validated", "global_arima"),
    Comparison("Global", "TimesFM validated vs AutoARIMA", "timesfm_C1_validated_global", "auto_arima", "global_timesfm_c1_validated", "global_auto_arima"),
    Comparison("Europe", "TimesFM context", "timesfm_C0_europe", "timesfm_C1_full_europe", "europe_timesfm_c0", "europe_timesfm_c1_full"),
    Comparison("Europe", "TimesFM validated context", "timesfm_C0_europe", "timesfm_C1_validated_europe", "europe_timesfm_c0", "europe_timesfm_c1_validated"),
    Comparison("Europe", "TimesFM regime context", "timesfm_C0_europe", "timesfm_C1_regime_europe", "europe_timesfm_c0", "europe_timesfm_c1_regime"),
    Comparison("Europe", "Chronos-2 regime context", "chronos2_C0_europe", "chronos2_C1_regime_europe", "europe_chronos_c0", "europe_chronos_c1_regime"),
    Comparison("Europe", "Chronos-2 forward-path context", "chronos2_C0_europe", "chronos2_C1_fwd_europe", "europe_chronos_c0", "europe_chronos_c1_fwd"),
    Comparison("Europe", "Chronos-2 forward vs flat-hold", "chronos2_C1_inst_europe", "chronos2_C1_fwd_europe", "europe_chronos_c1_inst", "europe_chronos_c1_fwd"),
    Comparison("Europe", "TimesFM vs SARIMA", "timesfm_C1_full_europe", "sarima", "europe_timesfm_c1_full", "europe_sarima"),
    Comparison("Europe", "TimesFM vs AutoARIMA", "timesfm_C1_full_europe", "auto_arima", "europe_timesfm_c1_full", "europe_auto_arima"),
    Comparison("Europe", "TimesFM validated vs SARIMA", "timesfm_C1_validated_europe", "sarima", "europe_timesfm_c1_validated", "europe_sarima"),
    Comparison("Europe", "TimesFM validated vs AutoARIMA", "timesfm_C1_validated_europe", "auto_arima", "europe_timesfm_c1_validated", "europe_auto_arima"),
]


def significance(p_value: float) -> str:
    if not math.isfinite(p_value):
        return "NA"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return "ns"


def load_predictions(spec: ModelSpec) -> tuple[pd.DataFrame | None, dict]:
    status = {
        "key": spec.key,
        "file": spec.path.name,
        "expected_model": spec.model,
        "exists": spec.path.exists(),
        "labels": [],
        "rows": 0,
        "max_error_diff": None,
        "max_abs_error_diff": None,
        "missing_columns": [],
    }
    if not spec.path.exists():
        return None, status

    df = pd.read_parquet(spec.path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    status["missing_columns"] = missing
    if missing:
        return None, status

    df = df.copy()
    df["origin"] = pd.to_datetime(df["origin"])
    df["fc_date"] = pd.to_datetime(df["fc_date"])
    labels = sorted(df["model"].astype(str).unique().tolist())
    status["labels"] = labels
    if spec.model not in labels:
        return None, status

    df = df[df["model"].astype(str) == spec.model].copy()
    df["error_recomputed"] = df["y_true"].astype(float) - df["y_pred"].astype(float)
    df["abs_error_recomputed"] = df["error_recomputed"].abs()
    status["rows"] = int(len(df))
    if "error" in df.columns:
        status["max_error_diff"] = float((df["error"].astype(float) - df["error_recomputed"]).abs().max())
    if "abs_error" in df.columns:
        status["max_abs_error_diff"] = float((df["abs_error"].astype(float) - df["abs_error_recomputed"]).abs().max())
    return df, status


def dm_hln(error_a: np.ndarray, error_b: np.ndarray, horizon: int, loss: str) -> tuple[float, float]:
    if loss == "abs":
        d = np.abs(error_a) - np.abs(error_b)
    elif loss == "sq":
        d = error_a**2 - error_b**2
    else:
        raise ValueError(f"Unknown loss {loss!r}.")

    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 4:
        return math.nan, math.nan

    d_bar = float(np.mean(d))
    centered = d - d_bar
    long_run = float(np.dot(centered, centered) / n)
    for lag in range(1, horizon):
        if n <= lag:
            break
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2.0 * gamma

    var_d_bar = long_run / n
    if not math.isfinite(var_d_bar) or var_d_bar <= 0:
        return math.nan, math.nan

    dm = d_bar / math.sqrt(var_d_bar)
    hln_factor = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if hln_factor <= 0:
        return math.nan, math.nan

    dm_adj = dm * math.sqrt(hln_factor)
    p_value = 2.0 * stats.t.sf(abs(dm_adj), df=n - 1)
    return float(dm_adj), float(p_value)


def bootstrap_delta_ci(aligned: pd.DataFrame, reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    """Origin-cluster paired bootstrap CI for delta MAE percent: (B - A) / A."""
    grouped = (
        aligned.groupby("origin", sort=False)
        .agg(abs_a=("abs_error_a", "sum"), abs_b=("abs_error_b", "sum"), n=("origin", "size"))
        .reset_index(drop=True)
    )
    if len(grouped) < 4:
        return math.nan, math.nan

    abs_a = grouped["abs_a"].to_numpy(dtype=float)
    abs_b = grouped["abs_b"].to_numpy(dtype=float)
    counts = grouped["n"].to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(grouped), size=(reps, len(grouped)))
    sampled_a = abs_a[indices].sum(axis=1)
    sampled_b = abs_b[indices].sum(axis=1)
    sampled_n = counts[indices].sum(axis=1)
    mae_a = sampled_a / sampled_n
    mae_b = sampled_b / sampled_n
    valid = mae_a > 0
    if not np.any(valid):
        return math.nan, math.nan
    values = (mae_b[valid] - mae_a[valid]) / mae_a[valid] * 100.0
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_delta_ci_slow(aligned: pd.DataFrame, reps: int = 200) -> tuple[float, float]:
    """Readable reference implementation kept for debugging, not used by default."""
    origins = aligned["origin"].drop_duplicates().to_numpy()
    if len(origins) < 4:
        return math.nan, math.nan
    by_origin = {origin: rows for origin, rows in aligned.groupby("origin", sort=False)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(reps):
        sample = rng.choice(origins, size=len(origins), replace=True)
        pieces = [by_origin[origin] for origin in sample]
        boot = pd.concat(pieces, ignore_index=True)
        mae_a = float(boot["abs_error_a"].mean())
        mae_b = float(boot["abs_error_b"].mean())
        if mae_a > 0:
            values.append((mae_b - mae_a) / mae_a * 100.0)

    if not values:
        return math.nan, math.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def align_pair(df_a: pd.DataFrame, df_b: pd.DataFrame, horizon: int, level: str) -> pd.DataFrame:
    a = df_a[df_a["horizon"] == horizon].copy()
    b = df_b[df_b["horizon"] == horizon].copy()
    if level == "endpoint":
        a = a[a["step"] == horizon]
        b = b[b["step"] == horizon]
    elif level != "path":
        raise ValueError(f"Unknown level {level!r}.")

    cols = ALIGN_KEYS + ["error_recomputed", "abs_error_recomputed"]
    aligned = a[cols].merge(b[cols], on=ALIGN_KEYS, suffixes=("_a", "_b"))
    return aligned.rename(
        columns={
            "error_recomputed_a": "error_a",
            "error_recomputed_b": "error_b",
            "abs_error_recomputed_a": "abs_error_a",
            "abs_error_recomputed_b": "abs_error_b",
        }
    ).sort_values(ALIGN_KEYS)


def compute_rows(loaded: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for comparison in COMPARISONS:
        df_a = loaded.get(comparison.spec_a)
        df_b = loaded.get(comparison.spec_b)
        if df_a is None or df_b is None:
            continue
        for level in ["path", "endpoint"]:
            for horizon in HORIZONS:
                aligned = align_pair(df_a, df_b, horizon, level)
                n = len(aligned)
                if n == 0:
                    continue
                error_a = aligned["error_a"].to_numpy(dtype=float)
                error_b = aligned["error_b"].to_numpy(dtype=float)
                mae_a = float(np.mean(np.abs(error_a)))
                mae_b = float(np.mean(np.abs(error_b)))
                delta_pct = (mae_b - mae_a) / mae_a * 100.0 if mae_a else math.nan
                better = comparison.model_a if mae_a < mae_b else comparison.model_b
                dm_abs, p_abs = dm_hln(error_a, error_b, horizon, "abs")
                dm_sq, p_sq = dm_hln(error_a, error_b, horizon, "sq")
                ci_lo, ci_hi = bootstrap_delta_ci(aligned)
                rows.append(
                    {
                        "level": level,
                        "series": comparison.series,
                        "comparison": comparison.label,
                        "model_a": comparison.model_a,
                        "model_b": comparison.model_b,
                        "horizon": horizon,
                        "n": n,
                        "origins": int(aligned["origin"].nunique()),
                        "mae_a": mae_a,
                        "mae_b": mae_b,
                        "delta_mae_pct_b_vs_a": delta_pct,
                        "delta_mae_pct_ci95_lo": ci_lo,
                        "delta_mae_pct_ci95_hi": ci_hi,
                        "better_by_mae": better,
                        "dm_abs": dm_abs,
                        "p_abs": p_abs,
                        "sig_abs": significance(p_abs),
                        "dm_sq": dm_sq,
                        "p_sq": p_sq,
                        "sig_sq": significance(p_sq),
                    }
                )
    return rows


def write_csv(rows: list[dict]) -> Path:
    out = RESULTS / "thesis_critical_dm_recompute.csv"
    columns = [
        "level", "series", "comparison", "model_a", "model_b", "horizon", "n", "origins",
        "mae_a", "mae_b", "delta_mae_pct_b_vs_a", "delta_mae_pct_ci95_lo",
        "delta_mae_pct_ci95_hi", "better_by_mae", "dm_abs", "p_abs", "sig_abs",
        "dm_sq", "p_sq", "sig_sq",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_markdown(rows: list[dict], statuses: list[dict]) -> Path:
    out = RESULTS / "thesis_critical_dm_recompute.md"
    df = pd.DataFrame(rows)
    lines = [
        "# Thesis-critical DM recomputation",
        "",
        "Independent recomputation from prediction Parquets only. Existing DM JSON/CSV/MD summaries are not read.",
        "",
        "DM method: abs-error and squared-error loss, HAC lags h-1, Harvey-Leybourne-Newbold adjustment, two-sided Student-t p-value with df=n-1.",
        "Delta is model B versus model A, so negative means model B has lower MAE.",
        "",
        "## File Checks",
        "",
        "| key | file | expected model | labels | rows | max error diff | max abs diff | status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for status in statuses:
        if not status["exists"]:
            state = "missing file"
        elif status["missing_columns"]:
            state = "missing columns: " + ", ".join(status["missing_columns"])
        elif status["expected_model"] not in status["labels"]:
            state = "missing model label"
        else:
            state = "ok"
        lines.append(
            "| {key} | {file} | {expected_model} | {labels} | {rows} | {err} | {abs_err} | {state} |".format(
                key=status["key"],
                file=status["file"],
                expected_model=status["expected_model"],
                labels=", ".join(status["labels"]),
                rows=status["rows"],
                err=fmt(status["max_error_diff"], 3) if status["max_error_diff"] is not None else "NA",
                abs_err=fmt(status["max_abs_error_diff"], 3) if status["max_abs_error_diff"] is not None else "NA",
                state=state,
            )
        )

    for level in ["path", "endpoint"]:
        lines.extend(
            [
                "",
                f"## {level.title()} Level",
                "",
                "| series | comparison | h | n | origins | MAE A | MAE B | delta % [95% boot CI] | better | p abs | sig | p sq | sig sq |",
                "|---|---|---:|---:|---:|---:|---:|---|---|---:|---|---:|---|",
            ]
        )
        sub = df[df["level"] == level].copy()
        for _, row in sub.iterrows():
            ci = f"[{fmt(row['delta_mae_pct_ci95_lo'], 1)}, {fmt(row['delta_mae_pct_ci95_hi'], 1)}]"
            lines.append(
                "| {series} | {comp} | {h} | {n} | {origins} | {mae_a} | {mae_b} | {delta}% {ci} | {better} | {p_abs} | {sig_abs} | {p_sq} | {sig_sq} |".format(
                    series=row["series"],
                    comp=row["comparison"],
                    h=int(row["horizon"]),
                    n=int(row["n"]),
                    origins=int(row["origins"]),
                    mae_a=fmt(row["mae_a"], 4),
                    mae_b=fmt(row["mae_b"], 4),
                    delta=fmt(row["delta_mae_pct_b_vs_a"], 1),
                    ci=ci,
                    better=row["better_by_mae"],
                    p_abs=fmt(row["p_abs"], 4),
                    sig_abs=row["sig_abs"],
                    p_sq=fmt(row["p_sq"], 4),
                    sig_sq=row["sig_sq"],
                )
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Path-level rows pool every step in each forecast path for the requested horizon.",
            "- Endpoint-only rows use only step == horizon, so h12 has fewer paired observations.",
            "- Bootstrap intervals are paired by forecast origin to preserve within-origin dependence.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    loaded: dict[str, pd.DataFrame] = {}
    statuses = []
    for spec in SPECS.values():
        df, status = load_predictions(spec)
        statuses.append(status)
        if df is not None:
            loaded[spec.key] = df

    rows = compute_rows(loaded)
    csv_path = write_csv(rows)
    md_path = write_markdown(rows, statuses)
    print(f"Rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")

    sig = [r for r in rows if r["sig_abs"] in {"*", "**"}]
    print("Abs-loss DM p<0.10:")
    for row in sig:
        direction = "B better" if row["mae_b"] < row["mae_a"] else "A better"
        print(
            f"  {row['level']:8s} {row['series']:6s} h={row['horizon']:2d} "
            f"{row['comparison']}: p={row['p_abs']:.4f} {row['sig_abs']} "
            f"delta={row['delta_mae_pct_b_vs_a']:+.1f}% ({direction})"
        )


if __name__ == "__main__":
    main()
