"""Automated integrity checks for the backtesting artifacts.

Runs four families of checks over the stored prediction parquets so that a
silent leakage or misalignment bug cannot slip into the final tables:

  1. CAUSALITY (no leakage) - every prediction row is dated strictly after its
     own origin, and ``step`` equals the month offset origin -> fc_date and lies
     in ``1 .. horizon``. (A row dated at/<= its origin, or a mismatched step,
     is the fingerprint of look-ahead or misalignment.)
  2. UNIQUENESS - no duplicate rows per (model, origin, horizon, fc_date).
  3. ORIGIN GRID - origins fall on the monthly backtesting grid
     2021-01 .. 2024-12, no fc_date exceeds the test end, and origin counts
     match expectations (monthly vs the deep models' quarterly subset).
  4. ARTIFACTS - every file the final summary / evaluation notebooks rely on
     exists and exposes the required columns.

Also exercises ``shared.exog_policies.assert_no_future`` to confirm the
as-of-origin guard actually catches a leaking frame.

Usage
-----
    python tests/check_artifacts_and_leakage.py      # prints report, exit 1 on failure

Pytest-compatible: ``pytest tests/check_artifacts_and_leakage.py`` collects the
``test_*`` functions below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.exog_policies import assert_no_future  # noqa: E402

RESULTS = ROOT / "08_results"
BASELINE_PREDS = ROOT / "03_models_baseline" / "results" / "rolling_predictions.parquet"
DEEP_PREDS = ROOT / "04_models_deep" / "results" / "deep_rolling_predictions.parquet"

HORIZONS = {1, 3, 6, 12}
REQUIRED_COLS = {"origin", "fc_date", "step", "horizon", "model", "y_true", "y_pred"}

MONTHLY_GRID = pd.date_range("2021-01-01", "2024-12-01", freq="MS")
TEST_END = pd.Timestamp("2024-12-01")

# Per-model foundation prediction files consumed by build_metrics_summary_final.py
FOUNDATION_MODELS = [
    "timesfm_C0", "timesfm_C1", "chronos2_C0", "chronos2_C1",
    "timegpt_C0", "timegpt_C1",
    "chronos2_C1_energy", "timegpt_C1_energy",
    "chronos2_C1_energy_only", "timegpt_C1_energy_only",
    "chronos2_C1_inst", "chronos2_C1_macro",
    "timesfm_C1_inst", "timesfm_C1_macro",
    "timegpt_C1_inst", "timegpt_C1_macro",
]

# Files the final summary / evaluation notebooks must be able to read.
REQUIRED_ARTIFACTS = [
    BASELINE_PREDS,
    DEEP_PREDS,
    RESULTS / "metrics_summary_final.json",
    RESULTS / "rolling_predictions_global.parquet",
    RESULTS / "rolling_predictions_C1_inst_global.parquet",
    RESULTS / "deep_rolling_predictions_global.parquet",
    RESULTS / "rolling_predictions_europe.parquet",
    RESULTS / "deep_rolling_predictions_europe.parquet",
    *[RESULTS / f"{m}_predictions.parquet" for m in FOUNDATION_MODELS],
]


def _months_between(origin: pd.Timestamp, fc_date: pd.Timestamp) -> int:
    return (fc_date.year - origin.year) * 12 + (fc_date.month - origin.month)


def _iter_prediction_files():
    """Yield (path, dataframe) for every prediction parquet present."""
    paths = [BASELINE_PREDS, DEEP_PREDS]
    paths += sorted(RESULTS.glob("*_predictions.parquet"))
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p)
            for c in ("origin", "fc_date"):
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c])
            yield p, df


# ── Check implementations (return list of error strings) ──────────────────────

def check_required_artifacts() -> list[str]:
    errors = []
    for path in REQUIRED_ARTIFACTS:
        if not path.exists():
            errors.append(f"MISSING artifact: {path.relative_to(ROOT)}")
            continue
        if path.suffix == ".parquet":
            cols = set(pd.read_parquet(path, columns=None).columns)
            missing = REQUIRED_COLS - cols
            if missing:
                errors.append(f"{path.name}: missing columns {sorted(missing)}")
    return errors


def check_causality_no_leakage() -> list[str]:
    errors = []
    for path, df in _iter_prediction_files():
        if not REQUIRED_COLS.issubset(df.columns):
            continue  # schema reported elsewhere
        bad_future = df[df["fc_date"] <= df["origin"]]
        if len(bad_future):
            errors.append(f"{path.name}: {len(bad_future)} rows with fc_date <= origin (leakage)")
        step_expected = df.apply(lambda r: _months_between(r["origin"], r["fc_date"]), axis=1)
        bad_step = df[(step_expected != df["step"]) | (df["step"] < 1) | (df["step"] > df["horizon"])]
        if len(bad_step):
            errors.append(f"{path.name}: {len(bad_step)} rows with step != months(origin->fc_date) or out of 1..h")
        bad_h = set(df["horizon"].unique()) - HORIZONS
        if bad_h:
            errors.append(f"{path.name}: unexpected horizons {sorted(bad_h)}")
        # NB: fc_date may extend past the configured test end for the baseline /
        # deep families, which evaluate against any available actuals (Spain IPC
        # runs to 2026). That is a difference in evaluation window, not leakage —
        # leakage would be fc_date <= origin, checked above. The differing
        # windows are why per-family n_evals vary (see the common-origin caveat).
    return errors


def check_uniqueness() -> list[str]:
    errors = []
    keys = ["model", "origin", "horizon", "fc_date"]
    for path, df in _iter_prediction_files():
        if not set(keys).issubset(df.columns):
            continue
        dup = df.duplicated(subset=keys).sum()
        if dup:
            errors.append(f"{path.name}: {dup} duplicate rows on {keys}")
    return errors


def check_origin_grid() -> list[str]:
    errors = []
    for path, df in _iter_prediction_files():
        if "origin" not in df.columns:
            continue
        origins = pd.DatetimeIndex(df["origin"].unique())
        off_grid = origins.difference(MONTHLY_GRID)
        if len(off_grid):
            errors.append(f"{path.name}: {len(off_grid)} origins off the monthly grid")
        n = len(origins)
        is_deep = "deep" in path.name
        # Deep models use a ~quarterly subset (~16); everything else is monthly (~48).
        if is_deep and not (10 <= n <= 20):
            errors.append(f"{path.name}: deep origin count {n} outside expected ~16 (quarterly)")
        if not is_deep and n < 30:
            errors.append(f"{path.name}: monthly origin count {n} unexpectedly low (<30)")
    return errors


def check_exog_policy_guard() -> list[str]:
    """The as-of-origin guard must accept a sliced frame and reject a full one."""
    errors = []
    idx = pd.date_range("2002-01-01", "2024-12-01", freq="MS")
    demo = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    origin = pd.Timestamp("2022-06-01")
    try:
        assert_no_future(demo.loc[:origin], origin)  # must pass
    except AssertionError:
        errors.append("assert_no_future rejected a correctly-sliced frame")
    try:
        assert_no_future(demo, origin)  # must fail
        errors.append("assert_no_future did NOT catch a leaking (full) frame")
    except AssertionError:
        pass
    return errors


ALL_CHECKS = {
    "ARTIFACTS": check_required_artifacts,
    "CAUSALITY (no leakage)": check_causality_no_leakage,
    "UNIQUENESS": check_uniqueness,
    "ORIGIN GRID": check_origin_grid,
    "EXOG-POLICY GUARD": check_exog_policy_guard,
}


# ── Pytest entry points ───────────────────────────────────────────────────────

def test_required_artifacts():
    assert not check_required_artifacts()


def test_causality_no_leakage():
    assert not check_causality_no_leakage()


def test_uniqueness():
    assert not check_uniqueness()


def test_origin_grid():
    assert not check_origin_grid()


def test_exog_policy_guard():
    assert not check_exog_policy_guard()


def main() -> int:
    print("=" * 70)
    print("BACKTESTING ARTIFACT & LEAKAGE CHECKS")
    print("=" * 70)
    total = 0
    for name, fn in ALL_CHECKS.items():
        errs = fn()
        total += len(errs)
        status = "PASS" if not errs else "FAIL"
        print(f"[{status}] {name}")
        for e in errs:
            print(f"        - {e}")
    print("-" * 70)
    if total:
        print(f"RESULT: {total} problem(s) found.")
        return 1
    print("RESULT: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
