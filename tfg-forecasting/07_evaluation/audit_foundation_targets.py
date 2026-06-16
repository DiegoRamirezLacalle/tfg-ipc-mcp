"""
audit_foundation_targets.py - integrity audit for foundation-model targets
===========================================================================

Guards against the Spain/Global C0 contamination bug: foundation-model
prediction files whose y_true silently came from the wrong series.

Checks (exit non-zero if ANY fails):
  1. Spain  C0 prediction files match ipc_spain_index.parquet::indice_general.
  2. Global C0 prediction files match cpi_global_monthly.parquet::cpi_global_rate.
  3. Global C1 prediction files match cpi_global_monthly.parquet::cpi_global_rate.
  4. No GLOBAL foundation prediction file has y_true on Spain's 80-100 index scale.
  5. Spain and Global C0 h=12 MAEs are NOT identical across all three foundation
     models (identical => Global is silently reusing Spain values).

Usage:
    python 07_evaluation/audit_foundation_targets.py
Exit code 0 = all checks pass; 1 = at least one failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))

from shared.logger import get_logger

logger = get_logger(__name__)

RESULTS = ROOT / "08_results"
PROCESSED = ROOT / "data" / "processed"

SPAIN_FILE = PROCESSED / "ipc_spain_index.parquet"
SPAIN_COL = "indice_general"
GLOBAL_FILE = PROCESSED / "cpi_global_monthly.parquet"
GLOBAL_COL = "cpi_global_rate"

SPAIN_C0 = ["chronos2_C0", "timesfm_C0", "timegpt_C0"]
GLOBAL_C0 = ["chronos2_C0_global", "timesfm_C0_global", "timegpt_C0_global"]
GLOBAL_C1 = ["chronos2_C1_inst_global", "timesfm_C1_inst_global", "timegpt_C1_inst_global"]
# Improved Global C1 variants added by the Phase 1-3 work (overlay corrections
# and the honest-forward-path Chronos-2). They must also carry the Global
# cpi_global_rate as y_true and never sit on the Spain index scale.
GLOBAL_C1_EXTRA = [
    "chronos2_C1_fwd_global",
    "chronos2_C1_regime_global", "timesfm_C1_regime_global",
    "chronos2_C1_validated_global", "timesfm_C1_validated_global",
]

FOUNDATION_FAMILIES = ["chronos2", "timesfm", "timegpt"]
SPAIN_INDEX_FLOOR = 40.0   # Spain index sits at 58-101; a CPI rate never exceeds ~10


def _load_target(file: Path, col: str) -> pd.Series:
    df = pd.read_parquet(file)
    df.index = pd.to_datetime(df.index)
    return df[col]


def _check_match(model: str, target: pd.Series, label: str, errors: list[str]) -> None:
    p = RESULTS / f"{model}_predictions.parquet"
    if not p.exists():
        errors.append(f"[{model}] prediction file MISSING ({p.name})")
        return
    d = pd.read_parquet(p)
    d["fc_date"] = pd.to_datetime(d["fc_date"])
    expected = target.reindex(d["fc_date"].values).values
    actual = d["y_true"].values
    if np.isnan(expected).any():
        n = int(np.isnan(expected).sum())
        errors.append(f"[{model}] {n} fc_date(s) not present in {label} target series")
        return
    if not np.allclose(actual, expected, atol=1e-6):
        n = int(np.sum(~np.isclose(actual, expected, atol=1e-6)))
        errors.append(f"[{model}] y_true != {label} ({n}/{len(d)} rows mismatch)")
    else:
        logger.info("  OK  %-28s y_true == %s  (n=%d)", model, label, len(d))


def _check_not_spain_scale(model: str, errors: list[str]) -> None:
    p = RESULTS / f"{model}_predictions.parquet"
    if not p.exists():
        return  # missing-file handled elsewhere
    d = pd.read_parquet(p)
    lo, hi = float(d["y_true"].min()), float(d["y_true"].max())
    if lo > SPAIN_INDEX_FLOOR:
        errors.append(
            f"[{model}] y_true range [{lo:.2f},{hi:.2f}] is on the Spain index "
            f"scale (>{SPAIN_INDEX_FLOOR}) but this is a GLOBAL file"
        )
    else:
        logger.info("  OK  %-28s y_true range [%.3f, %.3f] (not Spain scale)", model, lo, hi)


def _h12_mae(metrics_file: str, key: str) -> float | None:
    p = RESULTS / metrics_file
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get(key, {}).get("h12", {}).get("MAE")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    spain_t = _load_target(SPAIN_FILE, SPAIN_COL)
    global_t = _load_target(GLOBAL_FILE, GLOBAL_COL)

    logger.info("=" * 64)
    logger.info("FOUNDATION TARGET-INTEGRITY AUDIT")
    logger.info("=" * 64)

    # 1. Spain C0 must match indice_general
    logger.info("\n[1] Spain C0 predictions match indice_general:")
    for m in SPAIN_C0:
        _check_match(m, spain_t, "Spain.indice_general", errors)

    # 2. Global C0 must match cpi_global_rate
    logger.info("\n[2] Global C0 predictions match cpi_global_rate:")
    for m in GLOBAL_C0:
        if not (RESULTS / f"{m}_predictions.parquet").exists():
            warnings.append(f"[{m}] Global C0 prediction file not yet generated")
            logger.info("  --  %-28s not generated yet (run 30/31/32)", m)
            continue
        _check_match(m, global_t, "Global.cpi_global_rate", errors)

    # 3. Global C1 must match cpi_global_rate (institutional + improved variants)
    logger.info("\n[3] Global C1 predictions match cpi_global_rate:")
    for m in GLOBAL_C1:
        _check_match(m, global_t, "Global.cpi_global_rate", errors)
    for m in GLOBAL_C1_EXTRA:
        if (RESULTS / f"{m}_predictions.parquet").exists():
            _check_match(m, global_t, "Global.cpi_global_rate", errors)
        else:
            warnings.append(f"[{m}] improved Global C1 variant not generated")

    # 4. No GLOBAL foundation file has Spain-scale y_true
    logger.info("\n[4] Global foundation files NOT on Spain index scale:")
    for m in GLOBAL_C0 + GLOBAL_C1 + GLOBAL_C1_EXTRA:
        _check_not_spain_scale(m, errors)

    # 5. Spain vs Global C0 h=12 MAE must NOT be identical across all three families
    logger.info("\n[5] Spain C0 h12 MAE != Global C0 h12 MAE (per family):")
    identical = 0
    compared = 0
    for fam in FOUNDATION_FAMILIES:
        sp = _h12_mae(f"{fam}_C0_metrics.json", f"{fam}_C0")
        gl = _h12_mae(f"{fam}_C0_global_metrics.json", f"{fam}_C0_global")
        if sp is None or gl is None:
            logger.info("  --  %-10s spain=%s global=%s (one missing, skipped)", fam, sp, gl)
            continue
        compared += 1
        same = abs(sp - gl) < 1e-6
        identical += int(same)
        flag = "IDENTICAL!" if same else "distinct"
        logger.info("  %s  %-10s spain_h12=%.4f  global_h12=%.4f  (%s)",
                    "!!" if same else "OK", fam, sp, gl, flag)
    if compared > 0 and identical == compared:
        errors.append(
            f"All {compared} Global C0 h=12 MAEs are IDENTICAL to Spain C0 — "
            f"Global is silently reusing Spain values"
        )

    # ── Verdict ──
    logger.info("\n" + "=" * 64)
    if warnings:
        logger.warning("WARNINGS (not failures):")
        for w in warnings:
            logger.warning("  - %s", w)
    if errors:
        logger.error("AUDIT FAILED — %d problem(s):", len(errors))
        for e in errors:
            logger.error("  - %s", e)
        logger.info("=" * 64)
        return 1
    logger.info("AUDIT PASSED — all target-integrity checks OK.")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
