"""Drift detection endpoint.

GET /api/v1/drift?experiment_id=<id>

Computes a two-sample KS test on forecast residuals from the most recent
completed run of the specified experiment. Splits residuals into an early
window (first 60%) and a recent window (last 40%) and tests whether their
distributions differ significantly.

Returns:
    {
        "experiment_id": int,
        "run_id": int | null,
        "drifted": bool,
        "p_value": float | null,
        "ks_statistic": float | null,
        "n_early": int,
        "n_recent": int,
        "message": str
    }
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from scipy.stats import ks_2samp
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.permissions import get_current_user
from app.models.dataset import Observation
from app.models.experiment import Experiment, Metric, Prediction, Run, RunStatus
from app.models.user import User, UserRole

router = APIRouter(tags=["drift"])

_DRIFT_ALPHA = 0.05  # significance threshold


@router.get("/drift")
async def check_drift(
    experiment_id: int = Query(..., description="Experiment to check for residual drift"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Experiment not found")

    if current_user.role != UserRole.admin and exp.user_id != current_user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied")

    # Find the latest done run for this experiment
    run_result = await db.execute(
        select(Run)
        .where(Run.experiment_id == experiment_id, Run.status == RunStatus.done)
        .order_by(Run.finished_at.desc())
        .limit(1)
    )
    run = run_result.scalar_one_or_none()

    if not run:
        return {
            "experiment_id": experiment_id,
            "run_id": None,
            "drifted": False,
            "p_value": None,
            "ks_statistic": None,
            "n_early": 0,
            "n_recent": 0,
            "message": "No completed run found for this experiment.",
        }

    # Load predictions
    preds_result = await db.execute(
        select(Prediction)
        .where(Prediction.run_id == run.id)
        .order_by(Prediction.timestamp)
    )
    preds = preds_result.scalars().all()

    # Load actuals for the same timestamps
    obs_result = await db.execute(
        select(Observation)
        .where(
            Observation.series_id == exp.series_id,
            Observation.timestamp.in_([p.timestamp for p in preds]),
        )
        .order_by(Observation.timestamp)
    )
    obs_map = {o.timestamp: o.value for o in obs_result.scalars().all()}

    residuals = [
        float(p.value) - float(obs_map[p.timestamp])
        for p in preds
        if p.timestamp in obs_map
    ]

    n = len(residuals)
    if n < 4:
        return {
            "experiment_id": experiment_id,
            "run_id": run.id,
            "drifted": False,
            "p_value": None,
            "ks_statistic": None,
            "n_early": n,
            "n_recent": 0,
            "message": f"Insufficient residuals for KS test ({n} < 4).",
        }

    split = max(2, int(n * 0.6))
    early  = residuals[:split]
    recent = residuals[split:]

    ks_stat, p_value = ks_2samp(early, recent)
    drifted = bool(p_value < _DRIFT_ALPHA)

    return {
        "experiment_id": experiment_id,
        "run_id": run.id,
        "drifted": drifted,
        "p_value": round(float(p_value), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "n_early": len(early),
        "n_recent": len(recent),
        "message": (
            f"Drift detected (KS={ks_stat:.3f}, p={p_value:.4f} < {_DRIFT_ALPHA})"
            if drifted
            else f"No significant drift (KS={ks_stat:.3f}, p={p_value:.4f})"
        ),
    }
