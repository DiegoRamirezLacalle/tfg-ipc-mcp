"""Metrics aggregation — cross-experiment comparison table + DM significance."""

import numpy as np
from scipy import stats
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_db
from app.core.permissions import get_current_user
from app.models.dataset import Observation
from app.models.experiment import Experiment, Metric, Prediction, Run, RunStatus
from app.models.model_catalog import ModelCatalog
from app.models.user import User, UserRole
from app.schemas.metrics import ComparisonRow, MetricValues

router = APIRouter(prefix="/metrics", tags=["metrics"])

_MAX_EXPERIMENTS = 20
_DM_ALPHA = 0.05
_DM_MIN_N = 4


@router.get(
    "/compare",
    response_model=list[ComparisonRow],
    summary="Compare metrics across experiments",
    description=(
        "Returns the most-recent **done** run for each requested experiment, "
        "with MAE / RMSE / MAPE side-by-side. Useful for building model comparison "
        "tables identical to the rolling-backtesting results in the research phase. "
        "Non-admin users can only compare their own experiments. "
        f"Maximum {_MAX_EXPERIMENTS} experiments per request."
    ),
)
async def compare_experiments(
    experiment_ids: list[int] = Query(
        ...,
        description="Experiment IDs to compare (max 20).",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ComparisonRow]:
    if len(experiment_ids) > _MAX_EXPERIMENTS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Maximum {_MAX_EXPERIMENTS} experiment IDs allowed per request.",
        )
    if not experiment_ids:
        return []

    # Deduplicate while preserving order
    seen: set[int] = set()
    ordered_ids = [eid for eid in experiment_ids if not (eid in seen or seen.add(eid))]

    rows: list[ComparisonRow] = []

    for exp_id in ordered_ids:
        exp = await db.get(Experiment, exp_id)
        if not exp:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Experiment {exp_id} not found")
        if current_user.role != UserRole.admin and exp.user_id != current_user.id:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Access denied to experiment {exp_id}",
            )

        model_cat = await db.get(ModelCatalog, exp.model_id)

        # Latest done run for this experiment
        result = await db.execute(
            select(Run)
            .where(Run.experiment_id == exp_id, Run.status == RunStatus.done)
            .order_by(Run.finished_at.desc())
            .limit(1)
        )
        run = result.scalar_one_or_none()

        metric_values: MetricValues | None = None
        if run:
            metric_rows = await db.execute(
                select(Metric).where(Metric.run_id == run.id)
            )
            metrics_dict = {m.name: m.value for m in metric_rows.scalars().all()}
            metric_values = MetricValues(
                mae=metrics_dict.get("mae"),
                rmse=metrics_dict.get("rmse"),
                mape=metrics_dict.get("mape"),
            )

        rows.append(
            ComparisonRow(
                experiment_id=exp.id,
                experiment_name=exp.name,
                model_slug=model_cat.slug if model_cat else "unknown",
                model_name=model_cat.name if model_cat else "unknown",
                horizon=exp.horizon,
                use_mcp=exp.use_mcp,
                run_id=run.id if run else None,
                run_finished_at=run.finished_at if run else None,
                metrics=metric_values,
            )
        )

    return rows


def _diebold_mariano_hln(
    e1: np.ndarray, e2: np.ndarray, h: int, power: int = 2
) -> dict:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.

    Compares two models' forecast errors over a common set of observations.
    Loss differential d = |e1|^power - |e2|^power (power=2 → squared loss).

    The long-run variance uses autocovariances up to lag (h-1) for h-step
    forecasts. The HLN factor corrects DM for small samples and the statistic
    is referred to a t-distribution with (n-1) df — appropriate for the short
    per-run evaluation windows on this platform.

    dm_stat < 0 → model 1 better (lower loss); > 0 → model 2 better.
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_bar = float(np.mean(d))

    h_lag = max(1, min(h, n - 1))
    dev = d - d_bar
    gamma0 = float(np.mean(dev * dev))
    gammas = [
        float(np.mean(dev[k:] * dev[:-k])) for k in range(1, h_lag)
    ]
    var_d = (gamma0 + 2.0 * sum(gammas)) / n
    if var_d <= 0:
        var_d = gamma0 / n if gamma0 > 0 else 1e-12

    dm = d_bar / np.sqrt(var_d)

    # HLN small-sample correction factor
    hln_term = (n + 1 - 2 * h_lag + h_lag * (h_lag - 1) / n) / n
    k = float(np.sqrt(hln_term)) if hln_term > 0 else 1.0
    dm_hln = k * dm

    p_value = float(2.0 * stats.t.sf(abs(dm_hln), df=n - 1))

    better = "tie"
    if p_value < _DM_ALPHA:
        better = "model2" if dm_hln > 0 else "model1"

    return {
        "dm_stat": round(float(dm_hln), 4),
        "p_value": round(p_value, 4),
        "better": better,
        "significant": bool(p_value < _DM_ALPHA),
        "n": n,
    }


@router.get(
    "/dm-matrix",
    summary="Pairwise Diebold-Mariano significance matrix",
    description=(
        "Computes the HLN-corrected Diebold-Mariano test for every pair of the "
        "given runs. Pairs are only tested when the runs share the same target "
        "series and have at least 4 aligned forecast points. Returns the per-run "
        "MAE plus, for each pair, the DM statistic, p-value, and which model is "
        "significantly better (p < 0.05). Mirrors the rolling-backtest DM tests "
        "from the research phase, adapted to single-window per-run forecasts."
    ),
)
async def dm_matrix(
    run_ids: list[int] = Query(..., description="Run IDs to compare (max 20)."),
    power: int = Query(2, ge=1, le=2, description="Loss power: 1=MAE, 2=squared."),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    if len(run_ids) > _MAX_EXPERIMENTS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Maximum {_MAX_EXPERIMENTS} runs allowed per request.",
        )
    seen: set[int] = set()
    ordered = [r for r in run_ids if not (r in seen or seen.add(r))]

    runs_meta: list[dict] = []
    errors_by_run: dict[int, dict] = {}  # run_id -> {timestamp -> error}
    series_by_run: dict[int, int] = {}
    actuals_cache: dict[int, dict] = {}

    for rid in ordered:
        run = await db.get(Run, rid)
        if not run:
            continue
        exp = await db.get(Experiment, run.experiment_id)
        if not exp:
            continue
        if current_user.role != UserRole.admin and exp.user_id != current_user.id:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, f"Access denied to run {rid}"
            )
        model_cat = await db.get(ModelCatalog, exp.model_id)

        preds_rows = await db.execute(
            select(Prediction)
            .where(Prediction.run_id == rid)
            .order_by(Prediction.timestamp)
        )
        preds = preds_rows.scalars().all()

        # Actuals for this series (cached)
        if exp.series_id not in actuals_cache:
            obs_rows = await db.execute(
                select(Observation).where(Observation.series_id == exp.series_id)
            )
            actuals_cache[exp.series_id] = {
                o.timestamp: float(o.value) for o in obs_rows.scalars().all()
            }
        actuals = actuals_cache[exp.series_id]

        err = {
            p.timestamp: float(p.value) - actuals[p.timestamp]
            for p in preds
            if p.timestamp in actuals
        }
        errors_by_run[rid] = err
        series_by_run[rid] = exp.series_id

        mae = (
            round(float(np.mean([abs(v) for v in err.values()])), 4)
            if err
            else None
        )
        runs_meta.append(
            {
                "run_id": rid,
                "experiment_id": exp.id,
                "experiment_name": exp.name,
                "model_slug": model_cat.slug if model_cat else "unknown",
                "model_name": model_cat.name if model_cat else "unknown",
                "horizon": exp.horizon,
                "use_mcp": exp.use_mcp,
                "mae": mae,
                "n_points": len(err),
            }
        )

    pairs: list[dict] = []
    for i in range(len(runs_meta)):
        for j in range(i + 1, len(runs_meta)):
            a = runs_meta[i]["run_id"]
            b = runs_meta[j]["run_id"]
            if series_by_run.get(a) != series_by_run.get(b):
                pairs.append(
                    {"a_run_id": a, "b_run_id": b, "comparable": False,
                     "reason": "different series"}
                )
                continue
            ea, eb = errors_by_run[a], errors_by_run[b]
            common = sorted(set(ea) & set(eb))
            if len(common) < _DM_MIN_N:
                pairs.append(
                    {"a_run_id": a, "b_run_id": b, "comparable": False,
                     "reason": f"only {len(common)} aligned points"}
                )
                continue
            e1 = np.array([ea[t] for t in common], dtype=np.float64)
            e2 = np.array([eb[t] for t in common], dtype=np.float64)
            h = min(runs_meta[i]["horizon"], runs_meta[j]["horizon"], len(common))
            dm = _diebold_mariano_hln(e1, e2, h=h, power=power)
            pairs.append({"a_run_id": a, "b_run_id": b, "comparable": True, **dm})

    return {"power": power, "alpha": _DM_ALPHA, "runs": runs_meta, "pairs": pairs}
