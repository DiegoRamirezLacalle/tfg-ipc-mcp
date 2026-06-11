import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.config import settings
from app.core.permissions import get_current_user
from app.db.mongo import get_mongo_db
from app.db.postgres import AsyncSessionLocal
from app.forecasting.adapters.ridge import EXOG_COLS as _RIDGE_EXOG_COLS
from app.forecasting.base import ForecastInput, ForecastResult
from app.forecasting.registry import get_adapter, supported_slugs
from app.mcp.client import fetch_signals_for_timestamps
from app.models.dataset import Dataset, Observation, Series
from app.models.experiment import Experiment, Metric, Prediction, Run, RunStatus
from app.models.model_catalog import ModelCatalog
from app.models.user import User, UserRole
from app.schemas.run import MetricOut, PredictionOut, RunOut

router = APIRouter(tags=["runs"])
log = structlog.get_logger()


def _compute_metrics(result: ForecastResult) -> dict[str, float]:
    y_true = result.test_actuals
    y_pred = result.predictions
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else 0.0
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _owns_or_admin(exp: Experiment, user: User) -> None:
    if user.role != UserRole.admin and exp.user_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied")


async def _load_exog_for_ridge(db) -> pd.DataFrame | None:
    """Load ECB exogenous features from the features-exog dataset."""
    ds_result = await db.execute(
        select(Dataset).where(Dataset.slug == "features-exog")
    )
    ds = ds_result.scalar_one_or_none()
    if not ds:
        return None

    frames: dict[str, pd.Series] = {}
    for col_slug in _RIDGE_EXOG_COLS:
        s_result = await db.execute(
            select(Series).where(Series.dataset_id == ds.id, Series.slug == col_slug)
        )
        s = s_result.scalar_one_or_none()
        if not s:
            continue
        obs_result = await db.execute(
            select(Observation)
            .where(Observation.series_id == s.id)
            .order_by(Observation.timestamp)
        )
        obs = obs_result.scalars().all()
        if obs:
            frames[col_slug] = pd.Series(
                [o.value for o in obs],
                index=pd.DatetimeIndex([o.timestamp for o in obs]),
                name=col_slug,
            )

    if not frames:
        return None
    return pd.DataFrame(frames)


_FOUNDATION_SLUGS = {"timegpt", "chronos-2", "timesfm"}


async def _load_stack_preds(
    db, run_ids: list[int]
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Load predictions + inverse-MAE weights for an ensemble-stack run."""
    frames: dict[int, pd.Series] = {}
    maes: dict[int, float] = {}

    for rid in run_ids:
        preds_result = await db.execute(
            select(Prediction)
            .where(Prediction.run_id == rid)
            .order_by(Prediction.timestamp)
        )
        preds = preds_result.scalars().all()
        if not preds:
            log.warning("ensemble_stack_run_no_preds", run_id=rid)
            continue

        frames[rid] = pd.Series(
            [p.value for p in preds],
            index=pd.DatetimeIndex([p.timestamp for p in preds]),
        )

        mae_result = await db.execute(
            select(Metric).where(Metric.run_id == rid, Metric.name == "mae")
        )
        mae_row = mae_result.scalar_one_or_none()
        maes[rid] = float(mae_row.value) if mae_row else 1.0

    if not frames:
        return None, None

    stack_df = pd.DataFrame(frames)
    # Inverse-MAE weights: lower MAE → higher weight
    eps = 1e-9
    weights = np.array([1.0 / (maes.get(rid, 1.0) + eps) for rid in frames])
    return stack_df, weights


async def _execute_forecast(run_id: int) -> None:
    """Background worker — owns its own AsyncSession, never shares with the request."""
    async with AsyncSessionLocal() as db:
        run = await db.get(Run, run_id)
        if not run:
            log.warning("background_run_missing", run_id=run_id)
            return

        exp = await db.get(Experiment, run.experiment_id)
        model_cat = await db.get(ModelCatalog, exp.model_id)

        run.status = RunStatus.running
        run.started_at = datetime.now(timezone.utc)
        await db.commit()

        mcp_signals_list: list[dict] = []

        if exp.use_mcp:
            try:
                # Forecast period = last `horizon` months of observed data (backtest)
                last_obs = await db.scalar(
                    select(Observation.timestamp)
                    .where(Observation.series_id == exp.series_id)
                    .order_by(Observation.timestamp.desc())
                    .limit(1)
                )
                fc_timestamps = (
                    pd.date_range(
                        start=last_obs + pd.DateOffset(months=1),
                        periods=exp.horizon,
                        freq="MS",
                    ).to_list()
                    if last_obs
                    else []
                )
                signals = await fetch_signals_for_timestamps(fc_timestamps)
                if signals:
                    mcp_signals_list = signals
                    mongo_db = get_mongo_db()
                    await mongo_db["mcp_contexts"].replace_one(
                        {"run_id": run.id},
                        {
                            "run_id": run.id,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "signals": signals,
                        },
                        upsert=True,
                    )
                    log.info("mcp_context_stored", run_id=run.id, n_signals=len(signals))
            except Exception as mcp_err:
                log.warning("mcp_context_failed", run_id=run.id, error=str(mcp_err))

        try:
            adapter = get_adapter(model_cat.slug)
        except KeyError:
            run.status = RunStatus.failed
            run.error_message = f"Model '{model_cat.slug}' not implemented. Supported: {supported_slugs()}"
            run.finished_at = datetime.now(timezone.utc)
            await db.commit()
            log.error("run_failed_unsupported_model", run_id=run.id, model=model_cat.slug)
            return

        obs_rows = await db.execute(
            select(Observation)
            .where(Observation.series_id == exp.series_id)
            .order_by(Observation.timestamp)
        )
        obs = obs_rows.scalars().all()

        series = pd.Series(
            [o.value for o in obs],
            index=pd.DatetimeIndex([o.timestamp for o in obs]),
            name="value",
        )

        # Load exogenous features
        exog_df: pd.DataFrame | None = None
        if model_cat.slug in {"ridge-exog", "sarimax"}:
            exog_df = await _load_exog_for_ridge(db)
        elif exp.use_mcp and model_cat.slug in _FOUNDATION_SLUGS:
            from app.forecasting.mcp_exog import build_mcp_exog  # noqa: PLC0415
            exog_df = build_mcp_exog(series.index, mcp_signals_list)
            if exog_df is not None:
                log.info("mcp_exog_loaded", run_id=run.id, shape=exog_df.shape)

        # Load stack predictions for ensemble-stack
        stack_preds_df: pd.DataFrame | None = None
        stack_weights: np.ndarray | None = None
        if model_cat.slug == "ensemble-stack":
            run_ids = list((exp.config or {}).get("stack_run_ids", []))
            if run_ids:
                stack_preds_df, stack_weights = await _load_stack_preds(db, run_ids)
                if stack_preds_df is not None:
                    log.info("ensemble_stack_loaded", run_id=run.id, n_models=len(stack_preds_df.columns))
            else:
                log.warning("ensemble_stack_no_run_ids", run_id=run.id)

        inp = ForecastInput(
            series=series,
            horizon=exp.horizon,
            config=exp.config or {},
            exog=exog_df,
            stack_preds=stack_preds_df,
            stack_weights=stack_weights,
        )

        try:
            loop = asyncio.get_event_loop()
            result: ForecastResult = await loop.run_in_executor(None, adapter.run, inp)

            metrics_dict = _compute_metrics(result)

            for ts, val in zip(result.timestamps, result.predictions):
                db.add(Prediction(run_id=run.id, timestamp=ts, value=float(val)))

            for name, val in metrics_dict.items():
                db.add(Metric(run_id=run.id, name=name, value=val))

            # -- MLflow tracking -----------------------------------------------
            try:
                import mlflow  # noqa: PLC0415
                mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                mlflow.set_experiment("tfg-forecasting-platform")
                with mlflow.start_run(run_name=f"run-{run.id}-{model_cat.slug}"):
                    mlflow.log_params({
                        "model_slug":   model_cat.slug,
                        "horizon":      exp.horizon,
                        "use_mcp":      exp.use_mcp,
                        "series_id":    exp.series_id,
                        "experiment_id": exp.id,
                        "run_id":       run.id,
                        "n_mcp_signals": len(mcp_signals_list),
                    })
                    mlflow.log_metrics(metrics_dict)
                    mlflow.set_tag("platform.run_id", run.id)
            except Exception as mlf_err:
                log.warning("mlflow_tracking_failed", run_id=run.id, error=str(mlf_err))

            run.status = RunStatus.done
            run.finished_at = datetime.now(timezone.utc)
            log.info("run_done", run_id=run.id, model=model_cat.slug)
        except Exception as exc:
            run.status = RunStatus.failed
            run.error_message = str(exc)
            run.finished_at = datetime.now(timezone.utc)
            log.error("run_failed", run_id=run.id, error=str(exc))

        await db.commit()


@router.post(
    "/experiments/{experiment_id}/runs",
    response_model=RunOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_run(
    experiment_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RunOut:
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Experiment not found")
    _owns_or_admin(exp, current_user)

    obs_count = await db.scalar(
        select(Observation.id).where(Observation.series_id == exp.series_id).limit(1)
    )
    if obs_count is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Series has no observations",
        )

    run = Run(experiment_id=exp.id, status=RunStatus.pending)
    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_execute_forecast, run.id)
    log.info("run_queued", run_id=run.id, experiment_id=exp.id, user_id=current_user.id)

    return RunOut.model_validate(run)


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RunOut:
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    exp = await db.get(Experiment, run.experiment_id)
    _owns_or_admin(exp, current_user)
    return RunOut.model_validate(run)


@router.get("/runs/{run_id}/predictions", response_model=list[PredictionOut])
async def get_predictions(
    run_id: int,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PredictionOut]:
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    exp = await db.get(Experiment, run.experiment_id)
    _owns_or_admin(exp, current_user)

    rows = await db.execute(
        select(Prediction)
        .where(Prediction.run_id == run_id)
        .order_by(Prediction.timestamp)
        .offset(offset)
        .limit(limit)
    )
    return [PredictionOut.model_validate(p) for p in rows.scalars().all()]


@router.get("/runs/{run_id}/metrics", response_model=list[MetricOut])
async def get_metrics(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[MetricOut]:
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    exp = await db.get(Experiment, run.experiment_id)
    _owns_or_admin(exp, current_user)

    rows = await db.execute(
        select(Metric).where(Metric.run_id == run_id).order_by(Metric.name)
    )
    return [MetricOut.model_validate(m) for m in rows.scalars().all()]


@router.get("/runs/{run_id}/mcp-context")
async def get_mcp_context(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    exp = await db.get(Experiment, run.experiment_id)
    _owns_or_admin(exp, current_user)

    try:
        mongo_db = get_mongo_db()
        doc = await mongo_db["mcp_contexts"].find_one(
            {"run_id": run_id}, {"_id": 0}
        )
    except Exception as exc:
        log.warning("mongo_context_lookup_failed", run_id=run_id, error=str(exc))
        doc = None
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No MCP context stored for this run")
    return doc


@router.post("/runs/{run_id}/narration")
async def generate_run_narration(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Generate an LLM narrative analysis of a completed run via local Ollama."""
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    exp = await db.get(Experiment, run.experiment_id)
    _owns_or_admin(exp, current_user)

    if run.status != RunStatus.done:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Narration requires a completed run (status=done)",
        )

    metrics_rows = await db.execute(
        select(Metric).where(Metric.run_id == run_id)
    )
    metrics_dict = {m.name: m.value for m in metrics_rows.scalars().all()}

    preds_rows = await db.execute(
        select(Prediction)
        .where(Prediction.run_id == run_id)
        .order_by(Prediction.timestamp)
    )
    pred_values = [float(p.value) for p in preds_rows.scalars().all()]

    mcp_signals: list[dict] | None = None
    if exp.use_mcp:
        try:
            mongo_db = get_mongo_db()
            doc = await mongo_db["mcp_contexts"].find_one({"run_id": run_id}, {"_id": 0})
            if doc:
                mcp_signals = doc.get("signals")
        except Exception as exc:
            log.warning("narration_mcp_lookup_failed", run_id=run_id, error=str(exc))

    model_cat = await db.get(ModelCatalog, exp.model_id)

    from app.services.narration import generate_narration  # noqa: PLC0415
    try:
        text = await generate_narration(
            model_slug=model_cat.slug,
            metrics=metrics_dict,
            predictions=pred_values,
            use_mcp=exp.use_mcp,
            mcp_signals=mcp_signals,
        )
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Ollama unavailable: {exc}",
        ) from exc

    return {"narrative": text, "model": settings.OLLAMA_MODEL}
