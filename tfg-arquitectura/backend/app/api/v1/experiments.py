import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_db
from app.core.permissions import get_current_user, require_role
from app.models.dataset import Series
from app.models.experiment import Experiment, Run
from app.models.model_catalog import ModelCatalog
from app.models.user import User, UserRole
from app.schemas.experiment import ExperimentCreate, ExperimentDetailOut, ExperimentOut
from app.schemas.run import RunOut

router = APIRouter(prefix="/experiments", tags=["experiments"])
log = structlog.get_logger()

_researcher_or_admin = require_role(UserRole.researcher, UserRole.admin)


@router.post("", response_model=ExperimentOut, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    payload: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(_researcher_or_admin),
) -> ExperimentOut:
    if not await db.get(Series, payload.series_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Series not found")
    if not await db.get(ModelCatalog, payload.model_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")

    exp = Experiment(
        user_id=current_user.id,
        name=payload.name,
        series_id=payload.series_id,
        model_id=payload.model_id,
        horizon=payload.horizon,
        use_mcp=payload.use_mcp,
        config=payload.config,
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)
    log.info("experiment_created", experiment_id=exp.id, user_id=current_user.id)
    return ExperimentOut.model_validate(exp)


@router.get("", response_model=list[ExperimentOut])
async def list_experiments(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ExperimentOut]:
    q = select(Experiment).order_by(Experiment.created_at.desc()).offset(offset).limit(limit)
    # Non-admins only see their own experiments
    if current_user.role != UserRole.admin:
        q = q.where(Experiment.user_id == current_user.id)
    rows = await db.execute(q)
    return [ExperimentOut.model_validate(e) for e in rows.scalars().all()]


@router.get("/{experiment_id}", response_model=ExperimentDetailOut)
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ExperimentDetailOut:
    result = await db.execute(
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(selectinload(Experiment.runs))
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Experiment not found")
    if current_user.role != UserRole.admin and exp.user_id != current_user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied")
    return ExperimentDetailOut.model_validate(exp)


@router.get("/{experiment_id}/runs", response_model=list[RunOut])
async def list_runs(
    experiment_id: int,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[RunOut]:
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Experiment not found")
    if current_user.role != UserRole.admin and exp.user_id != current_user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied")

    rows = await db.execute(
        select(Run)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return [RunOut.model_validate(r) for r in rows.scalars().all()]


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(_researcher_or_admin),
) -> None:
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Experiment not found")
    if current_user.role != UserRole.admin and exp.user_id != current_user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied")
    await db.delete(exp)
    await db.commit()
    log.info("experiment_deleted", experiment_id=experiment_id, user_id=current_user.id)
