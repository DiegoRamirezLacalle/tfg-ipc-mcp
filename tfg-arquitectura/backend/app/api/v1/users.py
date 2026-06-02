import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.permissions import require_role
from app.models.user import User, UserRole
from app.schemas.auth import UserOut, UserRoleUpdate

router = APIRouter(prefix="/users", tags=["users"])
log = structlog.get_logger()

_admin_only = require_role(UserRole.admin)


@router.patch("/{user_id}/role", response_model=UserOut)
async def update_role(
    user_id: int,
    payload: UserRoleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(_admin_only),
) -> UserOut:
    target = await db.get(User, user_id)
    if not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    # prevent admin from demoting themselves — avoids lockout
    if target.id == current_user.id and payload.role != UserRole.admin:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Admin cannot demote their own account",
        )

    target.role = payload.role
    await db.commit()
    await db.refresh(target)
    log.info(
        "user_role_updated",
        target_id=target.id,
        new_role=payload.role.value,
        by_admin=current_user.id,
    )
    return UserOut.model_validate(target)
