import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.permissions import get_current_user
from app.core.security import create_token, hash_password, verify_password
from app.models.user import User, UserRole
from app.schemas.auth import AuthOut, LoginIn, SignupIn, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
log = structlog.get_logger()


@router.post("/signup", response_model=AuthOut, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupIn, db: AsyncSession = Depends(get_db)) -> AuthOut:
    email = payload.email.lower().strip()
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    # Only admin role can be assigned via signup when explicitly requested;
    # public signups are always viewer regardless of payload
    role = payload.role if payload.role != UserRole.admin else UserRole.viewer

    user = User(email=email, password_hash=hash_password(payload.password), role=role)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_token(str(user.id), {"email": user.email, "role": user.role.value})
    log.info("user_signup", user_id=user.id, email=user.email, role=user.role.value)
    return AuthOut(access_token=token, user=UserOut.model_validate(user))


@router.post("/login", response_model=AuthOut)
async def login(payload: LoginIn, db: AsyncSession = Depends(get_db)) -> AuthOut:
    email = payload.email.lower().strip()
    user = await db.scalar(select(User).where(User.email == email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account disabled")

    token = create_token(str(user.id), {"email": user.email, "role": user.role.value})
    log.info("user_login", user_id=user.id, email=user.email)
    return AuthOut(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)
