"""
Weave Authentication Endpoints (/api/v1/auth)
Implements JWT access tokens and secure HTTP-only refresh tokens (PRD Document 7.1).
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    hash_token,
)
from backend.app.models.models import User
from backend.app.schemas.schemas import UserCreate, UserLogin, UserResponse, TokenResponse
from backend.app.repositories.user_repository import UserRepository
from backend.app.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: UserCreate, response: Response, db: Session = Depends(get_db)):
    """Registers a new fashion designer or analyst account."""
    user_repo = UserRepository(db)
    existing = user_repo.get_by_email(payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists."
        )

    user = user_repo.create(
        email=payload.email,
        password=payload.password,
        display_name=payload.display_name,
        role="designer"
    )

    access_token = create_access_token(subject=user.id)
    raw_refresh, token_hash, expires_at = create_refresh_token()
    user_repo.create_refresh_token(user_id=user.id, token_hash=token_hash, expires_at=expires_at)

    response.set_cookie(
        key="weave_refresh_token",
        value=raw_refresh,
        httponly=True,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        samesite="lax",
    )

    return TokenResponse(
        user=UserResponse.model_validate(user),
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, response: Response, db: Session = Depends(get_db)):
    """Authenticates user credentials and issues tokens."""
    user_repo = UserRepository(db)
    user = user_repo.get_by_email(payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive."
        )

    access_token = create_access_token(subject=user.id)
    raw_refresh, token_hash, expires_at = create_refresh_token()
    user_repo.create_refresh_token(user_id=user.id, token_hash=token_hash, expires_at=expires_at)

    response.set_cookie(
        key="weave_refresh_token",
        value=raw_refresh,
        httponly=True,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        samesite="lax",
    )

    return TokenResponse(
        user=UserResponse.model_validate(user),
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(request: Request, response: Response, db: Session = Depends(get_db)):
    """Exchanges an HTTP-only refresh token for a fresh access token."""
    raw_token = request.cookies.get("weave_refresh_token")
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token cookie missing."
        )

    user_repo = UserRepository(db)
    token_hash = hash_token(raw_token)
    db_token = user_repo.get_active_refresh_token(token_hash)

    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token."
        )

    now = datetime.now(timezone.utc)
    token_exp = db_token.expires_at
    if token_exp.tzinfo is None:
        token_exp = token_exp.replace(tzinfo=timezone.utc)

    if token_exp < now:
        user_repo.revoke_refresh_token(token_hash)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired."
        )

    user = user_repo.get_by_id(db_token.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account inactive or deleted."
        )

    # Issue new access token
    new_access_token = create_access_token(subject=user.id)
    return TokenResponse(
        user=UserResponse.model_validate(user),
        access_token=new_access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    """Revokes refresh token and clears cookie."""
    raw_token = request.cookies.get("weave_refresh_token")
    if raw_token:
        user_repo = UserRepository(db)
        token_hash = hash_token(raw_token)
        user_repo.revoke_refresh_token(token_hash)

    response.delete_cookie(key="weave_refresh_token")
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Returns the profile of the currently authenticated designer."""
    return UserResponse.model_validate(current_user)
