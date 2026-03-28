"""
backend/app/api/auth_routes.py

Authentication endpoints:
  POST /auth/register
  POST /auth/login
  POST /auth/refresh
  POST /auth/password-reset/request
  POST /auth/password-reset/confirm
  GET  /auth/me
  PUT  /auth/me
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.core.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    create_reset_token, decode_token,
    get_current_user,
)
from backend.app.core.config import settings
from backend.app.db.database import get_db
from backend.app.db.models import User
from backend.app.schemas.auth import (
    RegisterRequest, LoginResponse, RefreshRequest,
    PasswordResetRequest, PasswordResetConfirm,
    UserPublic, UserUpdateRequest,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserPublic, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already taken.")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=LoginResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db:   Session = Depends(get_db),
):
    # Accept username or email in the username field
    user = (
        db.query(User).filter(User.username == form.username).first()
        or db.query(User).filter(User.email == form.username).first()
    )
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled.")

    token_data      = {"sub": str(user.id)}
    access_token    = create_access_token(token_data)
    refresh_token   = create_refresh_token(token_data)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user,
    )


@router.post("/refresh", response_model=LoginResponse)
def refresh_token(req: RefreshRequest, db: Session = Depends(get_db)):
    payload = decode_token(req.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type.")

    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found.")

    token_data    = {"sub": str(user.id)}
    access_token  = create_access_token(token_data)
    refresh_token_new = create_refresh_token(token_data)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token_new,
        user=user,
    )


@router.post("/password-reset/request")
def request_password_reset(req: PasswordResetRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    # Always return 200 to prevent email enumeration
    if user:
        token = create_reset_token()
        user.reset_token = token
        user.reset_token_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        db.commit()
        # In production: send email with reset link containing token
        # For now: return token directly (remove this in production)
        return {"message": "Reset token generated.", "debug_token": token}
    return {"message": "If that email exists, a reset link has been sent."}


@router.post("/password-reset/confirm")
def confirm_password_reset(req: PasswordResetConfirm, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.reset_token == req.token).first()
    if not user or not user.reset_token_expires:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")

    if datetime.now(timezone.utc) > user.reset_token_expires.replace(tzinfo=timezone.utc):
        raise HTTPException(status_code=400, detail="Reset token has expired.")

    user.hashed_password    = hash_password(req.new_password)
    user.reset_token        = None
    user.reset_token_expires = None
    db.commit()
    return {"message": "Password updated successfully."}


@router.get("/me", response_model=UserPublic)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserPublic)
def update_me(
    req:          UserUpdateRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    if req.username and req.username != current_user.username:
        if db.query(User).filter(User.username == req.username).first():
            raise HTTPException(status_code=400, detail="Username already taken.")
        current_user.username = req.username

    if req.email and req.email != current_user.email:
        if db.query(User).filter(User.email == req.email).first():
            raise HTTPException(status_code=400, detail="Email already registered.")
        current_user.email = req.email

    db.commit()
    db.refresh(current_user)
    return current_user
