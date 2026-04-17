"""
Auth - register, login, JWT.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import AUTH_SECRET, AUTH_ALGORITHM, AUTH_EXPIRE_MINUTES
from app.db import get_db, init_db
from app.models.user import User

router = APIRouter()
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# For environments that might reintroduce bcrypt into the CryptContext,
# enforce a safe upper bound for passwords when bcrypt is active.
_BCRYPT_MAX_BYTES = 72


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=100)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AuthResponse(BaseModel):
    token: str
    email: str
    role: str


def validate_password_policy(password: str) -> None:
    """
    Validate password against hashing-backend constraints.

    - If bcrypt is present in the configured schemes, enforce the 72-byte limit.
    - Otherwise (e.g. argon2-only, current default), rely on caller-side length checks.
    """
    # Passlib's CryptContext may be customized via environment or future config;
    # guard against accidental bcrypt re-introduction, which has a 72-byte limit.
    schemes = set(pwd_context.schemes())
    if "bcrypt" in schemes:
        if len(password.encode("utf-8")) > _BCRYPT_MAX_BYTES:
            raise ValueError(
                f"Password too long for bcrypt hashing backend "
                f"(>{_BCRYPT_MAX_BYTES} bytes). Please use a shorter ADMIN_PASSWORD."
            )


def _hash_password(password: str) -> str:
    validate_password_policy(password)
    return pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def _create_token(email: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=AUTH_EXPIRE_MINUTES)
    payload = {"sub": email, "role": role, "exp": expire}
    return jwt.encode(payload, AUTH_SECRET, algorithm=AUTH_ALGORITHM)


def get_current_user(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, AUTH_SECRET, algorithms=[AUTH_ALGORITHM])
        return {"email": payload.get("sub"), "role": payload.get("role", "user")}
    except JWTError:
        return None


@router.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    init_db()
    email = (req.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    existing = db.query(User).filter(func.lower(func.trim(User.email)) == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=email,
        password_hash=_hash_password(req.password),
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = _create_token(user.email, user.role)
    return AuthResponse(token=token, email=user.email, role=user.role)


@router.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    init_db()
    email = (req.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    user = db.query(User).filter(func.lower(func.trim(User.email)) == email).first()
    if not user or not _verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user.last_login = datetime.utcnow()
    db.commit()
    token = _create_token(user.email, user.role)
    return AuthResponse(token=token, email=user.email, role=user.role)


@router.get("/auth/me")
def get_me(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user
