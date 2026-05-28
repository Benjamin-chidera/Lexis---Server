import re

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from sqlmodel import Session, select

from database import get_session
from models import User
from auth import (
    COOKIE_NAME,
    TOKEN_EXPIRE_DAYS,
    IS_PRODUCTION,
    hash_password,
    verify_password,
    create_token,
    invalidate_token,
    get_current_user,
    require_admin,
)

router = APIRouter()

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


# --- Request schemas ---

class CheckEmailRequest(BaseModel):
    corporate_email: str

    @field_validator("corporate_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email address.")
        return v


class LoginRequest(BaseModel):
    corporate_email: str
    password: str

    @field_validator("corporate_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email address.")
        return v


class SetPasswordRequest(BaseModel):
    corporate_email: str
    password: str
    confirm_password: str

    @field_validator("corporate_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email address.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class RegisterUserRequest(BaseModel):
    corporate_email: str
    name: str
    role: str = "employee"

    @field_validator("corporate_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email address.")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters.")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("admin", "employee"):
            raise ValueError("Role must be 'admin' or 'employee'.")
        return v


class AdminSetupRequest(BaseModel):
    corporate_email: str
    name: str
    password: str
    confirm_password: str

    @field_validator("corporate_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email address.")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


def _make_auth_response(user: User) -> JSONResponse:
    token = create_token(user.id, user.role)
    response = JSONResponse({
        "user_id": user.id,
        "name": user.name,
        "role": user.role,
        "email": user.corporate_email,
    })
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=IS_PRODUCTION,
        # SameSite=None is required for cross-origin cookie sending (frontend on Vercel,
        # backend on a different domain). SameSite=Lax only sends on same-site requests,
        # so every fetch from the frontend would be cookieless → 401 on every API call.
        # SameSite=None requires Secure=True, which is already enforced in production.
        samesite="none" if IS_PRODUCTION else "lax",
        max_age=TOKEN_EXPIRE_DAYS * 86400,
    )
    return response


# --- Auth endpoints ---

@router.post("/api/auth/check-email")
def check_email(req: CheckEmailRequest, session: Session = Depends(get_session)):
    """
    Step 1 of login. Returns whether the user exists and has a password set.
    Frontend uses this to decide: show login form or new-password form.
    """
    user = session.exec(
        select(User).where(User.corporate_email == req.corporate_email)
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="This email is not registered. Contact your administrator.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Your account has been deactivated. Contact your administrator.")

    return {
        "name": user.name,
        "has_password": user.password_hash is not None,
    }


@router.post("/api/auth/login")
def login(req: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(
        select(User).where(User.corporate_email == req.corporate_email)
    ).first()

    if not user or user.password_hash is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Your account has been deactivated. Contact your administrator.")

    return _make_auth_response(user)


@router.post("/api/auth/set-password")
def set_password(req: SetPasswordRequest, session: Session = Depends(get_session)):
    """New employees use this to set their password on first login."""
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    user = session.exec(
        select(User).where(User.corporate_email == req.corporate_email)
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.password_hash is not None:
        raise HTTPException(status_code=400, detail="Password is already set. Use the login endpoint.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Your account has been deactivated. Contact your administrator.")

    user.password_hash = hash_password(req.password)
    session.add(user)
    session.commit()
    session.refresh(user)

    return _make_auth_response(user)


@router.post("/api/auth/logout")
def logout(current_user: dict = Depends(get_current_user)):
    invalidate_token(current_user["token"])
    response = JSONResponse({"message": "Logged out successfully."})
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/api/auth/me")
def get_me(
    session: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user),
):
    user = session.get(User, current_user["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {
        "user_id": user.id,
        "name": user.name,
        "email": user.corporate_email,
        "role": user.role,
        "is_active": user.is_active,
    }


# --- Admin endpoints ---

@router.post("/api/admin/users", status_code=201)
def register_user(
    req: RegisterUserRequest,
    session: Session = Depends(get_session),
    _admin: dict = Depends(require_admin),
):
    """Admin registers a new employee. The employee sets their own password on first login."""
    existing = session.exec(
        select(User).where(User.corporate_email == req.corporate_email)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="A user with this email already exists.")

    new_user = User(
        corporate_email=req.corporate_email,
        name=req.name,
        role=req.role,
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return {
        "message": "User registered successfully.",
        "user_id": new_user.id,
        "name": new_user.name,
        "email": new_user.corporate_email,
        "role": new_user.role,
    }


@router.get("/api/admin/users")
def list_users(
    session: Session = Depends(get_session),
    _admin: dict = Depends(require_admin),
):
    users = session.exec(select(User)).all()
    return [
        {
            "user_id": u.id,
            "name": u.name,
            "email": u.corporate_email,
            "role": u.role,
            "is_active": u.is_active,
            "has_password": u.password_hash is not None,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@router.patch("/api/admin/users/{user_id}/deactivate")
def deactivate_user(
    user_id: int,
    session: Session = Depends(get_session),
    current_admin: dict = Depends(require_admin),
):
    if user_id == current_admin["user_id"]:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account.")

    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.is_active = False
    session.add(user)
    session.commit()
    return {"message": f"User {user.corporate_email} has been deactivated."}
 

@router.patch("/api/admin/users/{user_id}/activate")
def activate_user(
    user_id: int,
    session: Session = Depends(get_session),
    _admin: dict = Depends(require_admin),
):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.") 

    user.is_active = True
    session.add(user)
    session.commit()
    return {"message": f"User {user.corporate_email} has been activated."}


# --- First-time admin setup ---

@router.post("/api/auth/admin-setup", status_code=201)
def admin_setup(req: AdminSetupRequest, session: Session = Depends(get_session)):
    """
    Creates the very first admin account.
    Only works when zero admins exist in the database.
    """
    existing_admin = session.exec(select(User).where(User.role == "admin")).first()
    if existing_admin:
        raise HTTPException(
            status_code=409,
            detail="An admin already exists. Log in and use POST /api/admin/users to add more users.",
        )

    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    existing_email = session.exec(
        select(User).where(User.corporate_email == req.corporate_email)
    ).first()
    if existing_email:
        raise HTTPException(status_code=409, detail="A user with this email already exists.")

    admin = User(
        corporate_email=req.corporate_email,
        name=req.name,
        password_hash=hash_password(req.password),
        role="admin",
    )
    session.add(admin)
    session.commit()
    session.refresh(admin)

    return {
        "message": "Admin account created. You can now log in.",
        "user_id": admin.id,
        "email": admin.corporate_email,
    }
