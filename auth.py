import os
from datetime import datetime, timedelta

import bcrypt
from jose import jwt, JWTError
from fastapi import HTTPException, Request, Depends

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30
COOKIE_NAME = "access_token"

# Maps token string → user_id. Allows logout/session invalidation without waiting for JWT expiry.
active_tokens: dict[str, int] = {}


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_token(user_id: int, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    active_tokens[token] = user_id
    return token


def invalidate_token(token: str) -> None:
    active_tokens.pop(token, None)


def get_current_user(request: Request) -> dict:
    access_token = request.cookies.get(COOKIE_NAME)
    if not access_token or access_token not in active_tokens:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid. Please log in again.",
        )
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "user_id": int(payload["sub"]),
            "role": payload["role"],
            "token": access_token,
        }
    except JWTError:
        active_tokens.pop(access_token, None)
        raise HTTPException(status_code=401, detail="Invalid token. Please log in again.")


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user
