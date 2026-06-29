import os
from datetime import datetime, timedelta

import bcrypt
from jose import jwt, JWTError
from fastapi import HTTPException, Request, Depends

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30
COOKIE_NAME = "access_token"

# True in production (HTTPS), False for local dev (HTTP)
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Scope cookies to the parent domain in production so subdomains share them
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", ".discoverbenix.com" if IS_PRODUCTION else None)

# Blocklist of tokens that were explicitly logged out.
# Only used to reject tokens before their JWT expiry after logout.
revoked_tokens: set[str] = set()


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
    return token


def invalidate_token(token: str) -> None:
    """Add the token to the revoked blocklist so it can't be reused after logout."""
    revoked_tokens.add(token)


def get_current_user(request: Request) -> dict:
    access_token = request.cookies.get(COOKIE_NAME)

    # No cookie at all
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid. Please log in again.",
        )

    # Token was explicitly logged out
    if access_token in revoked_tokens:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid. Please log in again.",
        )

    # Validate the JWT signature and expiry
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "user_id": int(payload["sub"]),
            "role": payload["role"],
            "token": access_token,
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token. Please log in again.")


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user
