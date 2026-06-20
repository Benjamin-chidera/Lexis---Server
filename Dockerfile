FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# First, install the dependencies into a virtual environment
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Then, copy the rest of the project and install it
COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

# Use uvicorn directly — "main:app" targets the socketio.ASGIApp in main.py
# "fastapi run" doesn't reliably serve custom ASGI wrappers
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
