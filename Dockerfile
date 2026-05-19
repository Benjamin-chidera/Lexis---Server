FROM python:3.12 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv .venv
COPY pyproject.toml ./
RUN .venv/bin/pip install .
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

# Explicitly target main.py so the socketio ASGI app (main:app) is served
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8080"]
