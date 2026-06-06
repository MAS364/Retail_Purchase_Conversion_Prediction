# ── Build stage: install dependencies ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Production stage ───────────────────────────────────────────────────────────
FROM python:3.12-slim

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and models
COPY app.py .
COPY src/ src/
COPY models/ models/

# Switch to non-root user
USER appuser

# Cloud Run sends SIGTERM; uvicorn handles graceful shutdown
# Using gunicorn with uvicorn workers for production
CMD exec gunicorn app:app \
    --bind :$PORT \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile - \
    --error-logfile -
