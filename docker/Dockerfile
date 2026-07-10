# Multi-stage build: stage 1 installs deps, stage 2 is the lean production image.
#
# Why python:3.13-slim and not alpine:
#   alpine lacks the C libraries required by numpy/scipy/lightgbm.
#   slim = full CPython without unnecessary system tools (~250 MB final image).
#
# Usage:
#   docker build -t mtg-price-api -f docker/Dockerfile .
#   docker run -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/mlruns:/app/mlruns \
#     -e MODEL_RUN_ID=<run_id> \
#     mtg-price-api

# ── Stage 1: install dependencies ───────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# uv is faster and deterministic (ADR-011)
RUN pip install --no-cache-dir uv

# Copy only the lock files so the dependency layer is cached on code-only changes
COPY pyproject.toml uv.lock ./

# Install production dependencies (no dev group: no pytest/jupyterlab/mypy)
RUN uv sync --frozen --no-dev

# ── Stage 2: production image ────────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# curl is needed for the Docker healthcheck command
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --system app && useradd --system --gid app --no-create-home app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ ./src/
COPY app/ ./app/

# logs/ must be writable by the non-root app user — this is where
# src.monitoring.alerts.send_alert writes logs/alerts.jsonl on a degraded
# startup (see app/main.py lifespan). Without this, the write fails with
# a silently-swallowed PermissionError (send_alert never raises) and the
# alert is lost. Mount a volume over it (see docker-compose.yml) so the
# file also survives container restarts.
RUN mkdir -p /app/logs && chown app:app /app/logs

# Add venv to PATH so uvicorn / python resolve from .venv
ENV PATH="/app/.venv/bin:$PATH"

USER app

EXPOSE 8000

# data/ and mlruns/ are mounted as volumes — see docker-compose.yml
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
