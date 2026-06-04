# syntax=docker/dockerfile:1

# ---- base: runtime deps in an isolated venv outside the workdir ----
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    HF_HOME=/opt/hf-cache \
    PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only dependency manifests first for layer caching.
COPY pyproject.toml uv.lock ./
# Install runtime dependencies (not the project itself) into /opt/venv.
# Pin Python 3.11 (uv auto-downloads a managed build): the Playwright base image
# ships 3.10, which is below the project's requires-python, and letting uv pick
# the newest 3.x pulls a version without prebuilt asyncpg/greenlet wheels (the
# image has no C compiler, so a source build fails).
RUN uv sync --frozen --no-install-project --no-dev --python 3.11

# ---- dev: adds dev tooling; source is bind-mounted at runtime ----
FROM base AS dev
RUN uv sync --frozen --no-install-project --python 3.11
EXPOSE 8000
# Command is supplied by docker-compose.override.yml (runs migrations + reload).

# ---- prod: copies source, runs as non-root, migrates then serves ----
FROM base AS prod
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh \
    && useradd --create-home appuser \
    && mkdir -p /opt/hf-cache /app/data \
    && chown -R appuser:appuser /opt/hf-cache /app \
    && chmod -R a+rX /opt/venv /opt/uv-python
USER appuser
EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
