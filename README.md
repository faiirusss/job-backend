# Lamarin AI Backend

**Lamarin AI** is an Indonesian-market job-search assistant. The user uploads a CV and
types a natural-language query; the backend scrapes job portals (currently Glints + LinkedIn),
structures each job description with an LLM, embeds and scores every job against the CV, and streams
results live to the frontend over a WebSocket.

This repository is the **backend** (FastAPI + Python 3.11). The frontend lives in a separate repository.

## Stack

- FastAPI + Uvicorn, async SQLAlchemy 2.0 (`asyncpg`) on Postgres + `pgvector`
- Playwright (headless Chromium) scrapers with a registry + stealth
- `sentence-transformers` (multilingual MiniLM, 384-dim) for embeddings
- Pluggable LLM behind a Protocol — `FakeLLM` (offline default) or `GeminiLLM`
- In-memory `EventBus` streaming pipeline progress over a WebSocket

## Run with Docker (recommended)

The whole stack (Postgres + backend) runs in containers. The `override` file is applied automatically
for dev (hot-reload + `FakeLLM`).

```bash
cp .env.example .env          # edit if you want real Gemini
docker network create jhai-net   # shared network (once); frontend repo joins the same net
docker compose up              # dev: postgres + backend with --reload on :8000
```

Production image (multi-stage, CPU-only Torch, non-root, migrates on boot):

```bash
docker compose -f docker-compose.yml up --build
```

## Run locally (without Docker)

All commands go through the `Makefile`, which wraps a `uv`-managed `.venv`:

```bash
make install        # uv venv + editable install + playwright install chromium
cp .env.example .env
make up             # start the Postgres + pgvector container
make migrate        # alembic upgrade head
make dev            # uvicorn app.main:app --reload on :8000
```

## Tests & quality

```bash
make test           # pytest (live-network tests excluded by default)
make test-live      # the Glints live smoke test only
make lint           # ruff check + mypy app
make fmt            # ruff format
```

## Configuration

Copy `.env.example` to `.env`. Key settings:

- `USE_FAKE_LLM` (default `true`) — set `false` + `GEMINI_API_KEY` for the real LLM path.
- `DATABASE_URL` — under Docker Compose the host is overridden to `postgres`.
- `GLINTS_EMAIL` / `GLINTS_PASSWORD` — optional service account to scrape past Glints' page-2 login wall.
