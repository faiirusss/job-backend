.PHONY: up down migrate migration dev test test-live lint fmt install

# Isolate Playwright's browser download from the shared ~/.cache/ms-playwright.
# Otherwise another tool installing a newer Playwright can delete the chromium
# build this backend pins, breaking scraping. Make's `export` passes it to every
# recipe (install downloads here; dev/test launch from here). Docker is unaffected
# (browsers come from the Playwright base image).
export PLAYWRIGHT_BROWSERS_PATH := $(CURDIR)/.ms-playwright

install:
	uv venv --python 3.11
	. .venv/bin/activate && uv pip install -e ".[dev]" && playwright install chromium

up:
	docker compose up -d postgres

down:
	docker compose down

migrate:
	. .venv/bin/activate && alembic upgrade head

migration:
	. .venv/bin/activate && alembic revision --autogenerate -m "$(MSG)"

dev:
	. .venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	. .venv/bin/activate && pytest

test-live:
	. .venv/bin/activate && pytest -m live

lint:
	. .venv/bin/activate && ruff check . && mypy app

fmt:
	. .venv/bin/activate && ruff format .
