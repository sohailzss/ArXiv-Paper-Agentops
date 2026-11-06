# Shortcuts. On Windows, run these from Git Bash, or copy the command after the colon into PowerShell.
COMPOSE = docker compose

.PHONY: install check up down logs test lint format api

install:      ## install Python deps (plus Docling for notebooks)
	uv sync --group ingest

check:        ## verify cloud services and keys
	uv run python scripts/test_connections.py

up:           ## build and start all containers
	$(COMPOSE) up --build -d

down:         ## stop containers (data volumes are kept)
	$(COMPOSE) down

logs:         ## follow logs for all containers
	$(COMPOSE) logs -f --tail=100

api:          ## run the API on your machine with hot reload (OpenSearch still in Docker)
	uv run fastapi dev src/main.py

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff check --fix . && uv run ruff format .
