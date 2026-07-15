.DEFAULT_GOAL := help
.PHONY: help install install-ocr up down logs run compile lint format typecheck test test-int crawl ingest reindex clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev dependencies
	poetry install --with dev

install-ocr: ## Also install the optional OCR extras (needs tesseract + poppler)
	poetry install --with dev,ocr

up: ## Start postgres + redis + qdrant + api via docker compose
	docker compose up --build -d

down: ## Stop the docker compose stack
	docker compose down

logs: ## Tail api logs
	docker compose logs -f api

run: ## Run the API locally with autoreload
	poetry run uvicorn dl_rag.main:app --reload --host 0.0.0.0 --port 8000

compile: ## Fast syntax check of the whole package
	poetry run python -m compileall -q src

lint: ## Ruff lint
	poetry run ruff check src tests

format: ## Ruff format
	poetry run ruff format src tests

typecheck: ## mypy strict
	poetry run mypy src

test: ## Run unit tests
	poetry run pytest -q -m "not integration"

test-int: ## Run integration tests (requires live services)
	poetry run pytest -q -m integration

crawl: ## Discover + crawl the archive (see scripts/run_crawl.py --help)
	poetry run dl-crawl

ingest: ## Run the full ingestion pipeline (see scripts/run_ingest.py --help)
	poetry run dl-ingest

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ dist build
