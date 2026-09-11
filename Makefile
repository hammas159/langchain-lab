.PHONY: install test lint bench web shots all

install:
	uv sync --extra dev

test:
	uv run pytest -q -m "not live"

test-live:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

bench:
	uv run python -m projects.p01_structured_output.benchmark --full

web:
	uv run uvicorn projects.p01_structured_output.web:app --reload --port 8101

shots:
	npm install
	node scripts/shoot.mjs --url http://127.0.0.1:8101 --prefix p01

models:
	ollama pull qwen2.5:3b-instruct
	ollama pull nomic-embed-text

all: install lint test
