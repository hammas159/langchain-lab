.PHONY: install test test-live lint bench bench01 bench02 web01 web02 shots models all

install:
	uv sync --extra dev

test:
	uv run pytest -q -m "not live"

test-live:
	uv run pytest -q

lint:
	uv run ruff check shared projects tests scripts
	uv run ruff format --check shared projects tests

bench: bench01 bench02

bench01:
	uv run python -m projects.p01_structured_output.benchmark --full

bench02:
	uv run python -m projects.p02_retrieval_absences.benchmark

web01:
	uv run uvicorn projects.p01_structured_output.web:app --reload --port 8101

web02:
	uv run uvicorn projects.p02_retrieval_absences.web:app --reload --port 8102

# Needs the matching server already running on the port above.
shots:
	npm install
	node scripts/shoot.mjs --project p01
	node scripts/shoot.mjs --project p02

models:
	ollama pull qwen2.5:3b-instruct
	ollama pull nomic-embed-text

all: install lint test
