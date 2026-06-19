.PHONY: install install-hooks lint format format-check type-check test coverage check pipeline train

install:
	uv sync --all-groups

install-hooks:
	git config core.hooksPath scripts

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

type-check:
	uv run mypy

test:
	uv run pytest

coverage:
	uv run pytest --cov=src --cov=app --cov-report=term-missing

check: lint format-check type-check test

pipeline:
	uv run python -m scripts.run_pipeline

train:
	uv run python -m scripts.train_model
