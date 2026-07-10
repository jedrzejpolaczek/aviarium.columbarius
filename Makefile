.PHONY: install install-hooks lint format format-check type-check test coverage check pipeline train monitor backup

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
	uv run mypy .

test:
	uv run pytest --ignore=tests/ml/training/test_tracking.py; s1=$$?; \
	uv run pytest tests/ml/training/test_tracking.py; s2=$$?; \
	[ $$s1 -eq 0 ] && [ $$s2 -eq 0 ]

coverage:
	uv run pytest --cov=src --cov=app --cov-report=term-missing

check: lint format-check type-check test

pipeline:
	uv run python -m scripts.run_pipeline

train:
	uv run python -m scripts.train_model

monitor:
	uv run python -m scripts.check_and_retrain

backup:
	uv run python -m scripts.backup_data
