# Contributing to aviarium.columbarius

Thank you for your interest in contributing. This document covers prerequisites, local setup, quality checks, and the PR workflow.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — the project's package and environment manager
- GNU Make (optional but recommended; all commands have `uv run` equivalents)

## Local Setup

```bash
git clone https://github.com/jpolaczek/aviarium.columbarius
cd aviarium.columbarius

# Create the virtual environment and install all dependencies (including dev)
make install

# Install git hooks (runs checks before every push)
make install-hooks
```

## Running Quality Checks

Before opening a PR, all checks must pass:

```bash
make check   # lint + format + type-check + test
```

Individual checks:

| Command | Tool |
|---|---|
| `make lint` | ruff check |
| `make format` | ruff format |
| `make type-check` | mypy (strict) |
| `make test` | pytest |

## Branching Conventions

- Branch from `main`.
- Use short, lowercase, hyphen-separated names: `feat/gold-features`, `fix/silver-join-null`, `chore/update-deps`.
- Keep branches focused — one logical change per PR.

## Pull Request Checklist

- [ ] `make check` passes locally
- [ ] New behaviour is covered by tests
- [ ] Relevant ADRs updated or a new ADR added if an architectural decision was made
- [ ] PR description explains *why* the change is needed, not just what it does

## Reporting Bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) when opening an issue.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful and constructive.
