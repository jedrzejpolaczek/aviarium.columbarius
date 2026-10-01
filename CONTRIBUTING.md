# Contributing to aviarium.columbarius

Thank you for your interest in contributing. This document covers prerequisites, local setup, quality checks, and the PR workflow.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — the project's package and environment manager
- GNU Make (optional but recommended; all commands have `uv run` equivalents)

## Local Setup

```bash
git clone https://github.com/jedrzejpolaczek/aviarium.columbarius
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
| `make test` | pytest — two invocations, see [ADR-026](docs/adr/ADR-026-isolate-mlflow-tracking-tests.md) |
| `make docs-check` | file paths cited in docs exist; no local paths in notebook outputs |

The frontend has its own suite:

```bash
cd frontend
npm ci
npm test
```

Without Make, run each command from the `Makefile` through `uv run`, e.g. `uv run pytest --ignore=tests/ml/training/test_tracking.py` followed by `uv run pytest tests/ml/training/test_tracking.py`.

## Branching Conventions

- Branch from `dev` and open your PR against `dev`. `main` receives only release merges from `dev`.
- Use short, lowercase, hyphen-separated names: `feat/gold-features`, `fix/silver-join-null`, `chore/update-deps`.
- Keep branches focused — one logical change per PR.

## Commit Messages

The history follows [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, where `type` is one of `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`, and `scope` is optional (`silver`, `alerts`, `storage`...).

Keep the summary in the imperative and under ~72 characters. For anything non-trivial, use the body to explain *why*: the failure that prompted the change, what was checked, and what was deliberately left out. `git log` is part of this project's documentation.

## Architecture Decisions

A change that picks between real alternatives (a library, a storage layout, a failure policy) gets an ADR in `docs/adr/`. Copy the structure of a recent one (`ADR-031` onward): Status, Context, Decision, Alternatives considered, Consequences. Number it next in sequence and add it to the ADR table in `README.md`.

When a change makes an existing ADR inaccurate, do not rewrite its history: set its Status to `Amended` or `Superseded by ADR-0XX` and add a dated note explaining what changed.

## Changelog

Add a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for any user-visible change, in the matching *Added / Changed / Fixed* section.

## Pull Request Checklist

- [ ] `make check` passes locally
- [ ] New behaviour is covered by tests
- [ ] Relevant ADRs updated or a new ADR added if an architectural decision was made
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR description explains *why* the change is needed, not just what it does

## Reporting Bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) when opening an issue. Security problems go by email instead — see [SECURITY.md](SECURITY.md).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful and constructive.
