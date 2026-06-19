# ADR-011: uv as the Python Package Manager

## Context

The project requires a reproducible Python environment across developer machines and
CI. Several package managers were evaluated:

**Option A — pip + requirements.txt:** Standard but no lockfile by default; `pip install`
produces non-deterministic environments unless pinned manually.

**Option B — Poetry:** Mature, widely adopted, deterministic lockfile (`poetry.lock`),
dependency groups. Slow resolver on complex dependency trees.

**Option C — uv (Astral):** Rust-based resolver and installer, drop-in compatible with
`pyproject.toml` PEP 517/518 standards, deterministic `uv.lock`, built-in dependency
group support, significantly faster than pip or Poetry.

## Decision

Use **uv** as the package manager.

- `pyproject.toml` declares all dependencies and dev dependency groups following
  PEP 517 standards, keeping the project portable.
- `uv.lock` is committed to version control to guarantee reproducible installs.
- Development dependencies (ruff, mypy, pytest, pandas, numpy, jupyter) are declared
  in a separate `[dependency-groups]` section so production installs stay lean.
- CI installs the environment with `uv sync` for deterministic, cached builds.

## Consequences

### Positive
- Install times are significantly faster than pip or Poetry, improving local setup
  and CI iteration speed.
- The lockfile is deterministic: every developer and every CI run resolves to the
  same exact package versions.
- `pyproject.toml` format is PEP-standard — migrating to a different tool later
  requires no restructuring of the dependency declarations.
- Dependency groups separate dev tools from runtime dependencies cleanly.

### Negative
- uv is a relatively new tool (first released 2024). Long-term maintenance and
  ecosystem support are less proven than pip or Poetry.
- Some existing tutorials, Docker base images, and CI templates assume pip or Poetry;
  minor adaptation is needed.

### Neutral
- The `uv` binary must be installed separately before `uv sync` can run. This is
  a one-line bootstrap step (`pip install uv` or OS package manager).

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| pip + requirements.txt | No native lockfile; manual pinning is error-prone and noisy to maintain |
| Poetry | Slower resolver; `pyproject.toml` format is Poetry-specific in places, reducing portability |
| Conda | Heavyweight; designed for scientific environments with C extensions, not a pure-Python pipeline |
