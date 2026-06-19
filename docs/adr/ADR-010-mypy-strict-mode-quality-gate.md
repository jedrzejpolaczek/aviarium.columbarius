# ADR-010: mypy Strict Mode as a Hard Quality Gate

## Context

The pipeline processes hundreds of thousands of records through a chain of
transformations. Type errors (passing a `list[str]` where `list[ScryfallCard]` is
expected, returning `None` from a function declared as `list[T]`) can cause silent
data corruption or cryptic runtime failures far from the error site.

Three approaches were considered for type safety:

**Option A — No type annotations:** Pure duck typing. Fast to write, impossible to
statically verify.

**Option B — Gradual typing:** Add annotations where convenient; run mypy in default
(lenient) mode. Unannotated code is silently ignored.

**Option C — Strict typing from day one:** Full annotations required everywhere;
mypy strict mode (`strict = true`) enforced as a pre-push gate.

## Decision

Enable **mypy strict mode** for the entire codebase, enforced by:

- `pyproject.toml`: `strict = true` under `[tool.mypy]`, with the pydantic plugin
  enabled for model-aware type checking.
- `scripts/pre-push`: A pre-push hook that runs `mypy`, `ruff` lint, `ruff` format,
  and `pytest` before any push is accepted. A failure in any check blocks the push.
- CI (`github/workflows/ci.yml`): The same checks run on every push and pull request.

No `# type: ignore` comments without justification; all type issues must be resolved.

## Consequences

### Positive
- Type errors in pipeline logic are caught at development time, not in a 2 AM
  production run processing 400k records.
- Function signatures serve as machine-checked documentation — callers know exactly
  what a function accepts and returns.
- Pydantic plugin integration means model field types flow through mypy correctly
  (e.g. `ScryfallCard.prices` is typed as `ScryfallPrices | None`, not `Any`).
- Refactors are safer: renaming a field or changing a return type produces a
  compile-time error list, not a runtime surprise.

### Negative
- Initial setup cost: every function, variable, and return type must be annotated.
- Some pandas and DuckDB operations return `Any` at the type level; these require
  explicit casts or `# type: ignore` with justification.
- Strict mode rejects a class of legitimate Python patterns (e.g. untyped decorators
  from third-party libraries), which require workarounds.

### Neutral
- mypy and ruff run in under 10 seconds on this codebase, so the gate adds negligible
  friction for the guarantees it provides.

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| No type annotations | Cannot catch contract violations between pipeline stages statically |
| Gradual / lenient typing | Unannotated code is silently ignored; the safety guarantee is incomplete and degrades as new code is added without annotations |
| Runtime-only validation (Pydantic at boundaries only) | Catches external data errors but not internal logic errors between functions |
