# Definition of Done

A pull request is mergeable only when all applicable items are satisfied.

## Always
- [ ] CI is green (ruff, format, mypy strict, pytest).
- [ ] The change is small and has one purpose.
- [ ] The PR description explains what, why and how to review.
- [ ] No secret, raw data or large file is committed.

## Code changes
- [ ] New behaviour is covered by tests.
- [ ] Public functions are fully typed.
- [ ] Dependencies are added with `uv add` and `uv.lock` is updated.

## Modelling changes
- [ ] The evaluation follows the frozen protocol (ADR-0002), with no temporal leakage.
- [ ] Results are compared to the baselines and logged in MLflow.
- [ ] A deployment sheet (latency, memory, image size, training time) is provided.

## Structural decisions
- [ ] An ADR is added or superseded when a decision affects architecture,
      budgets, the stack or the selection rule.
- [ ] Documentation affected by the change is updated.
