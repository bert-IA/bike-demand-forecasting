# ADR-0001: Default production target is Scenario A

- Status: Accepted
- Date: 2026-10-15
- Deciders: project owner

## Context

The final production server is not yet chosen. Deploying a model that does not
fit the real hardware would invalidate the benchmark: a champion that cannot run
in production is not a champion. The budgets of each hardware scenario are
defined in `configs/deployment_scenarios.yaml`.

## Decision

Until the real server is known, every model is designed and evaluated against
**Scenario A** (2 vCPU, 4 GB RAM, no GPU at inference) as the default
production target. Scenarios B and C are evaluated as upgrade options, not as
assumptions.

The default is revisited by a new ADR once the hosting is chosen (before
milestone M5).

## Alternatives considered

- **Assume Scenario B or C by default**: higher accuracy ceiling, but a late
  discovery that the server is smaller would force a costly rework.
- **Wait until the server is chosen**: blocks the model benchmark on a decision
  that does not depend on the models.

## Consequences

- Anything that fits Scenario A also fits B and C, so the default is safe.
- Some accuracy may be left on the table; this cost is measured explicitly as
  the "price of the constraint" (see ADR-0002) and presented to the client.
- Inference-time GPU use is forbidden in A and B; GPU is for training only.
- Budgets are frozen before champions are designated (M2); changing them
  requires an amended ADR.
