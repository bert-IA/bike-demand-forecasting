# ADR-0002: Two-track champion strategy and frozen selection rule

- Status: Accepted
- Date: 2026-10-15
- Deciders: project owner

## Context

"Best model" has two different meanings: the most accurate model we can train,
and the most accurate model we can actually deploy under hardware constraints.
Mixing them hides the cost of deployment constraints. In addition, selecting a
winner with a rule chosen after seeing the results would bias the benchmark.

## Decision

1. **Two champions are designated**:
   - the **research champion**: best accuracy with no resource constraint
     (GPU allowed for training);
   - one **production champion per scenario** (A, B, C): best accuracy among
     models that satisfy the budgets in `configs/deployment_scenarios.yaml`.
2. **Price of the constraint**: `delta = MASE(production) - MASE(research)`,
   reported per scenario. It is the central quantified result of the study.
3. **Selection rule, frozen before any experiment**, applied in this order:
   1. mean MASE (WAPE as a secondary check) in rolling-origin backtesting;
   2. empirical coverage of prediction intervals (80% and 95%);
   3. cost: inference latency and peak memory;
   4. maintainability.
4. **Complexity must be earned**: a complex model is retained only if it beats
   the global gradient boosting reference with a statistically significant gap
   (Diebold-Mariano test).
5. **Materiality thresholds**: a MASE gain below 0.02 is considered negligible;
   a gain above 0.05 justifies the higher-cost option.
6. **Budgets are frozen before champions are designated (M2)**. A model that
   exceeds a budget is rejected, unless the overrun is below 20% and is
   documented as an explicit waiver.
7. Every model is compared to the baselines (seasonal naive, ETS); a model
   with MASE >= 1 does not justify its complexity.
8. Results are tracked with MLflow aliases `champion-research` and
   `champion-prod-<scenario>`.

## Alternatives considered

- **Single champion**: simpler, but hides what the hardware constraint costs.
- **Choose the rule after the results**: rejected, it invites selection bias.
- **Accuracy-only comparison**: ignores latency and memory, which decide
  whether a model can run in production.

## Consequences

- The study produces quantified trade-offs that can be presented to a client
  (options O1-O5 in the project plan).
- Every model needs a deployment sheet (latency, memory, image size, training
  time) in addition to accuracy metrics.
- Changing the rule or the budgets afterwards requires a new ADR that
  supersedes this one.
- More work: two selection passes and a Pareto analysis (accuracy vs cost).
