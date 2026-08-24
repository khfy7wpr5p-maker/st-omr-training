# Meter V5-2W — Shared Affine Head Feasibility Audit V1

## Decision

V5-2V proved that the V5-2T 15-degree weight-space bound was not a functional
retention guarantee. Small 2-AI and 3-AI weight changes created large same-sign
logit shifts for both V5 positives and historical negatives. V5-2W therefore
does not authorize another repair run. It first asks whether one shared linear
head can satisfy the required TRAIN decisions at all.

## Exact prerequisite evidence

The audit binds to the completed V5-2V execution:

- implementation HEAD: `b1db7923e91cec534fcfd95afad7f8b4ef87607b`;
- report SHA256: `1ecc6b6600e0f01c1eeb4e8530d2184800dd470d2b66344c52a28a79d170bd3a`;
- execution-envelope SHA256:
  `f8c87b5ecec00f5a4e2cbaf5f1f07bb599f85dac41af8b989686c6f33f03ca4d`.

## Constraint surface

Only already-open frozen V5 TRAIN and historical TRAIN features are used. For
each of 2-AI and 3-AI, the joint constraint surface contains:

1. every V5 TRAIN example with its ground-truth decision;
2. every historical TRAIN example that the frozen head already classifies
   correctly, with that same ground-truth decision.

Historical examples already misclassified by the frozen head are unconstrained.
Preserving all frozen-correct historical decisions is a sufficient zero-drop
retention condition; it is deliberately stronger than the later 0.005 metric
drop allowance.

## Two separate feasibility questions

### Frozen runtime feasibility

Only a diagnostic 64D weight vector is free. The source `head.bias` and frozen
runtime threshold remain fixed. Every signed decision margin must be at least
`1e-4` logit. A returned witness is independently recomputed and accepted only
when every constraint is within `1e-7` of the preregistered margin.

### Free affine feasibility

A diagnostic 64D weight plus effective intercept is free, with normalized
signed margin at least `1`. This asks whether any affine separator exists in
the frozen feature representation, independent of the current calibration.
The diagnostic intercept is never selected for runtime use.

## Solver and claim discipline

Both questions use one pinned SciPy `1.18.0` `linprog` configuration with
`highs-ds`, presolve disabled, and primal/dual feasibility tolerances `1e-9`.
There is no solver sweep, fallback, threshold search, or bias search.

SciPy documents status `2` as a problem that *appears* infeasible. Therefore a
status-2 result is reported exactly as
`SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF`. It is never promoted to a formal
mathematical proof. Only a finite witness that passes independent residual
verification may be labelled `WITNESS_VERIFIED`.

## Safety boundary

The linear program fits temporary diagnostic witnesses in memory, and the
report says so explicitly. It performs no model training, autograd, backward,
optimizer step, model mutation, checkpoint write, threshold tuning, bias
selection, or repair selection. Witness values are neither emitted nor saved.

Historical validation examples, First-30, V5 validation, and FINAL_HOLDOUT stay
closed. 4-AI stays frozen and production promotion remains false.
