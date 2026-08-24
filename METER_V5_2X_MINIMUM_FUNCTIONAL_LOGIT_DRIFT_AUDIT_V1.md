# Meter V5-2X — Minimum Functional Logit Drift Audit V1

## Decision

V5-2W verified that both 2-AI and 3-AI have at least one shared 64D head
weight which, at the frozen bias and frozen threshold, classifies every V5
TRAIN example correctly and preserves every historical TRAIN decision the
frozen specialist gets right. This rules out a TRAIN-surface claim that no
shared head exists. It does not identify a safe repair: V5-2W minimized a zero
objective, so neither returned witness norm was minimal.

V5-2X does not authorize training. It measures the smallest functional change
that any such shared head must create on historical TRAIN.

## Exact prerequisite evidence

The audit binds to the completed V5-2W execution:

- implementation HEAD: `bdd82204182e3d5043a64907de7e0f0394089a20`;
- report SHA256: `0fdcff6a9114eec08a2f3c512de1336cf4af96be4ab82cf168d14fa2e77f4095`;
- execution-envelope SHA256:
  `4b86e81547d9d22aee60353e57eac7d28fd9ab8ed6988f7ec629241c6a723d54`.

## One preregistered objective

For each specialist, write the diagnostic weight as `w = w0 + delta`, where
`w0` is the frozen source weight. Bias and runtime threshold stay frozen.
The linear program minimizes one scalar:

`t = max over historical TRAIN examples of abs(feature dot delta)`.

The hard constraints require:

1. every V5 TRAIN example to have the correct decision with signed margin at
   least `1e-4` logit;
2. every historical TRAIN example that the frozen head gets right to remain
   correct with signed margin at least `1e-4` logit;
3. every historical TRAIN logit change to lie between `-t` and `+t`.

The objective is a functional decision-safety quantity. Weight L1/L2 norm and
angle are descriptive only and are explicitly **not** claimed to be minimal.
There is no second lexicographic solve.

## Solver and verification

The stage uses one pinned SciPy `1.18.0` `linprog` configuration with
`highs-ds`, presolve disabled, and primal/dual feasibility tolerances `1e-9`.
There is no solver sweep, fallback, threshold search, or bias search.

A returned finite witness is independently checked for all V5 decision
constraints, all frozen-correct historical decision constraints, all
historical logit-drift bounds, and the identity
`new_logit - frozen_logit = feature dot delta`. Only then is it labelled
`WITNESS_VERIFIED`.

HiGHS status is reported conservatively. Optimal status is
`SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF`; status 2 is
`SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF`. No formal-proof claim is made.

## Safety boundary

The diagnostic LP temporarily fits a delta vector in memory, but never emits
or persists its values, writes a checkpoint, mutates a model, or selects a
repair. There is no model training, autograd, backward, optimizer step,
threshold/bias change, architecture change, or production promotion.

Only aggregate TRAIN statistics are emitted. Historical validation,
First-30, V5 validation, and FINAL_HOLDOUT stay closed. 4-AI stays frozen.
