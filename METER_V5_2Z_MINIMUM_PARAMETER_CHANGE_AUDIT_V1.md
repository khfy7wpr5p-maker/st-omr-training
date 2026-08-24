# Meter V5-2Z — Minimum Parameter Change Audit V1

## Decision

V5-2Y proved that parameter movement remains very large when the V5-2X
minimum historical-logit-drift objective is held first in lexicographic
order. That rules out a purely arbitrary null-space explanation inside the
near-optimal functional-drift face. It does not prove that the hard decision
constraints themselves require a large head change.

V5-2Z removes the historical-logit-drift cap and measures the unconditional
minimum maximum absolute head-weight change. It remains a TRAIN-only
diagnostic and does not authorize training.

## Exact prerequisite evidence

The audit binds to the completed V5-2Y execution:

- implementation HEAD: `18e23ed2c25e50db03f41db70259db3fd74e224a`;
- report SHA256:
  `d9f7133d02a0875f09a79e0ecb53a5ae2f510e92164d14b38e171f1042655913`;
- execution-envelope SHA256:
  `b56ee42a865d61c4a19e5bc6038f5b9094b5d91b6135be2b39e5f8eecce43d10`.

The V5 slot-manifest hash, frozen bias, frozen threshold, record counts, and
frozen-correct historical surface are checked again before solving. The
verified V5-2Y conditional minimum is used only as a consistency upper bound.

## One unconditional objective

For each specialist, write the diagnostic weight as `w = w0 + delta`. Bias
and runtime threshold remain frozen. One linear program minimizes:

`r = max over 64 head components of abs(delta)`.

Hard constraints require:

1. every V5 TRAIN example to have the correct decision with signed margin at
   least `1e-4`;
2. every historical TRAIN example that the frozen head gets right to remain
   correct with signed margin at least `1e-4`;
3. every component of `delta` to lie between `-r` and `+r`.

Historical TRAIN logit drift is not constrained or optimized. It is reported
only as descriptive aggregate evidence. Weight L1/L2 norm and angle are also
descriptive. There is no solver sweep, fallback, threshold search, or bias
search.

Because V5-2Y solved the same decision constraints plus an additional drift
cap, the V5-2Z minimum may not exceed V5-2Y's conditional minimum beyond the
fixed `1e-7` witness tolerance. A violation is an evidence conflict.

## Solver and independent verification

The stage uses pinned SciPy `1.18.0` `linprog` with `highs-ds`, presolve
disabled, and primal/dual feasibility tolerances `1e-9`.

A finite witness is independently checked for every V5 constraint, every
frozen-correct historical constraint, every parameter bound, the functional
identity `new_logit - frozen_logit = feature dot delta`, and equality between
the solver objective and the independently recomputed maximum absolute
component. Only then is it labelled `WITNESS_VERIFIED`.

HiGHS optimality is reported as
`SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF`. Infeasibility is reported as
`SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF`. Neither is promoted to a
formal mathematical proof.

## Interpretation boundary

A small V5-2Z minimum would show that V5-2Y's large weights were caused by
prioritizing minimum historical logit drift. A still-large minimum would show
that even the hard shared-head decision constraints are ill-conditioned on
TRAIN at the frozen bias and threshold.

Neither outcome proves validation retention, representation failure, or
deployment safety. No parameter-size pass/fail threshold is selected here,
and no repair is selected.

## Safety boundary

The diagnostic witness remains in memory and its values are never emitted or
persisted. No checkpoint is written and no model parameter is mutated. There
is no model training, autograd, backward, optimizer step, threshold/bias
change, architecture change, or production promotion.

Only aggregate TRAIN statistics are emitted. Historical validation,
First-30, V5 validation, and FINAL_HOLDOUT remain closed. 4-AI remains frozen.
