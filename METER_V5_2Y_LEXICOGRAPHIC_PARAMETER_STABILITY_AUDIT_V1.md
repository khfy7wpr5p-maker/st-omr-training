# Meter V5-2Y — Lexicographic Parameter Stability Audit V1

## Decision

V5-2X verified on sealed TRAIN surfaces that both 2-AI and 3-AI can fix all
V5 decisions while preserving every historical decision their frozen heads
get right. It also minimized the worst historical TRAIN logit drift. The
returned witnesses were not safe candidates because weight geometry was not
part of that objective; 3-AI in particular had a very large diagnostic norm.

V5-2Y does not authorize training. It asks whether that large parameter
movement was an arbitrary LP/null-space choice or is still required after a
fixed, safety-oriented secondary objective.

## Exact prerequisite evidence

The audit binds to the completed V5-2X execution:

- implementation HEAD: `c276517e07ee129d80b53bb906ead06c3094f0af`;
- report SHA256:
  `46dbba37c1ae88e6212afa1a1fef92ecfb92935a21425437c494c9a320aafc51`;
- execution-envelope SHA256:
  `30aee74467349fd6649ea3c6bc27d2f7f669c7a8341da7ef131d2cafda171846`.

The V5 slot-manifest hash, frozen bias, frozen threshold, record counts, and
frozen-correct historical surface are checked again before solving.

## One fixed secondary objective

For each specialist, write the diagnostic weight as `w = w0 + delta`. Bias
and runtime threshold remain frozen. V5-2Y does not re-optimize V5-2X's
primary objective. It reads V5-2X's independently recomputed minimum maximum
historical logit drift and creates one cap:

`primary_cap = exact_v5_2x_recomputed_drift + 1e-6`.

The absolute `1e-6` allowance is preregistered numerical slack, not a tuned
hyperparameter. Under that cap, one linear program minimizes:

`r = max over 64 head components of abs(delta)`.

Hard constraints require:

1. every V5 TRAIN decision to be correct with signed margin at least `1e-4`;
2. every historical TRAIN decision the frozen head gets right to remain
   correct with signed margin at least `1e-4`;
3. every historical TRAIN logit change to remain inside the fixed primary
   cap;
4. every component of `delta` to lie between `-r` and `+r`.

Only `r` is minimized. Weight L1/L2 norm and angle are descriptive. There is
no solver sweep, fallback, alternative slack, threshold search, or bias
search.

## Solver and independent verification

The stage uses pinned SciPy `1.18.0` `linprog` with `highs-ds`, presolve
disabled, and primal/dual feasibility tolerances `1e-9`.

A finite witness is checked independently for every V5 constraint, every
frozen-correct historical constraint, the fixed historical drift cap, every
parameter bound, and the functional identity
`new_logit - frozen_logit = feature dot delta`. The reported objective must
also equal the independently recomputed maximum absolute component. Only
then is it labelled `WITNESS_VERIFIED`.

The recomputed drift may not undercut the exact V5-2X primary optimum by more
than `1e-7`. A larger improvement would contradict the bound that this stage
claims to hold fixed, so it is reported as an evidence gap instead of being
silently accepted.

HiGHS optimality is reported as
`SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF`. Infeasibility is reported as
`SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF`. Neither is promoted to a
formal mathematical proof.

## Interpretation boundary

A small minimum `r` would show that the huge V5-2X norm was largely an
arbitrary/null-space witness. A still-large minimum `r` would show that the
strict shared frozen-bias/frozen-threshold head route is ill-conditioned on
TRAIN even after removing that freedom.

Neither outcome proves validation retention or selects a repair. No
parameter-size pass/fail threshold is introduced in this stage.

## Safety boundary

The diagnostic witness remains in memory and its values are never emitted or
persisted. No checkpoint is written and no model parameter is mutated. There
is no model training, autograd, backward, optimizer step, threshold/bias
change, architecture change, or production promotion.

Only aggregate TRAIN statistics are emitted. Historical validation,
First-30, V5 validation, and FINAL_HOLDOUT remain closed. 4-AI remains frozen.
