# Meter V5-3A — Robust-Margin Head Candidate V1

## Decision

V5-2Z proved that the shared frozen-runtime head constraints are feasible on
TRAIN with maximum per-component changes below `0.74`.  That disproved the
claim that the huge V5-2Y component changes were unavoidable.  It did not
produce a deployable repair: the diagnostic heads rotated by about 88 degrees,
historical logits moved by up to 14–17 units, and active rows sat at a `1e-4`
margin.

V5-3A is the first candidate-selection stage after that diagnosis.  It uses a
fixed robust margin and minimizes total parameter movement before limiting the
largest individual movement.  It is not a solver or threshold search.

## Exact prerequisite evidence

V5-3A binds to the completed V5-2Z execution:

- implementation HEAD: `040e1d80fcbb09f6cac7b43e15fd34567c3f7dad`;
- report SHA256:
  `39fd82009f1bbef66877d0e65ad9719f7ecff9adc67f2c6d1a6a6e1a163ab8e4`;
- execution-envelope SHA256:
  `fa3adc4f96fcf1d3109b43750b0958a3267fa57075a9c6ff061ad30b42864e12`.

The V5 slot manifest, record counts, frozen-correct historical surface, source
checkpoint hashes, frozen bias, and frozen runtime thresholds are checked
again before fitting.

## Fixed robustness policy

The operational V5 signed decision margin is fixed at `0.25` logit.  Around
the frozen thresholds `0.48` and `0.60`, this is approximately a six
percentage-point probability cushion on either side of the threshold.  The LP
uses an additional fixed `1e-4` numerical buffer so the float32 checkpoint must
still meet the operational `0.25` margin after copy-back.

Every historical TRAIN row classified correctly by the frozen specialist has
the following required signed margin:

`max(1e-4, min(frozen signed margin, 0.25))`.

Thus a weak but correct frozen decision may not be degraded, while a strong
frozen decision must retain at least the operational `0.25` cushion.  Frozen
wrong historical rows are descriptive only and cannot force the candidate to
repeat an existing error.

## Two fixed lexicographic LPs

For each specialist, write `w = w0 + delta` while keeping bias and threshold
frozen.

1. Primary LP: minimize `sum(abs(delta))` under all V5 and frozen-correct
   historical margin constraints.
2. Secondary LP: fix the primary L1 optimum with absolute slack `1e-6`, then
   minimize `max(abs(delta))`.

Both LPs use pinned SciPy `1.18.0`, `highs-ds`, presolve disabled, and
primal/dual feasibility tolerances `1e-9`.  There is no weighted objective,
automatic fallback, alternative configuration, solver sweep, threshold search,
bias search, L2 objective, or historical-logit-drift objective.

## Fail-closed candidate gate

A candidate is accepted only when both specialists independently pass all of
the following:

- both solver stages report success;
- primary L1 and secondary Linf objectives match independent recomputation;
- every solver-margin constraint passes;
- the primary L1 cap and every component bound pass;
- `new_logit - frozen_logit = feature dot delta` passes independently;
- all V5 TRAIN rows remain correct after float32 copy-back with the operational
  `0.25` margin;
- every frozen-correct historical TRAIN decision remains correct and its
  required margin passes after float32 copy-back;
- only `head.weight` changes; backbone and `head.bias` are bit-identical;
- saved checkpoint hash, state fingerprint, metadata, and reload invariants
  pass.

If either specialist fails, no candidate checkpoint directory is published.
The report records `HOLD` and the process stops.

## Gate order

V5-3A stops immediately after candidate and numerical-integrity evidence.
Passing TRAIN constraints does not claim historical preservation or
generalization.  The only allowed next step is Historical Retention at the
same frozen thresholds.  Retention HOLD stops the path.  Only Retention PASS
may authorize immutable First-30.

Historical VALIDATION, First-30, V5 VAL, and FINAL_HOLDOUT are not opened by
this module.  4-AI remains frozen.  Production promotion is forbidden.
