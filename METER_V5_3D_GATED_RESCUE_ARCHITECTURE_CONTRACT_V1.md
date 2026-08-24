# Meter V5-3D — Gated Rescue Architecture Contract V1

## Decision

V5-3C closed the numerical uncertainty in V5-3A. Both 2-AI and 3-AI have a
verified shared linear-head TRAIN witness with robust V5 margins and zero
damage to examples that the frozen specialist already classified correctly.
The witness is not accepted as a repair: it changes the frozen head by roughly
20.3x/25.7x its original L2 norm and rotates it by roughly 87 degrees.

The shared-head repair lane is therefore closed by safety policy. This is not
a claim that a shared linear witness is infeasible; it is a decision not to
replace a mature frozen classifier with a nearly orthogonal classifier whose
validation behaviour is unknown.

V5-3D preregisters one new candidate architecture only. It does not train a model,
create a checkpoint, run retention, open First-30/V5 validation, or authorize
production behaviour.

## Exact prerequisite evidence

This contract is bound to the completed V5-3C execution:

- implementation HEAD: `61361612abfce132994abaca742c855f91305b44`;
- exact-SHA harness HEAD: `74bbdba45b08ee4ca350b487627b792cc5255806`;
- report SHA256: `630b202f5369f12ea2562c81799613b81a295d77da08f9b3dd94a8fb1d801389`;
- execution-envelope SHA256: `d682a31809702373df312b828a8452440517e480c55ad69fdf9633f35e5f436d`.

V5-3C reported `DIAGNOSTIC WITNESS GATE = PASS`, no training, no checkpoint
write, no model mutation, and all protected data surfaces closed.

## Frozen authority

The existing 2-AI and 3-AI specialists remain the authoritative base:

- exact approved 64x64 slot pixels and transforms;
- exact frozen 64D backbone features;
- exact frozen `head.weight` and `head.bias`;
- thresholds 2-AI `0.48`, 3-AI `0.60`, and 4-AI `0.47`;
- unchanged multi-specialist ambiguity behaviour;
- 4-AI fully frozen.

The rescue path may be consulted only when its own specialist's frozen decision
is negative. A frozen positive cannot be demoted, rescored, or routed through
the rescue path.

## Fixed rescue topology

Each of 2-AI and 3-AI receives its own independent rescue module:

`frozen 64D feature -> Linear(64, 8) -> tanh -> Linear(8, 1) -> sigmoid`

The rescue threshold is fixed at `0.50`. There is no hidden-width, activation,
threshold, or architecture sweep. Each rescue module has exactly 529 trainable
parameters; the two modules together have 1,058. Rescue parameters live in a
separate namespace and artifact. They never replace or rewrite the frozen
specialist checkpoint.

Missing, malformed, unverified, or unauthorized rescue evidence means
`no rescue`; the frozen decision remains authoritative. If more than one digit
specialist passes after an authorized future shadow evaluation, the existing
ambiguous result is preserved. No specialist priority or tie-break is added.

## TRAIN-only candidate surface

A later separately authorized training recipe may use only frozen-negative
examples from the already-open V5 TRAIN and historical TRAIN surfaces. The
four non-empty groups are fixed independently for each specialist:

1. V5 frozen false-negative positives;
2. V5 frozen true negatives;
3. historical frozen false-negative positives;
4. historical frozen true negatives.

Each group must contribute exactly `0.25` of the candidate objective so record
count cannot let the historical negatives dominate. The expected exact group
counts are:

- 2-AI: `90`, `450`, `14`, `25254`;
- 3-AI: `90`, `450`, `12`, `25364`.

No validation example, First-30 example, V5 reserve example, or final-holdout
example may select the architecture, objective, optimizer, stopping point, or
threshold. This architecture contract deliberately contains no optimizer and
grants no training authority.

## Mandatory future gate order

1. A separate fixed training recipe and exact CI-green SHA.
2. One candidate execution with numerical and state-isolation checks.
3. TRAIN gate: V5 F1 `1.0`, no frozen-correct historical example becomes
   wrong, only rescue parameters change, and frozen tensors remain bit-identical.
4. Historical retention at the unchanged frozen thresholds. HOLD stops.
5. Immutable First-30 only after retention PASS. HOLD stops.
6. V5 validation only under separate authorization.
7. FINAL_HOLDOUT only under a later separate authorization.

No sweep, automatic second configuration, alternate threshold, fallback solver,
or production promotion follows a failure.

## Current safety state

Training is not authorized or executed. No model/checkpoint is written. The
rescue path is shadow-disabled. Historical retention and First-30 are not run;
V5 validation is closed; FINAL_HOLDOUT is locked; 4-AI is frozen; production
promotion is false.
