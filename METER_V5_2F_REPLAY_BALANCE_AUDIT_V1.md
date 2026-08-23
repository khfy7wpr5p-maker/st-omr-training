# Meter V5-2F — Replay-Balance Audit V1

## Purpose

V5-2E established the conservative root-cause class:

`V5_ONLY_DOMAIN_ADAPTATION_WITH_UNCONSTRAINED_SOURCE_FORGETTING`

while preserving `DOMINANT_MECHANISM=UNRESOLVED` and showing that `pos_weight=5.0` is not the unique root cause because the frozen V5 target-positive pressure remains at least two orders dominant even in the diagnostic `pos_weight=1.0` counterfactual.

V5-2F is the next read-only analytical stage. It does **not** train a model. It reads the completed V5-2E report and quantifies the source-replay pressure required, at the frozen checkpoint, to counter the one-sided V5 target-domain logit pressure.

## Main question

For each trainable specialist (2-AI and 3-AI) and each already-audited diagnostic positive weight (`1.0`, `5.0`), what source-example replay ratio would make the **mean signed logit pressure** approximately zero at the frozen starting point?

This is a logit-level analytical balance calculation only. It does not prove parameter-gradient balance in every layer, does not guarantee a stable training trajectory, and cannot authorize repair training by itself.

## Carried-forward evidence

V5-2F must fail closed unless the V5-2E report confirms all of the following:

- schema is `st-omr-meter-v5-2e-gradient-pressure-audit-v1`;
- `ROOT_CAUSE_CLASS=V5_ONLY_DOMAIN_ADAPTATION_WITH_UNCONSTRAINED_SOURCE_FORGETTING`;
- `DOMINANT_MECHANISM=UNRESOLVED`;
- `POS_WEIGHT_5_UNIQUE_ROOT_CAUSE_SUPPORTED=False`;
- V5 adaptation TRAIN slot count is exactly 540;
- historical M4A TRAIN record count is exactly 26,964;
- counterfactual positive weights are exactly `[1.0, 5.0]`;
- training/backward/checkpoint/threshold/spatial/VAL/final safety flags remain closed;
- 4-AI remains frozen;
- Resolver wiring and production promotion remain unauthorized.

## Signed-pressure contract

For a pressure profile from V5-2E:

- positive examples contribute negative `dL/dz`;
- negative examples contribute positive `dL/dz`.

Therefore:

`signed_total = negative_pressure_total - positive_pressure_total`

and

`signed_mean = signed_total / count`.

For a source replay ratio `r` measured as **historical source examples per one V5 target example**:

`combined_signed_mean(r) = v5_signed_mean + r * source_signed_mean`

If the two domain means have opposite signs and `source_signed_mean != 0`, the analytical zero crossing is:

`r* = -v5_signed_mean / source_signed_mean`.

If signs do not oppose, or the source mean is zero/non-finite, no finite balance ratio is reported.

## Required report fields

For each specialist and each diagnostic positive weight, report:

- V5 frozen signed total and signed mean;
- historical frozen signed total and signed mean;
- whether the domain pressures oppose each other;
- exact analytical zero-crossing source/V5 example ratio when finite;
- equivalent historical examples for one 540-slot V5 pass;
- equivalent fraction of the full 26,964-record historical TRAIN pool;
- full-source-pass ratio (`26964 / 540`);
- residual signed pressure if one complete historical TRAIN pass is paired with one complete V5 TRAIN pass;
- whether the zero crossing lies within one full historical pass.

Also report, for each positive weight, the cross-specialist zero-crossing span across 2-AI and 3-AI if both are finite.

## Interpretation boundary

V5-2F may establish analytical feasibility evidence such as:

- both specialists have finite source-replay zero crossings;
- a zero crossing is or is not reachable within one full historical TRAIN pass under equal per-example weighting;
- the `pos_weight=1.0` and `pos_weight=5.0` counterfactuals imply materially different replay scales.

It must **not** automatically select:

- a positive weight;
- a replay ratio;
- a sampling strategy;
- a source subset;
- a specialist-specific training recipe;
- an optimizer or learning rate;
- a repair checkpoint.

No analytical result from this stage constitutes repair-training authorization.

## Safety boundary

V5-2F:

- TRAINING=False
- BACKWARD=False
- OPTIMIZER_STEPS=0
- CHECKPOINT_READ=False
- CHECKPOINT_WRITE=False
- IMAGE_READ=False
- NEW_BBOX=False
- NEW_CROP_GEOMETRY=False
- NEW_SPATIAL_HEURISTIC=False
- THRESHOLD_TUNING=False
- REPLAY_RATIO_SELECTED=False
- POSITIVE_WEIGHT_SELECTED=False
- REPAIR_TRAINING_AUTHORIZED=False
- RESERVE_V5_TRAIN=CLOSED
- V5_VAL=CLOSED
- FINAL_HOLDOUT=LOCKED
- 4-AI=FROZEN
- RESOLVER_WIRING=False
- PRODUCTION_PROMOTION=False

V5-2F reads only the existing V5-2E JSON evidence and writes a derived audit JSON. Any repair-training pilot must be a separate, preregistered stage after human review of this report.
