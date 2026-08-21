# Meter Real-Domain Adaptation V3-A3 — Residual Calibration Screen

Status: **shadow-only / development-only / TEST sealed**.

## Evidence entering V3-A3

The completed V3-A2 run remained `HOLD_NO_ACCEPTED_CANDIDATE` at 16/18 held-out REAL validation records. Source retention and localization passed, but the two remaining positive-class errors were `2/4 -> 3/4` and `4/4 -> 3/4`. A separate REAL TRAIN diagnostic showed the same `3/4` attraction pattern: every 2/4 and every 4/4 positive TRAIN record had 3/4 as its strongest wrong positive competitor, while 3/4 TRAIN classification was 9/9.

V3-A3 therefore does **not** increase the global positive margin. It tests a smaller post-hoc hypothesis: the already-trained V3-A2 residual needs bounded class-conditional gain calibration.

## Frozen parent

V3-A3 consumes the completed V3-A2 shadow model only if its resume state proves all of the following:

- adaptation version: `meter-real-domain-adaptation-v3-a2-positive-margin`;
- repository SHA: `2e3247d33d7d516a4def2aec87447ae7355e7e9d`;
- completed epoch: 20;
- best epoch: 20;
- exact D11 base checkpoint SHA-256: `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`.

The parent network is fully frozen. V3-A3 performs **zero optimizer steps** and changes no model weight.

## Single calibrated surface

Only the positive residual logits for 2/4 and 4/4 may be multiplied by deterministic gains. The none and 3/4 residual gains are exactly 1.0.

Candidate gains are the closed grid:

`1.000, 1.025, 1.050, ..., 1.250`

for each of 2/4 and 4/4, producing exactly 121 gain pairs.

For parent base logits `b` and adapter residual logits `r`, the calibrated logits are:

- none: `b_none + r_none`
- 2/4: `b_2/4 + g_2/4 * r_2/4`
- 3/4: `b_3/4 + r_3/4`
- 4/4: `b_4/4 + g_4/4 * r_4/4`

No record ID, family ID, validation example, or TEST example may influence the gain grid.

## Gain selection

Gain selection uses **REAL TRAIN only**. Held-out REAL validation is not inspected until after one pair is selected.

A candidate is eligible only if REAL TRAIN `none` recall does not fall below the frozen parent's REAL TRAIN `none` recall. Eligible candidates are ordered deterministically by:

1. highest minimum recall across {2/4, 3/4, 4/4};
2. highest macro-F1;
3. highest accuracy;
4. smallest total deviation from identity `(g_2/4-1) + (g_4/4-1)`;
5. smallest maximum gain;
6. smallest 2/4 gain;
7. smallest 4/4 gain.

This selection rule is frozen before held-out validation evaluation.

## Phase 0 gate — REAL only

After gain selection, evaluate exactly the existing 18 family-disjoint REAL validation records. Phase 0 passes only when all remain true:

- macro-F1 >= 0.900;
- accuracy >= 0.900;
- none recall >= 8/9;
- 2/4 recall = 3/3;
- 3/4 recall = 3/3;
- 4/4 recall = 3/3.

If Phase 0 fails, stop. Do not perform D10 source-retention I/O for V3-A3.

If Phase 0 passes, a separate Phase 1 must still prove unchanged D10 source-retention gates before any candidate can be considered. Phase 0 by itself never emits a deployable checkpoint.

## Safety boundary

- D11: fully frozen.
- V3-A2 adapter: fully frozen.
- bbox: exact frozen parent output.
- optimizer steps: 0.
- TEST: sealed; never enumerate or open.
- Resolver/runtime: disconnected.
- production promotion: forbidden.
- merge: forbidden until the bounded evidence is reviewed and all required CI and source-retention gates pass.
