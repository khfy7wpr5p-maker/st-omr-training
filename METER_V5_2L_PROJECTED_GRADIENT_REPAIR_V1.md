# Meter V5-2L — Projected-Gradient Repair V1

## Purpose

V5-2L is the first repair whose update rule directly addresses the parameter-space conflict measured by V5-2K. The V5-2K audit showed that the full-parameter V5 and historical gradients are strongly opposed for both 2-AI and 3-AI, with cosine similarities approximately -0.991 and -0.939. A fixed scalar mixture is therefore not the final repair mechanism.

V5-2L uses an A-GEM-style source-preserving projection. It does not choose a replay ratio or a fixed source-loss coefficient. The historical source gradient is a constraint direction.

## Exact authorized training surface

- Trainable specialists: 2-AI and 3-AI only.
- Initialization: exact frozen historical 2-AI and 3-AI checkpoints.
- V5 gradient surface: exactly the existing 540 `adaptation_train` slots from V5-2B.
- First-30 V5 diagnostic surface: zero-gradient only.
- Historical source surface: exact 26,964 M4A TRAIN records through the already-frozen historical M4A/D10 crop/preprocess path.
- Positive weight: `1.0` for both domains.
- 4-AI remains frozen.

## Projected update rule

At the start of each epoch, compute the exact full historical TRAIN mean gradient `g_source` at the current model parameters.

For each deterministic V5 minibatch, compute the V5 mean gradient `g_v5`.

If the gradients do not conflict:

`dot(g_v5, g_source) >= 0`

use:

`g_update = g_v5`

If they conflict:

`dot(g_v5, g_source) < 0`

use the closest first-order source-safe projection:

`g_update = g_v5 - dot(g_v5, g_source) / ||g_source||^2 * g_source`

This guarantees `dot(g_update, g_source) = 0` up to floating-point tolerance for conflicting batches. Under a direct gradient-descent step, the historical source loss therefore has zero first-order increase along the epoch reference gradient.

## Exact execution recipe

- Update method: direct projected SGD; no momentum, no Adam/AdamW state, no weight decay.
- Learning rate: `1e-4`.
- V5 batch size: `64`.
- Epochs: `12`.
- V5 batches per epoch: `9`.
- Parameter updates per specialist: `108`.
- Historical full-source reference gradient: recomputed once at the beginning of every epoch.
- Seed: `52023`, with deterministic specialist/epoch shuffle offsets.
- Projection scope: full trainable parameter vector, not per-layer projection.
- Gradient clipping: none.
- Gradient renormalization: none.
- Checkpoint selection: fixed final epoch only.
- No sweep, no automatic second configuration, no early stopping.
- Frozen thresholds remain unchanged: 2-AI `0.48`, 3-AI `0.60`, 4-AI `0.47`.

The 12-epoch count yields 108 V5 update opportunities, intentionally keeping the update-count scale close to the prior bounded pilots while changing the update geometry rather than adding another scalar replay/weight trial.

## Mandatory gate order

1. Run exactly the fixed projected-gradient repair.
2. Reproduce exact frozen M4A VALIDATION baselines.
3. Evaluate 2-AI/3-AI candidates on historical M4A VALIDATION using the existing retention gate:
   - absolute F1 drop <= `0.005`;
   - absolute recall drop <= `0.005`;
   - precision >= `0.98`;
   - recall >= `0.98`.
4. Only if historical retention is PASS, run the existing first-30 V5 diagnostic gate:
   - 2/4 >= `8/10`;
   - 3/4 >= `8/10`;
   - 4/4 >= `9/10`;
   - denominator exact-4 >= `26/30`.
5. If either gate fails, overall result is HOLD. No second repair configuration runs automatically.

## Closed surfaces

V5-2L does not authorize:

- threshold tuning;
- new BBox work;
- new crop geometry;
- new spatial heuristics;
- opening the 900 reserved V5 TRAIN examples;
- V5 validation;
- FINAL_HOLDOUT;
- 4-AI training or mutation;
- Resolver wiring;
- production promotion.

Historical replay uses only the previously frozen historical crop/preprocess contract and does not create new spatial semantics.

## Evidence and overwrite policy

V5-2L writes V5-2L-specific candidate and JSON evidence under the existing annotations directory. Existing V5-2L candidate/evidence paths are never silently overwritten. Partial failure is fail-closed and must be reviewed before any rerun.
