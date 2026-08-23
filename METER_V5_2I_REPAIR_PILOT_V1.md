# Meter V5-2I — Exact Repair Pilot V1

## Purpose

V5-2I executes exactly one repair-training configuration approved after V5-2F/V5-2G/V5-2H evidence. The objective is to test whether source-domain replay can prevent historical forgetting while preserving the V5 target-domain gain.

This is not a sweep. There is no automatic second configuration.

## Exact authorized recipe

- Trainable specialists: 2-AI and 3-AI only.
- Initialization: exact frozen historical 2-AI and 3-AI checkpoints.
- V5 gradient surface: exactly the existing 540 `adaptation_train` slots from V5-2B.
- First-30 V5 diagnostic surface: zero-gradient only.
- Historical replay: exactly 6,480 M4A TRAIN examples, 12 historical examples per V5 slot.
- Historical replay allocation: `2=367`, `3=381`, `4=1537`, `NONE=4195`.
- Replay selection: deterministic SHA-256 ranking, stratified, without replacement, same selected source rows for both 2-AI and 3-AI.
- Historical pixels: exact previously recovered M4A/D10 historical crop/preprocess path only.
- Positive weight: `1.0`.
- Optimizer: AdamW.
- Learning rate: `1e-4`.
- Weight decay: `1e-4`.
- Batch size: `64`.
- Epochs: `1`.
- Seed: `52023` with specialist-specific deterministic shuffle offset.
- Combined examples per specialist: `7020`.
- Expected optimizer steps per specialist: `110`.
- Checkpoint selection: fixed final epoch only.
- Frozen thresholds remain unchanged: 2-AI `0.48`, 3-AI `0.60`, 4-AI `0.47`.

## Gate order

Gate ordering is mandatory and fail-closed:

1. Run the one authorized repair-training configuration.
2. Reproduce the exact historical M4A VALIDATION frozen baselines.
3. Evaluate repair candidates on historical M4A VALIDATION using the existing retention limits:
   - absolute F1 drop <= `0.005`;
   - absolute recall drop <= `0.005`;
   - precision >= `0.98`;
   - recall >= `0.98`.
4. Only if historical retention is PASS, run the existing first-30 V5 diagnostic gate:
   - 2/4 >= `8/10`;
   - 3/4 >= `8/10`;
   - 4/4 >= `9/10`;
   - denominator exact-4 >= `26/30`.
5. If either gate fails, overall result is HOLD. No second configuration runs automatically.

## Explicitly closed surfaces

V5-2I does not authorize:

- threshold tuning;
- any new BBox work;
- any new crop geometry;
- any new spatial heuristic;
- the 900 reserved V5 TRAIN examples;
- V5 validation;
- FINAL_HOLDOUT;
- 4-AI training or mutation;
- Resolver wiring;
- production promotion.

Historical M4A replay uses the already-frozen historical crop/preprocess contract and is not a new spatial heuristic.

## Evidence handling

Runtime evidence is written under the existing V5 annotations directory with V5-2I-specific filenames. Existing V5-2I evidence or candidate directories are never silently overwritten. Partial failure is fail-closed and must be reviewed before any rerun.
