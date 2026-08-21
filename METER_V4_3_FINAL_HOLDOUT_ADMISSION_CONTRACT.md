# Meter V4-3 Final Holdout Admission Contract

## Purpose

Freeze a new, independent, family-disjoint real holdout before any V4-2 candidate evaluation.
This stage is **admission only**: it does not open the candidate checkpoint, does not run inference,
does not tune the model, and does not authorize production promotion.

## Candidate surface

- Drive root: `TEST/METER_V1/03_FINAL_HOLDOUT_150`
- Planned candidate pool: 65 per class (195 total).
- A bounded surplus is allowed at admission: each numerator class may contain **65..80** candidates.
- The final selection remains exactly 50/50/50 (150 total), regardless of bounded surplus size.
- Each candidate folder must contain a non-empty `image.png` and a bounded `bbox_meter.txt`.
- The `bbox_meter.txt` meter label must agree with the candidate folder class.
- Bbox coordinates must still be blank during admission. Human annotation happens only after the 150-family selection is frozen.

The bounded surplus rule exists only to provide extra replacement candidates after leakage/duplicate rejection. It does not change the final holdout size and does not permit arbitrary extra data.

## Leakage boundary

The selector reads only the completed V4-1 and V4-2 result metadata to construct the previously-observed family set:

- 27 V4-1 OOF/training families;
- 9 V4-2 development-validation families.

Those 36 families are ineligible for the final holdout. Any family appearing more than once anywhere in the candidate pool is also ineligible.

## Deterministic selection

For each numerator class independently:

1. reject previously-observed families;
2. reject any duplicate family in the candidate pool;
3. sort the remaining candidates by `(folder_name, family_id)`;
4. take exactly the first 50.

The final manifest therefore contains exactly 150 unique families, balanced 50/50/50 for numerators 2/3/4.
The canonical selected list is bound by `selection_sha256`. The manifest records the actual admitted candidate count and per-class counts.

## Safety state

The admission manifest must record:

- `bbox_annotation_complete = false`
- `model_evaluated = false`
- `candidate_checkpoint_opened = false`
- `test_opened = false`
- `runtime_connected = false`
- `production_promotion_authorized = false`

No final metric may be produced by this stage.

## Next gate

After the 150-family manifest is frozen, only those 150 samples are presented for human full-meter bbox annotation. The numerator crop is then derived deterministically from the accepted meter bbox using the already-frozen V4 crop geometry. Only after bbox QA is complete may the frozen V4-2 candidate be evaluated exactly once on this independent holdout.
