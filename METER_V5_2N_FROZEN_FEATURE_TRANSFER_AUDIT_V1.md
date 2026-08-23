# Meter V5-2N — Frozen Feature Transfer Audit V1

## Purpose

V5-2L showed that full-parameter source-safe projection still suppressed V5 learning while slightly degrading historical recall. Before another training repair is selected, V5-2N asks a narrower architectural question:

> Do the exact frozen 2-AI and 3-AI feature extractors already contain class-discriminative structure for the existing V5 adaptation TRAIN slots, even though the frozen output heads fail on that domain?

If the frozen 64-dimensional representation transfers meaningfully, a future frozen-base additive residual/head-only repair may be scientifically justified. If it does not, a larger residual branch may be required. V5-2N does not choose either architecture and authorizes no training.

## Surfaces

Read-only surfaces only:

- existing 540 V5 `adaptation_train` slots from the approved V5-2B slot manifest;
- historical M4A TRAIN, exactly 26,964 rows;
- exact frozen 2-AI and 3-AI checkpoints;
- historical pixels replayed only through the already-frozen M4A/D10 crop/preprocess helper;
- V5 pixels read only from the already-existing approved 64x64 slot crops.

The first-30 diagnostic rows receive no gradient and are not used by this audit. V5 validation, FINAL_HOLDOUT and the 900 reserved V5 TRAIN examples remain closed.

## Representation contract

The current digit specialist is used exactly as frozen:

`64x64 slot -> frozen feature extractor -> 64D feature vector -> frozen linear head`

V5-2N extracts the 64D vector immediately before the existing linear head. It does not add a layer, train a classifier, modify a checkpoint, or introduce a new crop/window/spatial rule.

Historical D11 four-class adapter experiments (`meter_real_domain_adaptation_v2`, `v3_a1`, `v3_a2`) are **concept-only references**. Their 4-class D11 model, 96x96 ROI and glyph/window semantics are not reused in the current digit-specialist lane.

## Preregistered descriptive metrics

For each of 2-AI and 3-AI independently:

1. Build historical M4A TRAIN positive and negative centroids in the frozen 64D representation.
2. For each V5 adaptation TRAIN row, compute squared-Euclidean distance to the two historical centroids.
3. Report the fraction whose nearest historical centroid matches its binary target label, separately for V5 positives, V5 negatives and overall.
4. Report the correct-centroid margin `distance_wrong - distance_correct`; positive values mean the V5 feature is closer to the source centroid carrying the same target label.
5. Compare the historical and V5 class-separation vectors:
   - `delta_source = centroid_source_positive - centroid_source_negative`
   - `delta_v5 = centroid_v5_positive - centroid_v5_negative`
   - report cosine similarity and norm ratio.
6. Report cosine alignment of the existing frozen linear-head weight with `delta_source` and `delta_v5`.
7. Reproduce the frozen-head V5 adaptation TRAIN confusion metrics at the unchanged thresholds as a binding sanity check, not a tuning surface.

These metrics are descriptive. No numeric PASS threshold is preregistered and no architecture is automatically selected from them.

## Safety boundary

V5-2N authorizes no:

- training;
- backward/autograd gradient computation;
- optimizer step;
- checkpoint write;
- threshold tuning;
- new BBox;
- new crop geometry;
- new spatial heuristic;
- old D11 glyph/window reuse;
- 900 reserve TRAIN opening;
- V5 validation opening;
- FINAL_HOLDOUT opening;
- 4-AI mutation;
- Resolver wiring;
- production promotion.

The only output is a V5-2N JSON evidence report under the existing annotations directory. Existing V5-2N evidence is never silently overwritten.
