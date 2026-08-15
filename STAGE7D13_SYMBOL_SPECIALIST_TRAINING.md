# Stage 7-D13 — NoteHead + Rest + Accidental specialist training contract

Status: **active — pre-training contract; optimizer not yet authorized**.

Stage 7-D13 turns the accepted D12 deterministic symbol ground truth into a frozen training package for three separate learned specialists:

```text
NoteHeadSet   -> note-head detection + open|filled
RestSet       -> rest detection + half|quarter|eighth
AccidentalSet -> accidental detection + sharp|flat|natural
```

D13 does not collapse these tasks into one joint optimizer. The three specialists may reuse the same architecture implementation and derivative format, but each has an independent model state, optimizer state, validation history, checkpoint identity, metrics, and acceptance decision.

## Accepted D12 dependency

D13 may consume only the independently verified D12 development bundle produced from authoritative executable head:

```text
e2de6f64c27be2dd6d706a700553ef4f5c236e25
```

Accepted D12 persisted identity:

```text
derivative build id:
35323e831c5c693bf607808c5f846624445bf537f30e1d93db9ca949a7eed106

manifest SHA-256:
a372eba640b38704020922ad4eb102738fc4492d278a38e4b51b8ad0b78d4ea1

artifact binding SHA-256:
14c64e16ca2f993bf94f8009bf0bcd974b7ddee87c19bb748219ba3f774b229d
```

D12 cardinality remains:

```text
TRAIN        1230 source images / 410 families
VALIDATION    153 source images /  51 families
TEST            0 specialist records
```

The D12 class inventory is frozen as input evidence to D13. D13 must re-open and independently verify the accepted D12 bundle before any training derivative is admitted.

## Class-readiness gate

D13 freezes the following minimum readiness rule **before any optimizer step**:

```text
per supported class:
TRAIN instances      >= 1000
VALIDATION instances >=  150
```

The accepted D12 inventory satisfies this first-pass synthetic specialist-training gate for every supported V1 class. These are data-readiness minima only; they are not model-quality acceptance thresholds.

## Class-imbalance policy

No page/family is oversampled by class. Family-exclusive source distribution remains intact.

Classification imbalance is handled only inside the TRAIN objective by deterministic positive-class weights:

```text
raw_weight(c) = 1 / sqrt(TRAIN_count(c))
normalized(c) = raw_weight(c) / mean(raw_weight(all classes in specialist))
weight(c)     = clip(normalized(c), 0.5, 3.0)
```

Rules:

- weights are computed from TRAIN counts only;
- VALIDATION never influences class weights;
- VALIDATION is never resampled;
- weights are frozen into the training-profile fingerprint before optimizer creation;
- no post-hoc manual class weight may be selected after seeing validation metrics.

This specifically protects the rarer `open`, `half`, and `natural` classes without duplicating families or changing the validation distribution.

## D13 training-derivative surface

Training does not operate directly on arbitrary full pages. Each accepted D12 measure becomes one deterministic measure record.

For every measure:

1. use the accepted D12 `measure_bbox` as the only crop authority;
2. crop from the exact frozen source PNG;
3. preserve aspect ratio;
4. letterbox the complete measure into a fixed `512 x 128` grayscale canvas;
5. use white background padding;
6. never crop away measure content to satisfy the fixed canvas;
7. transform every specialist bbox and center with the exact same scale + pad transform;
8. persist the transform and its SHA-256 fingerprint;
9. reject any target that becomes non-finite, non-positive, or leaves the transformed measure bounds;
10. keep TRAIN/VALIDATION family isolation unchanged.

No TEST derivative is permitted.

The D13 derivative builder must be completed and independently verified before optimizer creation. Exact measure-record counts and exact optimizer-step counts are therefore derived from the verified D13 derivative bundle, not guessed in advance.

## Frozen model architecture

Each specialist is a separate compact fully convolutional center detector using the same architecture family but independent weights.

Input:

```text
1 x 128 x 512 grayscale measure image
```

Output stride:

```text
4 pixels
```

Heads:

```text
class heatmap: C channels
bbox size:     2 channels (width, height)
center offset: 2 channels (sub-cell x, y)
```

where:

```text
NoteHeadSet   C=2
RestSet       C=3
AccidentalSet C=3
```

The architecture is a small encoder/decoder CNN. Each specialist must remain at or below `1,500,000` trainable parameters; the three accepted specialist checkpoints combined must remain at or below `4,500,000` trainable parameters.

No pretrained external vision backbone is authorized in D13. No D7/D11 learned state is loaded into these specialists.

## Frozen objective

Each specialist uses a CenterNet-style local-center objective:

```text
weighted positive focal heatmap loss
+ 1.0 * positive bbox-size Smooth-L1
+ 1.0 * positive center-offset Smooth-L1
```

The class weights above affect only positive heatmap terms. Background/negative terms are not multiplied by inverse-frequency class weights.

Only target centers contribute bbox-size and center-offset regression loss.

## Frozen optimizer profile

The first authoritative D13 training profile is:

```text
optimizer            AdamW
batch size           16
epochs               10
learning rate        0.0007
weight decay         0.0001
grad clip            1.0
master seed          713013
heartbeat            every 50 TRAIN batches
checkpoint selection minimum validation loss per specialist
execution            deterministic pinned CPU runtime
```

The three specialists train sequentially with three separate optimizers. No optimizer state is shared across specialists.

Exact optimizer steps are computed only after the verified measure-derivative record counts are known. A completed run must prove the exact expected count for every specialist.

## Frozen validation decoder

Validation decoding is deterministic and fixed before training:

- 3x3 local-max suppression on each class heatmap;
- score threshold `0.25`;
- maximum `256` decoded candidates per measure per specialist;
- predicted bbox reconstructed from center + positive width/height regression;
- deterministic score-descending ordering with stable coordinate/class tie breaks;
- one-to-one greedy matching; once a GT or prediction is matched it cannot be reused.

No validation-dependent threshold tuning is permitted.

## Metrics

Each specialist reports three validation metrics.

### 1. Class-aware center F1 @ 4 px

A prediction is correct only when:

- class is correct;
- predicted center is within Euclidean distance `<= 4.0` input pixels of the GT center;
- the prediction and GT have not already been matched.

This metric measures detection + class + center localization.

### 2. Class-aware bbox F1 @ IoU 0.50

A prediction is correct only when:

- class is correct;
- bbox IoU with a GT box is `>= 0.50`;
- one-to-one matching is respected.

### 3. Macro class F1 on center-matched detections

Predictions and GT are first matched by center distance `<= 4.0` without requiring class equality; the matched class labels then produce per-class F1 and unweighted macro-F1. Unmatched GT and predictions count as class-specific false negatives/false positives.

## Frozen quality gates

The first D13 synthetic validation gates are frozen before training:

```text
NoteHeadSet
  class-aware center F1 @4px >= 0.85
  class-aware bbox F1 @IoU50  >= 0.75
  macro class F1              >= 0.90

RestSet
  class-aware center F1 @4px >= 0.80
  class-aware bbox F1 @IoU50  >= 0.70
  macro class F1              >= 0.85

AccidentalSet
  class-aware center F1 @4px >= 0.80
  class-aware bbox F1 @IoU50  >= 0.70
  macro class F1              >= 0.85
```

All three metrics must pass for a specialist. D13 technical acceptance requires all three specialists to pass; one successful specialist cannot mask another failed specialist.

These gates evaluate the accepted synthetic TRAIN/VALIDATION curriculum only. They are not claims about production accuracy or unseen real scanned scores.

## Checkpoint and run policy

For each specialist:

- start from deterministic random initialization;
- measure untrained validation loss before optimizer step 1;
- train exactly the frozen number of epochs;
- evaluate read-only VALIDATION after every epoch;
- select the epoch with minimum validation loss;
- restore that exact best state before final metrics/persistence;
- persist only the restored accepted-candidate state in the combined authoritative checkpoint;
- store epoch-level training/validation history in metrics evidence;
- checkpoint reload must use safe tensor/state loading policy;
- final persisted model hashes must be independently recomputed.

## Safety boundary

D13 must fail closed if any of the following occurs:

- accepted D12 identity differs;
- D12/D13 source or label hash changes;
- TEST is encountered beyond reading only `split`;
- a family crosses TRAIN/VALIDATION;
- class-readiness minimum is not met;
- the derivative transform is ambiguous or clips a target;
- a model exceeds its parameter cap;
- a tensor/loss/gradient/model state becomes non-finite;
- optimizer-step count differs from the derived frozen expectation;
- repository/runtime identity changes during an authoritative run;
- persisted metrics/checkpoint/verification identities disagree;
- independent verifier fails.

No ScoreMosaic runtime integration, real-data ingestion, teacher-correction auto-learning, TEST opening, RhythmSet, PitchSet, or ChordSet training is authorized by D13.

## Controlled implementation sequence

```text
D13-0 training contract + invariant tests
        ↓
D13-1 deterministic measure-derivative builder
        ↓
D13-2 independent persisted derivative verifier
        ↓
D13-3 freeze exact derivative counts + optimizer steps
        ↓
D13-4 implement three compact specialists + metrics
        ↓
D13-5 exact-head regression / architecture-safety review
        ↓
D13-6 authoritative external training
        ↓
D13-7 independent persisted-run verification
        ↓
closure evidence
        ↓
explicit merge approval
```

No optimizer may run before D13-1 through D13-5 are complete and the exact training profile is fingerprinted on the verified derivative identity.