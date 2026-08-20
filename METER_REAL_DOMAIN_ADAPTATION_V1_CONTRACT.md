# Meter Real-Domain Adaptation v1 Contract

## Purpose

The frozen D11 Meter refiner is a valid synthetic technical baseline but the
first real-input replay collapsed to `none`.  This stage creates one bounded,
offline, shadow-only adaptation experiment from explicitly reviewed real Meter
ROIs.  It does not replace the frozen D11 checkpoint, connect runtime output,
open TEST, or authorize production promotion.

The path is:

```text
explicit permission + privacy review
    -> 72 accepted teacher-gold decisions
    -> deterministic 256x192 historical Meter ROIs
    -> family-disjoint real TRAIN / VALIDATION
    -> exact audited D11 Meter state
    -> frozen encoder + trainable projection/classifier/bbox heads
    -> balanced D10 TRAIN replay
    -> real and unchanged D10 VALIDATION gates
    -> HOLD or shadow candidate
```

## Input boundary

The admission layer accepts exactly the approved `METER_V1` pilot:

- source: `METER_V1/01_REVIEW/train`;
- 36 source images and 72 reviewed tasks;
- 36 visible Meter records: 12 each of `2/4`, `3/4`, and `4/4`;
- 36 later-measure `none` records;
- every admitted record must be `accepted`, `label_confirmed=true`, and
  `crop_usable=true`;
- positive records require a bounded Meter bbox; `none` records forbid one;
- source and answer task identities must match exactly;
- `test_opened` must be false in pilot, choices, privacy evidence, manifest,
  receipt, metrics, and verification.

Teacher corrections are not automatic training data.  Admission additionally
requires separate canonical evidence objects for:

1. explicit approval of the one offline Meter adaptation pilot;
2. a privacy review confirming no detected personal data;
3. no automatic learning, no TEST authorization, no redistribution, and no
   production-promotion authorization.

The source bytes, choices, evidence, derivatives, checkpoints, and metrics stay
outside Git.

## Deterministic derivative and split

Every accepted source ROI is converted to grayscale, cropped with the reviewed
ROI box, fit-padded with preserved aspect ratio onto a white `256x192` canvas,
and resized with bilinear interpolation.  Positive source bboxes are mapped
through the same transform.  Every PNG, label, record, manifest, receipt, and
artifact binding receives a SHA-256 identity and is independently reopened.

The 36 source families are disjoint across adaptation splits.  Within every
Meter-class/package stratum, one of four `aa` families and two of eight `ab`
families are selected deterministically for validation.  Paired visible/none
records stay together.

| Adaptation split | none | 2/4 | 3/4 | 4/4 | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 27 | 9 | 9 | 9 | 54 |
| VALIDATION | 9 | 3 | 3 | 3 | 18 |

No row is derived from or moved into TEST.

## Base checkpoint and replay

The only allowed initialization is the exact audited D11 checkpoint:

`cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`

The D11 convolutional encoder is frozen and hash-checked before and after the
run.  Only the existing projection, four-class classifier, and bbox head may
update.  No layer or class is added.

Each epoch mixes the repeated 54-record real TRAIN surface with a deterministic
balanced sample of 64 accepted D10 TRAIN Meter records per class (256 synthetic
records).  Normalized inverse-frequency class weights compensate for the
resulting 172/100/100/100 effective class counts.  The unchanged 1,224-record
D10 Meter VALIDATION surface remains the
synthetic-regression guard.  D10 TEST is neither enumerated nor opened.

## Candidate gates

A checkpoint is not emitted unless one epoch satisfies every gate:

| Gate | Minimum/maximum |
| --- | ---: |
| Real VALIDATION macro F1 | at least 0.800 |
| Real VALIDATION accuracy | at least 0.833 |
| Real `none` recall | at least 0.888 (8/9) |
| Each real positive-class recall | at least 0.666 (2/3) |
| Real macro-F1 gain over base D11 | at least 0.200 |
| D10 VALIDATION macro-F1 drop | at most 0.020 |
| D10 positive localization F1 drop | at most 0.030 |

Candidate selection first requires all gates, then maximizes real macro F1 and
uses real validation loss only as the tie-breaker.  If no epoch passes, the run
finishes as `HOLD_NO_ACCEPTED_CANDIDATE` and emits no checkpoint.

Before an accepted shadow checkpoint is written, its complete real-validation
probabilities and bboxes must reproduce with one identical fingerprint in
10/10 CPU inference replays.  The serialized checkpoint is then reopened with
the weights-only loader, strictly loaded into the frozen architecture, checked
for finite values, and required to reproduce the candidate model-state hash.

## Colab background execution and recovery

The background Colab entry point first materializes the accepted D10
TRAIN/VALIDATION bundle from mounted Drive onto the runtime's local SSD.  It
reports an exact copied-file numerator/denominator and then performs the full
authoritative D10 verification against the unchanged manifest and artifact
binding.  The cache planner rejects TEST before copying any record.

The runner writes an atomic status heartbeat to Drive at least every 30
seconds.  Status exposes phase `?/9`, epoch `?/8`, batch `?/total`, elapsed
time, and terminal result/error.  The monitor may be stopped and reopened
without stopping the background process.  Colab itself may still reclaim a
disconnected runtime; no notebook can guarantee execution after that event.

After every complete epoch, a weights-only-loadable resume state is atomically
written to the shadow run directory.  Resume is accepted only when repository,
configuration, teacher manifest, D10 identity, D11 checkpoint, base model,
frozen encoder, and recomputed baseline metrics all match.  A partial epoch is
never claimed as complete and is repeated after restart.

## Outputs and closed boundaries

An accepted checkpoint is labeled
`meter-real-domain-shadow-candidate-v1`.  Metrics and verification bind the
repository SHA, base checkpoint SHA, D10 manifest/binding, teacher-gold
manifest/binding, frozen configuration, state hashes, optimizer steps, and
all gate results.

The following remain false after this stage:

- sealed TEST access;
- runtime connection;
- Deterministic Resolver connection;
- frozen D11 checkpoint replacement;
- online or automatic learning;
- production promotion.

A later decision requires a larger independently reviewed real corpus, a fresh
sealed evaluation authorization, runtime replay evidence, and an explicit
promotion contract.
