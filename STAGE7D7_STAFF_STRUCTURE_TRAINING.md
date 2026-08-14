# Stage 7-D7 — StaffSet + StructureSet specialist training

## Purpose

Stage 7-D7 is the first real specialist-model training package. It consumes the accepted Stage 7-D6 derivative dataset and trains two independent models:

1. `Staff` specialist — staff-line and staff-region dense geometry.
2. `Structure` specialist — system/measure regions, trailing barlines, visible G2 clef and visible `2/4|3/4|4/4` meter geometry.

The two specialists do **not** share weights or an optimizer. This preserves the specialist decomposition selected after the D2/D3 monolithic baseline failed to achieve useful recognition.

## Accepted input only

D7 is bound to the authoritative D6 build:

```text
derivative build ID      0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519
D6 manifest SHA-256      e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9
artifact binding SHA-256 3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1
TRAIN                    1,230 images / 410 families
VALIDATION                 153 images /  51 families
TEST specialist records      0
```

Before training, D7 reruns the independent D6 persisted-output verifier. A different but structurally valid derivative build is not accepted silently.

## Split boundary

```text
D6 manifest
   ↓ read split first
TRAIN       → image + label access → optimizer allowed
VALIDATION  → image + label access → read-only metrics
TEST        → immediate failure before D7 image/label path derivation
```

Validation is never passed to an optimizer. TEST remains sealed until Stage 9.

## Dense training targets

D6 variable-length geometry is rasterized deterministically into fixed-size `96 × 512` masks.

### Staff specialist

```text
channel 0  staff_lines
channel 1  staff_region
```

`staff_lines` is derived from the five final-PNG staff-line segments. `staff_region` is derived from the final-PNG staff-instance bbox.

### Structure specialist

```text
channel 0  system_region
channel 1  measure_region
channel 2  barline
channel 3  clef_g2
channel 4  meter_2_4
channel 5  meter_3_4
channel 6  meter_4_4
```

Visible meter boxes are placed only in the matching meter-class channel. Courtesy meter geometry already rejected by the D6/D5-v2 boundary cannot become a D7 target.

## Models

Each specialist is a small independent fully convolutional encoder-decoder:

```text
1-channel grayscale image
  ↓
Conv 1→8
  ↓
Conv 8→16 /2
  ↓
Conv 16→24 /2
  ↓
Conv 24→24
  ↓
upsample + Conv 24→16
  ↓
upsample + Conv 16→8
  ↓
1×1 task head
```

The Staff head has 2 output channels; the Structure head has 7. Trainable parameter counts remain below the repository V1 ceiling.

## Frozen training profile

```text
input            96 × 512 grayscale
batch size       6
epochs           8 per specialist
optimizer        AdamW
learning rate    0.0007
weight decay     0.0001
grad clip        1.0
objective        BCE-with-logits + soft Dice
selection        minimum validation loss independently per specialist
runtime          pinned deterministic CPU PyTorch
```

All TRAIN ordering is deterministic from the frozen master seed. Each model starts from its own deterministic seed.

## Metrics

For each specialist D7 records:

- untrained validation loss;
- best validation loss;
- best epoch;
- optimizer-step count;
- thresholded per-channel Dice;
- model fingerprint;
- exact model-state SHA-256;
- trainable parameter count.

An authoritative model is accepted only if training improves validation loss over the untrained model and the selected best state reproduces the recorded validation loss when restored.

## Evidence output

Large model artifacts remain outside normal Git:

```text
<run-root>/<run-id>/
  checkpoint-<sha256>.pt
  metrics-<sha256>.json
  verification-<sha256>.json
  COMPLETE
```

The checkpoint stores the two task-isolated state dictionaries. It is reloaded with `torch.load(..., weights_only=True)` and both state hashes are independently rechecked before the run is complete.

The verification binds:

- exact repository SHA/origin;
- exact pinned runtime versions;
- accepted D6 manifest/artifact identities;
- TRAIN/VALIDATION counts;
- TEST records/opened = 0/false;
- checkpoint file hash;
- Staff and Structure model-state hashes;
- optimizer-step counts;
- stable repository/runtime identity for the entire run.

## Not in D7

D7 does not implement:

- NoteHead/Rest/Accidental/Rhythm/Pitch/Chord specialists;
- learned fusion;
- MusicXML candidate production;
- TEST evaluation;
- Stage 8 real-data fine-tuning;
- ScoreMosaic integration;
- online or automatic learning.

The next decision after the authoritative D7 run is based on validation evidence. TEST is not opened to decide whether Staff/Structure training is good enough to continue.
