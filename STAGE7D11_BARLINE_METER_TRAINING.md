# Stage 7-D11 — Barline + Meter Refiner Training

## Purpose

D11 is the first optimizer package after the D9 Structure refinement decision and the accepted D10 local ROI derivative build. It trains only two new local models:

```text
D10 measure-end ROI   192 x 128 -> barline_refiner -> barline mask
D10 measure-start ROI 192 x 256 -> meter_refiner   -> none|2/4|3/4|4/4 + bbox
```

The accepted D7 Structure core is not loaded by the D11 training module. Its accepted state identity remains frozen at the D9-recorded SHA-256 and cannot be mutated by a D11 optimizer.

## Accepted input surface

D11 accepts only a fully verified, COMPLETE D10 authoritative bundle whose D10 repository identity is:

`562c8fcfabf1b41573f1ef591d88ae65335ce16a`

The external run must additionally supply the independently recorded accepted D10 manifest SHA-256 and artifact-binding SHA-256. D11 refuses to infer those acceptance identities from an arbitrary candidate bundle.

Frozen cardinality:

```text
D10 ROI records        22,128
TRAIN ROI              19,680
VALIDATION ROI          2,448
TEST                         0
barline TRAIN           9,840
barline VALIDATION      1,224
meter TRAIN             9,840
meter VALIDATION        1,224
```

Every image and label is hash-checked again when materialized into a batch.

## Models

### Barline refiner

A small fully convolutional local segmenter operates on the frozen D9 `192 x 128` measure-end crop. The raster training target is derived only from the D10 `barline_segment` geometry.

Frozen parameter budget: at most `500,000` trainable parameters.

### Meter refiner

A small convolutional local encoder preserves coarse spatial layout through a fixed `6 x 8` adaptive pool. It has two heads:

- four-class meter classification: `none | 2/4 | 3/4 | 4/4`;
- normalized positive-record meter bounding-box regression.

Classification uses a deterministic TRAIN-only inverse-frequency weighting rule. Bounding-box loss is applied only when D10 carries a visible current-measure meter bbox.

Frozen parameter budget: at most `750,000` trainable parameters. Combined D9 ceiling: at most `1,250,000` new trainable parameters.

## Training profile

```text
runtime              pinned CPU PyTorch
batch size           32
epochs               8 / refiner
optimizer            AdamW
learning rate        0.0007
weight decay         0.0001
grad clip            1.0
checkpoint selection minimum validation loss per refiner
barline objective    BCE-with-logits + soft Dice
meter objective      balanced CE + positive Smooth-L1 bbox
heartbeat            every 50 TRAIN batches
```

Expected optimizer steps with the frozen D10 surface:

```text
ceil(9840 / 32) * 8 = 2,464 barline
ceil(9840 / 32) * 8 = 2,464 meter
TOTAL                 4,928
```

VALIDATION is read-only. An optimizer call with any split other than `train` fails closed.

## Validation metrics and frozen acceptance gates

The D9 gates are reused without adjustment after seeing D11 results:

```text
barline strict Dice                     >= 0.500
barline 2px tolerant F1                  >= 0.700
meter four-class macro F1                >= 0.800
meter positive localization 2px F1       >= 0.600
```

A completed training run may persist evidence even when one or more thresholds fail. In that case `acceptance_passed=false`; TEST remains sealed and the stage is not eligible for closure/merge on quality grounds.

## Safety boundary

- TEST records are forbidden and rejected before any non-split field is touched.
- D7 Structure core is not instantiated or loaded.
- Only the new barline/meter refiner parameters enter optimizers.
- Synthetic D10 targets remain deterministic derivatives of accepted D6 renderer geometry; model predictions never become ground truth.
- Repository and pinned runtime identities are checked before and after training.
- Run output must be fresh and outside normal Git.
- Checkpoints are safely reloaded with `weights_only=True` and both state hashes are reproduced before `COMPLETE`.
- A failed D11 validation gate does not authorize TEST access or post-hoc threshold changes.

## Closure gate

D11 can close only after:

1. focused tests + full regression + exact-head CI pass;
2. independent P1/P2 review has no blocker;
3. accepted D10 manifest/artifact-binding identities are explicitly frozen for the external run;
4. authoritative external training completes with exactly 2,464 optimizer steps per refiner;
5. checkpoint/metrics/verification hashes are recorded outside Git;
6. all four frozen D9 validation thresholds pass;
7. TEST remains unopened;
8. explicit merge approval is obtained.
