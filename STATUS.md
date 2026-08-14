# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D9 work:

- `main`: `afa5b71d1dd94803a1037322b7dfe3d13135a711`
- PR #45 — Stage 7-D8 Structure validation diagnostics: MERGED
- D8 post-merge main CI: run #177 (`31838215784`) — SUCCESS
- Stage 7-D0 through Stage 7-D8: CLOSED

The current active lane is **Stage 7-D9 — Structure refinement contract**. D9 is declarative only. It freezes the D8-selected internal decomposition and acceptance gates before any new optimizer run. TEST remains sealed.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0–6 | Deterministic music → validated synthetic dataset pipeline | ✅ Closed / main CI verified |
| 7-A | Training contract freeze | ✅ Closed / main CI verified |
| 7-B | Tokenizer/data/model/trainer implementation | ✅ Closed / main CI verified |
| 7-C | Bounded baseline training + evidence | ✅ Closed / non-production baseline |
| 7-D0 | Synthetic Curriculum v1 export-evidence identity gate | ✅ Closed |
| 7-D1 | Synthetic corpus transport/byte/manifest acceptance | ✅ Closed |
| 7-D2 | Synthetic V1 monolithic train/validation execution | ✅ Closed / non-production baseline |
| 7-D3 | Validation-only semantic error diagnostics | ✅ Closed / specialist decomposition selected |
| 7-D4 | Specialist OMR architecture + GT/fusion contract | ✅ Closed |
| 7-D5 | StaffSet + StructureSet deterministic geometry | ✅ Closed |
| 7-D6 | TRAIN/VALIDATION StaffSet + StructureSet derivatives | ✅ Closed / PR #43 / main CI #171 PASS |
| 7-D7 | StaffSet + StructureSet specialist training | ✅ Closed / PR #44 / main CI #175 PASS |
| 7-D8 | Structure validation-only diagnostics | ✅ Closed / PR #45 / main CI #177 PASS |
| 7-D9 | Structure refinement architecture/contract | 🔄 Active — no training / TEST sealed |
| 8-0 | Real-data rights/provenance/fine-tuning contract | ✅ Closed / preserved |
| 8-1 | Real-data quarantine/intake + byte validation | ✅ Closed / preserved |
| 8-2 | Paired experiment profile | ✅ Closed / preserved |
| 8-3A | Real pilot preparation/admission components | ⏸ Parked during specialist synthetic work |
| 8-3B | Paired real train/validation execution | 🔒 Not started |
| 9 | Sealed benchmark and candidate decision | 🔒 Not started — TEST sealed |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Frozen Synthetic Curriculum v1

```text
source commit       adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90
config fingerprint  154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb
build id            d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352
manifest SHA-256     44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34
transport SHA-256    4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b
families             512 = 410 train + 51 validation + 51 test
images               1536 = 1230 train + 153 validation + 153 test
targets              512 MusicXML
```

## Accepted D2 / D3 finding

D2 proved the monolithic execution path but not usable OMR recognition: best validation loss `0.9379074645594616`, exact sequence accuracy `0.0`, TER `0.847364947676063`, semantic/MusicXML validity `1.0`, TEST development exposure `0`.

D3 diagnosed the accepted D2 checkpoint on 153 validation images with zero optimizer steps and zero TEST exposure. Key results included pitch identity `0.0`, duration accuracy about `0.1842`, chord-size accuracy `0.0`, rest recognition about `0.6563`, and TER about `0.8474`. This rejected a simple “more epochs” response and selected specialist musical-task decomposition.

## Accepted specialist architecture

```text
StaffSet      -> staff geometry
StructureSet  -> system / measure / barline / G2 / meter
NoteHeadSet   -> note-head center / bbox / fill
RestSet       -> supported rest glyph + duration
AccidentalSet -> sharp / flat / natural glyphs
RhythmSet     -> stem / beam / flag / duration
PitchSet      -> staff position only
ChordSet      -> 2–4 note vertical grouping
ContextSet    -> deterministic musical validation
```

Absolute pitch is not authoritative learned output in V1. It is resolved deterministically from `G2 + staff position + accidental state`.

## Accepted D5 geometry

D5 established synthetic spatial GT using pinned Verovio 6.2.1 and deterministic final-PNG coordinate replay. The accepted `stage7d5-staff-structure-geometry-v2` contract excludes post-barline courtesy meter signatures from the current measure and uses `barline_segment` for rotation-safe geometry.

## Accepted D6 derivative evidence

```text
TRAIN       410 families / 1,230 PNG → 1,230 label sidecars
VALIDATION   51 families /   153 PNG →   153 label sidecars
TEST         51 families /   153 PNG →     0 specialist labels
```

```text
derivative build ID      0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519
manifest SHA-256          e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9
artifact binding SHA-256  3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1
receipt SHA-256           8fe85747b77f2282be3662f0c3d180a440c88028638bf1df7ddadfbb7650fff2
label count               1,383
family count                461
TEST specialist records       0
```

## Accepted D7 training evidence

Authoritative external D7 identity:

```text
run ID                 4ce2903206c7965471bb9569d379d8d9d1022d9248d80886638acfe0bd822598
D7 repository head     25bdf2b3146faba54a93c00f05537f522c75b532
profile fingerprint    7b7fbc79c748da0f1195bc9273fe012e0b1128b3a1e491bb484653d47cb5201a
checkpoint SHA-256     5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc
metrics SHA-256        43cd98a75c2db740b4af6ee3c8826122fa387347820d2e7d2c639ac2fe30f792
verification SHA-256   cdc0733af1bd6c7336f5bd2a0cb12fcae269120d8b5a9a564f08db860ee21a0a
TEST opened            false
```

Staff result:

```text
best validation loss       0.11952157418888348
staff_lines Dice            0.9216719705324906
staff_region Dice           0.9126690970017359
optimizer steps             1,640
```

Structure result:

```text
best validation loss       0.49127569106908947
system_region Dice          0.93046746804164
measure_region Dice         0.8445145579484793
clef_g2 Dice                0.8228637140530807
barline Dice                0.2667824041384917
meter_2_4 Dice              0.34398488560691476
meter_3_4 Dice              0.34151152062874574
meter_4_4 Dice              0.3092358358777486
optimizer steps             1,640
```

D7 closed as a successful first specialist training stage, but barline/meter were explicitly unresolved.

## Accepted D8 diagnostic evidence

D8 reloaded the exact accepted D7 checkpoint, reproduced the accepted Structure validation baseline, and performed only VALIDATION inference.

```text
D8 repository head       e0e721bf5a6d13025546fdf5eeb755647eef383f
report SHA-256            46de5f6766f78bb567f70794a364ccd44835d09af94ef29c3f1eab5cd13ce968
baseline validation loss  0.49127569106908947
TRAIN tensors             0
VALIDATION tensors        153 / 51 families
TEST records              0
optimizer steps           0
model mutated             false
TEST opened               false
```

Key D8 results:

| channel | Dice@0.50 | best threshold | best Dice | threshold gain | tolerant F1 @2px |
|---|---:|---:|---:|---:|---:|
| system_region | 0.9304226398 | 0.25 | 0.9339324257 | 0.0035097860 | 0.9898584521 |
| measure_region | 0.8449312699 | 0.45 | 0.8455369242 | 0.0006056543 | 0.9793445289 |
| barline | 0.2736204205 | 0.35 | 0.2749698120 | 0.0013493914 | 0.3670878904 |
| clef_g2 | 0.8286431574 | 0.30 | 0.8298937531 | 0.0012505957 | 0.9441274383 |
| meter_2_4 | 0.3481060606 | 0.50 | 0.3481060606 | 0.0 | 0.3892495018 |
| meter_3_4 | 0.3528485803 | 0.70 | 0.3557100829 | 0.0028615027 | 0.4025582667 |
| meter_4_4 | 0.3103351169 | 0.55 | 0.3109830753 | 0.0006479584 | 0.3547076746 |

D8 rejected a simple threshold/calibration response. The weak channels occupy only about `0.00066–0.00104` of page pixels, and even 2px tolerance leaves them around `0.35–0.40` F1. The accepted interpretation is sparse-object/representation pressure in the shared whole-page Structure segmentation model.

## Active D9 boundary

D9 keeps the external D4 `StructureSet` contract unchanged while freezing this internal decomposition before training:

```text
accepted D7 Structure core
  ├─ system_region  -> frozen
  ├─ measure_region -> frozen
  └─ clef_g2        -> frozen

barline_refiner
  └─ high-resolution measure-end ROI -> barline_segment + confidence

meter_refiner
  └─ measure-start ROI -> none|2/4|3/4|4/4 + meter_bbox + confidence

structure_fusion
  └─ deterministic fail-closed fusion to unchanged StructureSet outputs
```

Frozen local ROI policies:

```text
barline  measure-end   192x128  aspect-preserving fit/pad
meter    measure-start 192x256  aspect-preserving fit/pad
```

The accepted D7 Structure core is not retrained in the first refinement run. Only new local refiner weights may later reach an optimizer.

Frozen pre-training validation gates:

```text
TEST records                                  0
accepted D7 core mutation                     forbidden
new trainable parameters                      <= 1,250,000
barline strict Dice                           >= 0.500
barline tolerant F1 @2px                      >= 0.700
meter none|2/4|3/4|4/4 macro F1              >= 0.800
meter positive localization tolerant F1 @2px  >= 0.600
```

D9 itself has no model/trainer/checkpoint execution path and authorizes no optimizer run.

## Safety boundaries

- No direct commits to `main`; changes use branch/PR packages.
- Large corpus/checkpoint artifacts stay outside normal Git.
- D9 is declarative only and performs no training.
- VALIDATION remains read-only for future refinement selection.
- TEST remains sealed until Stage 9.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain preserved and parked.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Real geometry labels require human-verified annotation and explicit admission.
- No online or automatic learning path is allowed.
- Deterministic validators retain veto authority over learned specialist candidates.

## Next gate

Close D9 contract/review/CI on one exact branch head. Only after D9 merge may the next package implement deterministic ROI derivatives and bounded barline/meter training under this frozen contract. TEST must remain sealed.

See `STAGE7D9_STRUCTURE_REFINEMENT_CONTRACT.md` for the active contract.
