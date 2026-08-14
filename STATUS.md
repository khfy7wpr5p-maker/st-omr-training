# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D8 work:

- `main`: `8d5e9e32cb9a96de08d81d6f62fbd6deee18df83`
- PR #44 — Stage 7-D7 StaffSet + StructureSet specialist training: MERGED
- D7 post-merge main CI: run #175 (`31836565453`) — SUCCESS
- D7 post-merge regression: 531/531 PASS
- Stage 7-D0 through Stage 7-D7: CLOSED

The current active lane is **Stage 7-D8 — Structure validation diagnostics**. D8 does not train a model. It binds the exact accepted D7 external checkpoint and diagnoses the Structure specialist on VALIDATION only so that weak barline/meter channels are understood before any refinement is selected. TEST remains sealed.

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
| 7-D8 | Structure validation-only diagnostics | 🔄 Active — optimizer 0 / TEST sealed |
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

D6 materialized only TRAIN and VALIDATION specialist labels:

```text
TRAIN       410 families / 1,230 PNG → 1,230 label sidecars
VALIDATION   51 families /   153 PNG →   153 label sidecars
TEST         51 families /   153 PNG →     0 specialist labels
```

Authoritative D6 identities:

```text
derivative build ID      0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519
manifest SHA-256          e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9
artifact binding SHA-256  3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1
receipt SHA-256           8fe85747b77f2282be3662f0c3d180a440c88028638bf1df7ddadfbb7650fff2
label count               1,383
family count                461
TEST specialist records       0
```

No Staff/Structure model was trained in D6.

## Accepted D7 training evidence

D7 trained two independent dense-geometry specialists using the same accepted D6 development surface. TRAIN alone reached optimizers; VALIDATION remained read-only; TEST remained sealed.

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
untrained validation loss  1.5256422758102417
best validation loss       0.11952157418888348
best epoch                  8
optimizer steps             1,640
staff_lines Dice            0.9216719705324906
staff_region Dice           0.9126690970017359
state SHA-256               3131548548521229e6acd6fee8cffc66081cb54125645f9eff5a488de7603af8
```

Structure result:

```text
untrained validation loss  1.7060188742784352
best validation loss       0.49127569106908947
best epoch                  8
optimizer steps             1,640
system_region Dice          0.93046746804164
measure_region Dice         0.8445145579484793
clef_g2 Dice                0.8228637140530807
barline Dice                0.2667824041384917
meter_2_4 Dice              0.34398488560691476
meter_3_4 Dice              0.34151152062874574
meter_4_4 Dice              0.3092358358777486
state SHA-256               0d11b2ae414959b678ccc22a6b8cfcc1edc1ecadc3c73ed6ab5a0cda6e593907
```

D7 therefore closed as a successful first specialist training stage, but barline/meter channels are explicitly not treated as solved.

## Active D8 boundary

D8 is diagnostic only. It must first safely reload the exact accepted D7 checkpoint and reproduce the exact accepted Structure validation loss and seven channel Dice values. Only then may it compute additional VALIDATION diagnostics.

Frozen D8 diagnostic surface:

```text
TRAIN tensor records       0
VALIDATION tensor records 153 / 51 families
TEST records               0
optimizer steps            0
model mutation             false
```

D8 measures every Structure channel with:

- global threshold sweep `0.05 ... 0.95`;
- exact `0.50` precision/recall/Dice;
- deterministic best diagnostic threshold;
- positive-record and positive-pixel prevalence;
- predicted probability separation on positive vs negative target pixels;
- 1-pixel and 2-pixel tolerant localization F1 at `0.50` and at the diagnostic best threshold.

The purpose is to distinguish threshold/calibration error, thin-object near-miss localization, and sparse/representation limitations **before** changing loss, target rasterization, crop strategy, channel decomposition, epochs, or architecture.

D8 writes only a canonical hash-addressed diagnostic report plus `COMPLETE` outside normal Git. It creates no checkpoint.

## Safety boundaries

- No direct commits to `main`; changes use branch/PR packages.
- Large corpus/checkpoint artifacts stay outside normal Git.
- D8 constructs no optimizer and performs no backward pass.
- VALIDATION is read-only and cannot mutate model weights.
- TEST remains sealed until Stage 9.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain preserved and parked.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Real geometry labels require human-verified annotation and explicit admission.
- No online or automatic learning path is allowed.
- Deterministic validators retain veto authority over learned specialist candidates.

## Next gate

Close D8 code/review/CI on one exact branch head, then run the authoritative validation-only diagnostic outside Git against the exact accepted D7 artifact bundle. Interpret the D8 report before selecting any Structure refinement. TEST is not opened for that decision.

See `STAGE7D8_STRUCTURE_DIAGNOSTICS.md` for the exact active diagnostic contract.
