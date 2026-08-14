# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D7 work:

- `main`: `be8177bd294f6554e558f8385d2e00d89bc9dede`
- PR #43 — Stage 7-D6 StaffSet + StructureSet specialist derivatives: MERGED
- D6 post-merge main CI: run #171 (`31821595254`) — SUCCESS
- D6 post-merge regression: 522/522 PASS
- Stage 7-D0 through Stage 7-D6: CLOSED

The current active lane is **Stage 7-D7 — StaffSet + StructureSet specialist training**. D7 is the first real specialist-model training stage. It consumes only the accepted D6 TRAIN/VALIDATION derivative set, trains two independent models, keeps VALIDATION read-only, and keeps TEST sealed.

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
| 7-D7 | StaffSet + StructureSet specialist training | 🔄 Active — TEST sealed |
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

The authoritative external build and second independent persisted-output gate both passed before PR #43 merged. No Staff/Structure model was trained in D6.

## Active D7 boundary

D7 trains two task-isolated models:

### Staff specialist

Dense targets:

- `staff_lines`
- `staff_region`

### Structure specialist

Dense targets:

- `system_region`
- `measure_region`
- `barline`
- `clef_g2`
- `meter_2_4`
- `meter_3_4`
- `meter_4_4`

Both consume resized/inverted `96 × 512` grayscale images and deterministic rasterizations of the accepted D6 final-PNG geometry. They do not share weights or optimizers.

Frozen D7 profile:

```text
TRAIN samples       1,230 / optimizer allowed
VALIDATION samples    153 / read-only
TEST records             0 / forbidden
batch size                6
epochs                     8 per specialist
optimizer              AdamW
learning rate          0.0007
weight decay           0.0001
grad clip              1.0
objective              BCE-with-logits + soft Dice
checkpoint selection   minimum validation loss per specialist
runtime                pinned deterministic CPU PyTorch
```

D7 reruns the independent D6 verifier before loading training records and requires the exact accepted D6 manifest/build/artifact identities. A TEST record fails immediately after reading only `split`, before D7 image/label path derivation.

Each authoritative D7 run stays outside normal Git and writes hash-addressed checkpoint, metrics, verification and COMPLETE artifacts. Staff and Structure checkpoint states must reload safely with `torch.load(..., weights_only=True)` and reproduce their exact model-state hashes.

## Safety boundaries

- No direct commits to `main`; changes use branch/PR packages.
- Large corpus/checkpoint artifacts stay outside normal Git.
- TRAIN only can execute optimizer steps in D7.
- VALIDATION is read-only and cannot mutate model weights.
- TEST remains sealed until Stage 9.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain preserved and parked.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Real geometry labels require human-verified annotation and explicit admission.
- No online or automatic learning path is allowed.
- Deterministic validators retain veto authority over learned specialist candidates.

## Next gate

Close D7 code/review/CI on one exact branch head, then run the authoritative Staff/Structure training outside Git against the accepted D6 derivatives. D7 closes only after checkpoint, metrics and verification hashes are independently accepted. TEST is not opened to decide whether the Staff/Structure specialists are good enough to continue.

See `STAGE7D7_STAFF_STRUCTURE_TRAINING.md` for the exact active training contract.
