# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D4 work:

- `main`: `168c03755f0e06e8042fc0a391a357c71c6288fe`
- PR #40 — Stage 7-D3 validation error diagnostics: MERGED
- post-merge main CI: run #146 (`31797006780`) — SUCCESS
- post-merge regression: 483/483 PASS
- Stage 7-D0: CLOSED
- Stage 7-D1: CLOSED
- Stage 7-D2: CLOSED
- Stage 7-D3: CLOSED

The current active lane is **Stage 7-D4 — Specialist OMR Architecture Contract**. D4 performs no training and opens no TEST data. It freezes the decomposition into small visual/musical specialist tasks plus deterministic fusion/context validation.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0–6 | Deterministic music → validated synthetic dataset pipeline | ✅ Closed / main CI verified |
| 7-A | Training contract freeze | ✅ Closed / main CI verified |
| 7-B | Tokenizer/data/model/trainer implementation | ✅ Closed / main CI verified |
| 7-C | Bounded baseline training + evidence | ✅ Closed / non-production baseline |
| 7-D0 | Synthetic Curriculum v1 export-evidence identity gate | ✅ Closed |
| 7-D1 | Synthetic corpus transport/byte/manifest acceptance | ✅ Closed |
| 7-D2 | Synthetic V1 train/validation execution | ✅ Closed / non-production baseline |
| 7-D3 | Validation-only semantic error diagnostics | ✅ Closed / PR #40 merged / main CI #146 PASS |
| 7-D4 | Specialist OMR architecture + ground-truth/fusion contract | 🔄 Active — no training / TEST sealed |
| 8-0 | Real-data rights/provenance/fine-tuning contract | ✅ Closed / preserved |
| 8-1 | Real-data quarantine/intake + byte validation | ✅ Closed / preserved |
| 8-2 | Paired experiment profile | ✅ Closed / preserved |
| 8-3A | Real pilot preparation/admission components | ⏸ Parked during specialist synthetic architecture work |
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

## Accepted Stage 7-D2 result

```text
run id                 14d63841254c03463ad76bbed83df95045742c23f71ad91d7b0c5dc19495a373
checkpoint SHA-256     239cf3dbdf80235bfc7e4a68fe5fecc03e8cd6fefc8a9ff6e27a2ca879ed5291
checkpoint state SHA   466cefcd40887cb0578b7bbc87c6a1b5f676dc0272ab5eee1142e45e7da8e17d
metrics SHA-256        e80b8aed13cc8c7aafae283f4306f1f60821fbf75faaaf568ddff7b132c318bd
verification SHA-256   6743425d42da77dfacef50388e879d45aa01f01b740cfd2deb381a55436500c3
best epoch             20
untrained val loss     3.5657773256519163
best val loss          0.9379074645594616
exact sequence         0.0
token error rate       0.847364947676063
semantic validity      1.0
MusicXML validity      1.0
TEST development use   0
```

The D2 model is a diagnostic baseline, not a production OMR candidate.

## Accepted Stage 7-D3 diagnosis

Authoritative validation-only D3 execution:

```text
run id                    22b7d63f5112fb9d41fa72d502c7a3648781d692949bedf5fbbad8142e910ab7
diagnostics SHA-256       b5843f896a2f75f8c0b111a8d1dd562a74b15cf67d48c0d4e1dfa8655ed41a6b
verification SHA-256      558fb0a6e0bfe7e7f461361773a9f8a08b48c5dc4613bd1a3d3a73da7e5186e9
validation samples        153 / 51 families
TRAIN diagnostic exposure 0
TEST diagnostic exposure  0
optimizer steps            0
pitch identity accuracy    0.0
duration accuracy          0.1842105263
rest recognition accuracy  0.6563467492
chord-size accuracy        0.0
token error rate           0.8473649477
```

The error map shows broad representation failure across pitch, rhythm, event typing, chord grouping and event completeness. Clean/light/medium derivatives were diagnostically identical family-by-family. D3 therefore rejects a simple "more epochs" response and selects **specialist musical-task decomposition** as the next architecture axis.

## Stage 7-D4 boundary

D4 freezes the V1 expert/task graph and the ground-truth authority rules:

```text
StaffSet      -> staff geometry
StructureSet  -> system / measure / barline / G2 / meter
NoteHeadSet   -> note-head center / bbox / fill
RestSet       -> supported rest glyph + duration
AccidentalSet -> sharp / flat / natural glyphs
RhythmSet     -> stem / beam / flag / duration
PitchSet      -> staff position only
ChordSet      -> 2-4 note vertical grouping
ContextSet    -> deterministic musical validation
```

Absolute pitch is **not** authoritative learned output in V1. It is resolved deterministically from `G2 + staff position + accidental state`.

Synthetic symbolic ground truth comes from canonical music. Synthetic spatial ground truth must come from pinned renderer geometry plus an exact deterministic raster/degradation coordinate transform. Real spatial ground truth requires independently human-verified annotation; an admitted image+MusicXML pair alone is insufficient for spatial specialist training.

## Safety boundaries

- No direct commits to `main`; changes use small branch/PR packages.
- Large corpus/checkpoint artifacts stay outside normal Git.
- D4 contains no model, trainer, optimizer, checkpoint loader or TEST evaluator.
- Specialist derivatives inherit the source family split.
- TEST specialist labels are not derived during development and TEST remains sealed until Stage 9.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain intact and parked.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Real geometry labels require human-verified annotation and explicit admission.
- No online or automatic learning path is allowed.
- Deterministic validators retain veto authority over learned specialist candidates.

## Next gate

Pass focused/full CI and independent review for Stage 7-D4. After D4 merge, the first implementation package is **StaffSet + StructureSet ground-truth extraction/pilot**. It must prove deterministic renderer-to-raster geometry labels before any specialist model training begins.