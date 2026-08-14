# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage architecture remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D5 work:

- `main`: `96dbb720df7845baebc980518180b8dd9183776b`
- PR #41 — Stage 7-D4 specialist OMR architecture contract: MERGED
- post-merge main CI: run #151 (`31801492807`) — SUCCESS
- post-merge regression: 499/499 PASS
- Stage 7-D0: CLOSED
- Stage 7-D1: CLOSED
- Stage 7-D2: CLOSED
- Stage 7-D3: CLOSED
- Stage 7-D4: CLOSED

The current active lane is **Stage 7-D5 — StaffSet + StructureSet deterministic geometry pilot**. D5 performs no training and opens no TEST data. It proves that synthetic spatial ground truth can be extracted from the pinned Verovio layout and mapped deterministically into the exact final PNG coordinate space.

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
| 7-D4 | Specialist OMR architecture + ground-truth/fusion contract | ✅ Closed / PR #41 merged / main CI #151 PASS |
| 7-D5 | StaffSet + StructureSet deterministic geometry pilot | 🔄 Active — no training / TEST sealed |
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

D4 froze the V1 expert/task graph and the ground-truth authority rules:

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

## Stage 7-D5 geometry proof

D5 implements a secondary, separately fingerprinted geometry render of the same validated MusicXML using the same pinned Verovio 6.2.1 layout plus only invisible `svgBoundingBoxes` and `svgContentBoundingBoxes` instrumentation. The frozen Stage-3 renderer defaults/fingerprint are not modified.

The pilot extracts:

- one `StaffSet` graphical instance per rendered system in V1;
- exactly five staff-line segments and staff spacing;
- system and measure bounding boxes;
- trailing barline segments;
- visible G2-clef and meter bounding boxes when present;
- canonical measure number and meter class from independently parsed MusicXML semantics.

The pinned Verovio SVG uses a nested `class="definition-scale"` coordinate space plus ancestor transforms such as the page-margin translation. D5 resolves that coordinate space explicitly, applies the required SVG transforms, then maps geometry through CairoSVG scaling and the exact Pillow 12.3.0 `rotate(..., expand=True)` geometry used by controlled degradation. Photometric degradation does not alter coordinates.

A D4 representation issue was found and corrected without rewriting the historical D4 fingerprint: scalar `barline_x` is superseded operationally by `barline_segment`, because a rotated final PNG makes a barline non-vertical.

Pre-documentation exact-head CI #156 (`31804989648`) passed **514/514 tests**, including all six Stage-2 golden MusicXML live geometry/raster-equivalence tests and clean/light/medium final-coordinate mapping. Final exact-head CI is required again after this documentation sync.

## Safety boundaries

- No direct commits to `main`; changes use small branch/PR packages.
- Large corpus/checkpoint artifacts stay outside normal Git.
- D5 contains no model, trainer, optimizer, checkpoint loader or TEST evaluator.
- D5 has no dataset split loader; its live proof uses repository golden fixtures only.
- Specialist derivatives inherit the source family split when a later corpus builder is introduced.
- TEST specialist labels are not derived during development and TEST remains sealed until Stage 9.
- Existing Stage 8 rights/provenance/privacy/duplicate/leakage controls remain intact and parked.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Real geometry labels require human-verified annotation and explicit admission.
- No online or automatic learning path is allowed.
- Deterministic validators retain veto authority over learned specialist candidates.

## Next gate

Pass final exact-head CI and independent review for Stage 7-D5, then obtain explicit merge approval. After D5 closes, the next small package should build and independently validate TRAIN/VALIDATION-only `StaffSet` + `StructureSet` specialist derivatives from the frozen Synthetic Curriculum; TEST remains sealed and no specialist model training starts until that derivative-data gate is accepted.